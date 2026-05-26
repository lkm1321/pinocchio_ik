"""Cap'n Proto SDF client that matches the ``PointCloudSDF`` calling contract.

``CapnpSdfClient(query_points)`` returns ``(distances, gradients)`` so it can
be dropped in wherever ``PointCloudSDF`` is accepted (e.g. as the ``env_sdf``
of ``PinocchioFKCBF`` with ``env_grad=None``).

This is the Cap'n Proto RPC counterpart of the older ``ros_sdf_client``: same
calling contract, but it talks to the capnp RPC server hosted by
``erl_gp_sdf_ros``'s ``sdf_mapping_node`` (enabled via ``capnp_rpc.enabled:
true`` in the node's YAML config) over a plain TCP socket instead of a ROS2
service. Avoids the DDS round-trip and the cross-image ROS msg dependency
(``erl_gp_sdf_msgs`` is no longer needed on this side).

Threading model
---------------
pycapnp 1.x removed the old ``create_event_loop(threaded=True)`` reactor and
runs the KJ event loop on whatever thread calls ``Promise.wait()``. Because
the loop and the ``TwoPartyClient`` are bound to a single thread, capnp RPC
cannot be driven directly from ROS executor threads without hanging them.

This client therefore owns one dedicated reactor thread. *All* capnp objects
(loop, client, mapper, request builders) live on that thread; the CBF callback
submits a query and blocks on a ``concurrent.futures.Future`` with a real
timeout. A hung RPC stalls a single query, never a ROS executor thread.

The capnp schema (``sdf_query.capnp``) is searched for in this order:

1. ``schema_path`` constructor argument
2. ``ERL_GP_SDF_ROS_CAPNP_SCHEMA`` environment variable
3. ``share/<pkg>/capnp/sdf_query.capnp`` under each ``AMENT_PREFIX_PATH``
   entry, for ``pinocchio_ik`` (bundled copy) then ``erl_gp_sdf_ros``
4. ``../capnp/sdf_query.capnp`` relative to this file (source-tree fallback)
"""

from __future__ import annotations

import concurrent.futures
import os
import queue
import socket
import threading
import time
from pathlib import Path

import capnp  # type: ignore
import numpy as np


def _find_schema(explicit):
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get('ERL_GP_SDF_ROS_CAPNP_SCHEMA')
    if env:
        return Path(env).resolve()
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        if not prefix:
            continue
        for pkg in ('pinocchio_ik', 'erl_gp_sdf_ros'):
            candidate = (
                Path(prefix) / 'share' / pkg / 'capnp' / 'sdf_query.capnp'
            )
            if candidate.is_file():
                return candidate.resolve()
    fallback = (
        Path(__file__).resolve().parent.parent / 'capnp' / 'sdf_query.capnp'
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        'Could not locate sdf_query.capnp. Pass schema_path, set '
        'ERL_GP_SDF_ROS_CAPNP_SCHEMA, or source the colcon install.'
    )


class CapnpSdfClient:
    def __init__(
        self,
        host='127.0.0.1',
        port=51111,
        timeout_sec=2.0,
        wait_for_server_sec=5.0,
        schema_path=None,
        logger=None,
    ):
        self._host = str(host)
        self._port = int(port)
        self._timeout_sec = float(timeout_sec)
        self._wait_for_server_sec = float(wait_for_server_sec)
        self._logger = logger

        # Schema loading is thread-agnostic, so it can happen here.
        self._schema = capnp.load(str(_find_schema(schema_path)))

        # capnp connection objects -- ONLY ever touched on the reactor thread.
        self._client = None
        self._mapper = None
        self._warned_empty = False

        # Per-call latency stats; reset on each periodic log emission.
        # Only mutated on the reactor thread, so no lock is needed.
        self._call_count = 0
        self._call_points = 0
        self._call_total = 0.0
        self._call_wait = 0.0
        self._call_max = 0.0
        self._diag_window_s = 2.0
        self._last_log_t = time.monotonic()

        # Reactor plumbing. The queue holds at most one pending request: a
        # stale obstacle distance is worse for the CBF than a skipped cycle,
        # so a newer query evicts an older queued one ("latest wins").
        self._req_q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._reactor, name='capnp_sdf_reactor', daemon=True
        )
        self._thread.start()

        # Block construction until the reactor has attempted its first
        # connect, mirroring the old wait-for-server behaviour. A failed
        # connect is non-fatal -- calls retry on demand -- so we only wait.
        if not self._ready.wait(timeout=self._wait_for_server_sec + 5.0):
            if self._logger is not None:
                self._logger.warn(
                    'capnp SDF reactor thread did not become ready in time; '
                    'continuing -- calls will retry on demand.'
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, query_points):
        """Submit a query to the reactor thread and block (with timeout)."""
        query_points = np.asarray(query_points, dtype=np.float64)
        single = query_points.ndim == 1
        if single:
            query_points = query_points[np.newaxis, :]
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(
                f'query_points must be (N,3) or (3,); got {query_points.shape}'
            )

        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError('capnp SDF client has been shut down')

        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._submit((fut, query_points, time.monotonic()))

        try:
            distances, gradients = fut.result(timeout=self._timeout_sec)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError(
                f'capnp SDF query did not complete within '
                f'{self._timeout_sec:.2f}s'
            )

        if single:
            return distances[0], gradients[0]
        return distances, gradients

    def shutdown(self):
        """Stop the reactor thread. Safe to call more than once."""
        self._stop.set()
        # Unblock the reactor if it is parked on an empty queue.
        try:
            self._req_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    # Context-manager sugar so callers can `with CapnpSdfClient(...) as c:`.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False

    # ------------------------------------------------------------------
    # Submission helpers
    # ------------------------------------------------------------------
    def _submit(self, item):
        """Enqueue a request, evicting any stale queued request first."""
        while True:
            try:
                self._req_q.put_nowait(item)
                return
            except queue.Full:
                # Drop the older pending request and fail its future so the
                # waiting caller wakes up instead of timing out silently.
                try:
                    stale = self._req_q.get_nowait()
                except queue.Empty:
                    continue
                if stale is not None:
                    stale_fut = stale[0]
                    if not stale_fut.done():
                        stale_fut.set_exception(
                            RuntimeError(
                                'capnp SDF query superseded by a newer request'
                            )
                        )

    # ------------------------------------------------------------------
    # Reactor thread -- everything below runs on self._thread only
    # ------------------------------------------------------------------
    def _reactor(self):
        # First connection attempt happens here so the KJ loop / client are
        # created on this thread. Failure is tolerated; calls retry later.
        try:
            self._wait_for_server(self._wait_for_server_sec)
        except Exception as e:  # noqa: BLE001 -- never let the thread die here
            if self._logger is not None:
                self._logger.warn(f'capnp SDF initial connect failed: {e}')
        finally:
            self._ready.set()

        while not self._stop.is_set():
            try:
                item = self._req_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:  # shutdown sentinel
                break

            fut, query_points, t_enter = item
            if fut.done() or not fut.set_running_or_notify_cancel():
                # Caller already timed out / cancelled -- skip the work.
                continue
            try:
                fut.set_result(self._do_query(query_points, t_enter))
            except BaseException as e:  # noqa: BLE001 -- capnp may raise odd types
                fut.set_exception(e)

        # Drain any stragglers so their callers don't hang on shutdown.
        self._drain_pending('capnp SDF client shutting down')
        self._drop_connection()

    def _drain_pending(self, reason):
        while True:
            try:
                item = self._req_q.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            fut = item[0]
            if not fut.done():
                fut.set_exception(RuntimeError(reason))

    def _address(self):
        return f'{self._host}:{self._port}'

    def _probe_tcp(self, timeout_sec):
        """Open and close a TCP connection to confirm the server is listening.

        ``TwoPartyClient`` doesn't fail synchronously on a closed port -- the
        first RPC is where the failure surfaces -- so a plain socket probe is
        the cleanest way to bound startup wait time.
        """
        with socket.create_connection(
            (self._host, self._port), timeout=timeout_sec
        ):
            pass

    def _connect(self):
        self._client = capnp.TwoPartyClient(self._address())
        self._mapper = self._client.bootstrap().cast_as(self._schema.SdfMapper)

    def _drop_connection(self):
        self._mapper = None
        self._client = None

    def _ensure_connected(self):
        if self._mapper is None:
            self._connect()

    def _wait_for_server(self, timeout_sec):
        if timeout_sec <= 0:
            return
        deadline = time.monotonic() + timeout_sec
        last_err = None
        attempt = 0
        backoff = 0.2
        while not self._stop.is_set():
            attempt += 1
            try:
                remaining = max(0.1, deadline - time.monotonic())
                self._probe_tcp(min(remaining, 1.0))
                self._connect()
                return
            except (OSError, capnp.KjException) as e:
                last_err = e
                self._drop_connection()
                now = time.monotonic()
                if now >= deadline:
                    break
                time.sleep(min(backoff, deadline - now))
                backoff = min(backoff * 1.5, 1.0)
        if self._logger is not None:
            self._logger.warn(
                f"capnp SDF server '{self._address()}' not reachable after "
                f"{timeout_sec:.1f}s ({attempt} attempts): {last_err}; "
                f"calls will retry on demand."
            )

    def _do_query(self, query_points, t_enter):
        """Build, send, and decode one SDF query. Runs on the reactor thread."""
        n = query_points.shape[0]
        try:
            self._ensure_connected()
            req = self._schema.SdfQueryRequest.new_message()
            pts = req.init('queryPoints', n)
            for i, p in enumerate(query_points):
                pts[i].x = float(p[0])
                pts[i].y = float(p[1])
                pts[i].z = float(p[2])

            t_sent = time.monotonic()
            # .wait() runs the KJ event loop on THIS (reactor) thread, which
            # is exactly the thread that created the client -- so it is safe.
            result = self._mapper.querySdf(request=req).wait()
            response = result.response
        except (capnp.KjException, OSError) as e:
            self._drop_connection()
            raise RuntimeError(
                f"capnp SDF query to {self._address()} failed: {e}"
            ) from e

        t_done = time.monotonic()
        self._record_call(t_enter, t_sent, t_done, n)

        if not response.success:
            # Mapping node returns success=False when its map is empty (no
            # scans integrated yet). For the CBF this is "no obstacles known"
            # -- far positive distance, zero gradient, no constraint.
            if not self._warned_empty:
                if self._logger is not None:
                    self._logger.warn(
                        f"capnp SDF '{self._address()}' returned "
                        f"success=False (no map yet); treating as "
                        f"'no obstacles'."
                    )
                self._warned_empty = True
            distances = np.full(n, 1e3, dtype=np.float64)
            gradients = np.zeros((n, 3), dtype=np.float64)
            return distances, gradients

        if not response.computeGradient:
            raise RuntimeError(
                f"capnp SDF server '{self._address()}' is configured with "
                f"compute_gradient=false; the CBF needs gradients. Set "
                f"test_query.compute_gradient: true in the sdf_mapping config."
            )

        distances = np.asarray(response.signedDistances, dtype=np.float64)
        if distances.shape != (n,):
            raise RuntimeError(
                f'capnp SDF returned {distances.shape[0]} distances for '
                f'{n} points'
            )
        if len(response.gradients) != n:
            raise RuntimeError(
                f'capnp SDF returned {len(response.gradients)} gradients for '
                f'{n} points'
            )
        gradients = np.array(
            [[g.x, g.y, g.z] for g in response.gradients], dtype=np.float64
        )
        return distances, gradients

    def _record_call(self, t_enter, t_sent, t_done, n_points):
        self._call_count += 1
        self._call_points += n_points
        dt = t_done - t_enter
        wait = t_done - t_sent
        self._call_total += dt
        self._call_wait += wait
        if dt > self._call_max:
            self._call_max = dt

        now = time.monotonic()
        window = now - self._last_log_t
        if window < self._diag_window_s:
            return

        n = max(self._call_count, 1)
        if self._logger is not None:
            self._logger.info(
                f"capnp_sdf_client[{self._address()}]: "
                f"calls={self._call_count} ({self._call_count / window:.1f}Hz), "
                f"pts/call={self._call_points / n:.1f}, "
                f"mean={self._call_total / n * 1e3:.1f}ms "
                f"(wait={self._call_wait / n * 1e3:.1f}ms), "
                f"max={self._call_max * 1e3:.1f}ms"
            )
        self._last_log_t = now
        self._call_count = 0
        self._call_points = 0
        self._call_total = 0.0
        self._call_wait = 0.0
        self._call_max = 0.0
