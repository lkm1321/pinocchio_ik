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

The capnp schema (``sdf_query.capnp``) is searched for in this order:

1. ``schema_path`` constructor argument
2. ``ERL_GP_SDF_ROS_CAPNP_SCHEMA`` environment variable
3. ``share/<pkg>/capnp/sdf_query.capnp`` under each ``AMENT_PREFIX_PATH``
   entry, for ``pinocchio_ik`` (bundled copy) then ``erl_gp_sdf_ros``
4. ``../capnp/sdf_query.capnp`` relative to this file (source-tree fallback)
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import capnp  # type: ignore
import numpy as np


# pycapnp's TwoPartyClient uses a background reactor thread; the threaded
# event loop must be installed before any client is created. Both calls are
# safe to repeat (remove_event_loop(True) is a no-op when none exists), but
# we guard with a flag so multiple clients in the same process share one loop.
_EVENT_LOOP_READY = False


def _init_event_loop() -> None:
    global _EVENT_LOOP_READY
    if _EVENT_LOOP_READY:
        return
    capnp.remove_event_loop(True)
    capnp.create_event_loop(threaded=True)
    _EVENT_LOOP_READY = True


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
        _init_event_loop()

        self._host = str(host)
        self._port = int(port)
        self._timeout_sec = float(timeout_sec)
        self._logger = logger

        self._schema = capnp.load(str(_find_schema(schema_path)))
        self._client = None
        self._mapper = None
        # Serialize calls: pycapnp Promises are tied to the reactor thread but
        # the request builders / RPC state aren't safe to share across CBF
        # threads at random. The CBF runs at 100Hz from one timer callback,
        # so contention is effectively zero.
        self._lock = threading.Lock()

        # Per-call latency stats; reset on each periodic log emission.
        self._call_count = 0
        self._call_points = 0
        self._call_total = 0.0
        self._call_wait = 0.0
        self._call_max = 0.0
        self._diag_window_s = 2.0
        self._last_log_t = time.monotonic()

        if wait_for_server_sec > 0:
            self._wait_for_server(wait_for_server_sec)

    def _address(self):
        return f'{self._host}:{self._port}'

    def _probe_tcp(self, timeout_sec):
        """Open and close a TCP connection to confirm the server is listening.

        ``TwoPartyClient`` doesn't fail synchronously on a closed port — the
        first RPC is where the failure surfaces — so a plain socket probe is
        the cleanest way to bound startup wait time.
        """
        with socket.create_connection(
            (self._host, self._port), timeout=timeout_sec
        ):
            pass

    def _connect(self):
        self._client = capnp.TwoPartyClient(self._address())
        self._mapper = self._client.bootstrap().cast_as(
            self._schema.SdfMapper
        )

    def _drop_connection(self):
        self._mapper = None
        self._client = None

    def _ensure_connected(self):
        if self._mapper is None:
            self._connect()

    def _wait_for_server(self, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        last_err = None
        attempt = 0
        backoff = 0.2
        while True:
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

    @staticmethod
    def _wait_promise(promise, timeout_sec):
        """Block on a capnp Promise with a soft timeout via polling.

        pycapnp's ``Promise.wait()`` has no timeout; ``Promise.poll()`` lets
        us bound how long we block. Polling cadence ramps from 1ms → 50ms so
        latency stays low for fast queries without burning CPU on slow ones.
        """
        deadline = time.monotonic() + timeout_sec
        interval = 0.001
        while True:
            if promise.poll():
                return promise.wait()
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f'capnp SDF query did not complete within {timeout_sec:.2f}s'
                )
            time.sleep(min(interval, deadline - now))
            interval = min(interval * 1.5, 0.05)

    def __call__(self, query_points):
        t_enter = time.monotonic()
        query_points = np.asarray(query_points, dtype=np.float64)
        single = query_points.ndim == 1
        if single:
            query_points = query_points[np.newaxis, :]
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(
                f'query_points must be (N,3) or (3,); got {query_points.shape}'
            )

        n = query_points.shape[0]

        with self._lock:
            try:
                self._ensure_connected()
                req = self._schema.SdfQueryRequest.new_message()
                pts = req.init('queryPoints', n)
                for i, p in enumerate(query_points):
                    pts[i].x = float(p[0])
                    pts[i].y = float(p[1])
                    pts[i].z = float(p[2])

                t_sent = time.monotonic()
                promise = self._mapper.querySdf(request=req)
                try:
                    result = self._wait_promise(promise, self._timeout_sec)
                except TimeoutError:
                    self._drop_connection()
                    raise
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
            # — far positive distance, zero gradient, no constraint.
            if not getattr(self, '_warned_empty', False):
                if self._logger is not None:
                    self._logger.warn(
                        f"capnp SDF '{self._address()}' returned "
                        f"success=False (no map yet); treating as "
                        f"'no obstacles'."
                    )
                self._warned_empty = True
            distances = np.full(n, 1e3, dtype=np.float64)
            gradients = np.zeros((n, 3), dtype=np.float64)
            if single:
                return distances[0], gradients[0]
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

        if single:
            return distances[0], gradients[0]
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
