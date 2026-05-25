"""Service-call-based SDF that matches the ``PointCloudSDF`` calling contract.

``RosSdfClient(query_points)`` returns ``(distances, gradients)`` so it can be
dropped in wherever ``PointCloudSDF`` is accepted (e.g. as the ``env_sdf`` of
``PinocchioFKCBF`` with ``env_grad=None``).

The service definition lives in another workspace package (default
``erl_gp_sdf_msgs``, srv ``SdfQuery``); the type is loaded lazily so this
module imports cleanly even when that package isn't on the path.
"""

import time
from importlib import import_module
from threading import Event

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


def _executor_spinning(node):
    """True if `node` is owned by an executor (i.e. likely being spun)."""
    try:
        return node.executor is not None
    except AttributeError:
        # Old rclpy without the public attribute — fall back to the private
        # weakref check; if we can't tell, assume yes so we don't spin twice.
        return getattr(node, '_executor_weak', None) is not None


class RosSdfClient:
    def __init__(
        self,
        node,
        service_name='/sdf_query',
        srv_pkg='erl_gp_sdf_msgs',
        srv_type='SdfQuery',
        timeout_sec=2.0,
        wait_for_service_sec=5.0,
        callback_group=None,
    ):
        self._node = node
        self._timeout_sec = float(timeout_sec)
        self._SrvType = getattr(import_module(f'{srv_pkg}.srv'), srv_type)
        # Own callback group so the service response is dispatched on a
        # different executor thread than the caller (required to make the
        # blocking wait below safe under MultiThreadedExecutor).
        self._cb_group = callback_group or MutuallyExclusiveCallbackGroup()
        self._client = node.create_client(
            self._SrvType, service_name, callback_group=self._cb_group,
        )
        if wait_for_service_sec > 0 and not self._client.wait_for_service(
            timeout_sec=wait_for_service_sec
        ):
            node.get_logger().warn(
                f"SDF service '{service_name}' not available after "
                f"{wait_for_service_sec:.1f}s; calls will block until it appears."
            )

        # Per-call latency stats; reset on each periodic log emission.
        self._call_count = 0
        self._call_points = 0
        self._call_total = 0.0   # entry → done
        self._call_wait = 0.0    # call_async → done (network/server)
        self._call_max = 0.0
        self._diag_window_s = 2.0
        self._last_log_t = time.monotonic()

    def __call__(self, query_points):
        t_enter = time.monotonic()
        query_points = np.asarray(query_points, dtype=np.float64)
        single = query_points.ndim == 1
        if single:
            query_points = query_points[np.newaxis, :]
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(
                f"query_points must be (N,3) or (3,); got {query_points.shape}"
            )

        req = self._SrvType.Request()
        req.query_points = [
            Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            for p in query_points
        ]
        # Optional flags in the response-side declaration of SdfQuery.srv —
        # set defensively in case future revisions expose them on the request.
        for flag in ('compute_gradient',):
            if hasattr(req, flag):
                setattr(req, flag, True)

        t_sent = time.monotonic()
        future = self._client.call_async(req)
        # If we're already inside a spinning executor (the steady-state),
        # Event.wait() lets the executor thread dispatch the response. If we
        # are NOT spinning yet — e.g. during DistanceCBFNode construction,
        # which happens before executor.spin() — Event.wait() would deadlock
        # because nothing is processing the client's response. Fall back to
        # manual spin_once on this node so the call can complete either way.
        done = Event()
        future.add_done_callback(lambda _f: done.set())

        deadline = time.monotonic() + self._timeout_sec
        spin_node = self._node if not _executor_spinning(self._node) else None
        while not done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TimeoutError(
                    f"SDF service '{self._client.srv_name}' did not respond within "
                    f"{self._timeout_sec:.2f}s"
                )
            if spin_node is not None:
                # Manual pump while no executor is running yet.
                rclpy.spin_once(spin_node, timeout_sec=min(0.1, remaining))
            else:
                done.wait(min(0.1, remaining))

        t_done = time.monotonic()
        self._record_call(t_enter, t_sent, t_done, len(req.query_points))

        resp = future.result()
        if resp is None:
            raise RuntimeError(
                f"SDF service '{self._client.srv_name}' returned no result"
            )
        n = len(req.query_points)
        if hasattr(resp, 'success') and not resp.success:
            # The mapping node returns success=False when its map is empty
            # (no scans integrated yet). For the CBF this is "no obstacles
            # known" — far positive distance, zero gradient, no constraint.
            # We log once so the operator notices but don't crash.
            if not getattr(self, '_warned_empty', False):
                self._node.get_logger().warn(
                    f"SDF service '{self._client.srv_name}' returned "
                    f"success=False (no map yet); treating as 'no obstacles'."
                )
                self._warned_empty = True
            distances = np.full(n, 1e3, dtype=np.float64)
            gradients = np.zeros((n, 3), dtype=np.float64)
            if single:
                return distances[0], gradients[0]
            return distances, gradients

        distances = np.asarray(resp.signed_distances, dtype=np.float64)
        if distances.shape != (n,):
            raise RuntimeError(
                f"SDF service returned {distances.shape[0]} distances for "
                f"{n} query points"
            )
        if len(resp.gradients) != n:
            raise RuntimeError(
                f"SDF service returned {len(resp.gradients)} gradients for "
                f"{n} query points"
            )
        gradients = np.array(
            [[g.x, g.y, g.z] for g in resp.gradients], dtype=np.float64
        )

        if single:
            return distances[0], gradients[0]
        return distances, gradients

    def _record_call(self, t_enter, t_sent, t_done, n_points):
        """Update rolling latency stats and periodically log a summary."""
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
        self._node.get_logger().info(
            f"sdf_client[{self._client.srv_name}]: "
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
