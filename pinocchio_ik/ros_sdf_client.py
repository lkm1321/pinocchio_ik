"""Service-call-based SDF that matches the ``PointCloudSDF`` calling contract.

``RosSdfClient(query_points)`` returns ``(distances, gradients)`` so it can be
dropped in wherever ``PointCloudSDF`` is accepted (e.g. as the ``env_sdf`` of
``PinocchioFKCBF`` with ``env_grad=None``).

The service definition lives in another workspace package (default
``erl_gp_sdf_msgs``, srv ``SdfQuery``); the type is loaded lazily so this
module imports cleanly even when that package isn't on the path.
"""

from importlib import import_module
from threading import Event

import numpy as np
from geometry_msgs.msg import Vector3
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


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

    def __call__(self, query_points):
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

        future = self._client.call_async(req)
        done = Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout_sec):
            future.cancel()
            raise TimeoutError(
                f"SDF service '{self._client.srv_name}' did not respond within "
                f"{self._timeout_sec:.2f}s"
            )

        resp = future.result()
        if resp is None:
            raise RuntimeError(
                f"SDF service '{self._client.srv_name}' returned no result"
            )
        if hasattr(resp, 'success') and not resp.success:
            raise RuntimeError(
                f"SDF service '{self._client.srv_name}' returned success=False"
            )

        n = len(req.query_points)
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
