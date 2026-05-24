import numpy as np
from scipy.spatial import KDTree


class PointCloudSDF:
    """KDTree-backed signed distance field over a static point cloud.

    Call as ``sdf(query_points) -> (distance, gradient)``. Matches the
    ``(distance, gradient)`` contract that ``PinocchioFKCBF`` expects when
    constructed with ``env_grad=None``.
    """

    def __init__(self, points=None, buffer=0.0):
        self.buffer = float(buffer)
        self.kdtree = None
        if points is not None:
            self.update_points(points)

    def update_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3); got {points.shape}")
        if points.shape[0] == 0:
            self.kdtree = None
            return
        self.kdtree = KDTree(points, copy_data=True)

    def __call__(self, query_point):
        query_point = np.asarray(query_point)
        if self.kdtree is None:
            return (
                100.0 * np.ones(query_point.shape[:-1]),
                np.broadcast_to(np.array([1.0, 0.0, 0.0]), query_point.shape).copy(),
            )
        dist, idx = self.kdtree.query(query_point)
        closest = self.kdtree.data[idx, :]
        safe = np.where(dist > 1e-12, dist, 1.0)
        gradient = (query_point - closest) / safe[..., np.newaxis]
        return dist - self.buffer, gradient
