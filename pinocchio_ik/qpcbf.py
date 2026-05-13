import cvxpy as cp
import numpy as np
import pinocchio as pin
from scipy.spatial import KDTree
from urdf_parser_py import urdf


class QPCBF:
    def __init__(self, nu, update_method, nx=None, alpha_function=lambda x: x):
        self.nominal_control = cp.Parameter(nu)
        self.control = cp.Variable(nu)
        if nx is None:
            nx = nu

        self.update_matrices = update_method

        A_matrix_test, b_vector_test, _ = self.update_matrices(np.ones(nx))

        self.A_matrix = cp.Parameter(A_matrix_test.shape)
        self.b_vector = cp.Parameter(b_vector_test.shape)
        self.M_matrix = cp.Parameter((nu, nu), PSD=True)

        self.objective = cp.Minimize(cp.quad_form(self.control - self.nominal_control, self.M_matrix))
        self.constraints = [self.A_matrix @ self.control + self.b_vector >= 0]
        self.problem = cp.Problem(self.objective, self.constraints)
        self.alpha_function = alpha_function

    def get_control(self, current_state, nominal_control_np):

        A_matrix, b_vector, M_matrix = self.update_matrices(current_state)

        objective = cp.Minimize(cp.quad_form(self.control - nominal_control_np, M_matrix))
        b_vector = self.alpha_function(b_vector)
        constraints = [A_matrix @ self.control + b_vector >= 0]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver='osqp')

        if self.control.value is not None:
            return self.control.value
        else:
            raise ValueError("Infeasible CBF QP!")

    def solve_ode(self, start_state, nominal_controller, n_steps, dt, terminal_condition=None):
        solutions = np.empty((n_steps, start_state.shape[-1]))
        solutions[0] = start_state

        controls = np.empty((n_steps - 1, start_state.shape[-1]))
        for idx in range(n_steps - 1):
            controls[idx] = self.get_control(solutions[idx], nominal_controller(solutions[idx]))
            solutions[idx + 1] = solutions[idx] + controls[idx] * dt

            if terminal_condition is not None and terminal_condition(solutions[idx], controls[idx]):
                controls = controls[:idx]
                solutions = solutions[:idx + 1]
                break

        return solutions, controls


def extract_spheres_from_urdf(robot: urdf.Robot):
    positions = {
        link_name: np.array([
            collision.origin.xyz
            for collision in link.collisions
        ])
        for link_name, link in robot.link_map.items()
    }
    radii = {
        link_name: np.array([
            collision.geometry.radius
            for collision in link.collisions
        ])
        for link_name, link in robot.link_map.items()
    }

    return positions, radii


def table_sdf(z0, query_point):
    dist = query_point[..., 2] - z0
    gradient = np.zeros_like(query_point)
    gradient[..., 2] = 1.

    return dist, gradient


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


class PinocchioFKCBF:

    def __init__(
        self,
        model,
        urdf_model,
        env_sdf,
        env_grad=None,
        controlled_joint_names=None,
        ee_frame_name=None,
    ):
        self.env_sdf = env_sdf
        self.env_grad = env_grad

        self.model = model
        self.data = self.model.createData()

        if controlled_joint_names is None:
            controlled_joint_names = [name for name in self.model.names if name != "universe"]
        self.controlled_joint_names = controlled_joint_names
        self.controlled_joint_idxs = [self.model.getJointId(name) - 1 for name in controlled_joint_names]

        self.ee_frame_name = ee_frame_name

        self.urdf_model = urdf_model
        self.sphere_positions, self.sphere_radii = extract_spheres_from_urdf(self.urdf_model)
        self.controller = QPCBF(
            nu=len(self.controlled_joint_idxs),
            update_method=self.build_matrix,
            nx=len(self.controlled_joint_idxs),
            alpha_function=lambda x: 5.0 * x,
        )

        self.get_control = self.controller.get_control

    @classmethod
    def from_urdf_file(cls, urdf_file, env_sdf, env_grad=None, controlled_joint_names=None, ee_frame_name=None):
        return cls(
            pin.buildModelFromUrdf(urdf_file),
            urdf.URDF.from_xml_file(urdf_file),
            env_sdf,
            env_grad=env_grad,
            controlled_joint_names=controlled_joint_names,
            ee_frame_name=ee_frame_name,
        )

    def get_frame_pose(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[frame_id]

    def get_frame_jacobian(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        return pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.WORLD)

    def build_matrix(self, configuration: np.ndarray):

        pin.computeJointJacobians(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)

        A_matrix = []
        b_vector = []

        for link_name in self.urdf_model.link_map.keys():
            if link_name == self.urdf_model.get_root():
                continue

            if link_name not in self.sphere_positions:
                continue
            if self.sphere_positions[link_name].size == 0:
                continue

            link_pose = self.get_frame_pose(link_name)

            sphere_pos_local = self.sphere_positions[link_name]
            sphere_pos_world = (link_pose.rotation @ sphere_pos_local[..., np.newaxis] + link_pose.translation[:, np.newaxis])[..., 0]

            spatial_jacobian = self.get_frame_jacobian(link_name)
            spatial_jacobian = spatial_jacobian[:, self.controlled_joint_idxs]

            if self.env_grad is not None:
                env_distance_values = self.env_sdf(sphere_pos_world) - self.sphere_radii[link_name]
                env_grad_values = self.env_grad(sphere_pos_world)
            else:
                env_distance_values, env_grad_values = self.env_sdf(sphere_pos_world)
                env_distance_values -= self.sphere_radii[link_name]

            distance_cross_product = np.cross(sphere_pos_world, env_grad_values)

            vel = spatial_jacobian[:3, :]
            omega = spatial_jacobian[3:, :]

            partial_d_partial_theta = (distance_cross_product @ omega + env_grad_values @ vel)

            A_matrix.append(partial_d_partial_theta)
            b_vector.append(env_distance_values)

        if self.ee_frame_name is not None:
            ee_jacobian = self.get_frame_jacobian(self.ee_frame_name)
            ee_jacobian = ee_jacobian[:, self.controlled_joint_idxs]
            M_matrix = ee_jacobian.T @ ee_jacobian + 1e-3 * np.eye(ee_jacobian.shape[-1])
        else:
            M_matrix = np.eye(len(self.controlled_joint_idxs))

        return np.concatenate(A_matrix, axis=0), np.concatenate(b_vector, axis=0), M_matrix
