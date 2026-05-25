import time

import cvxpy as cp
import numpy as np
import pinocchio as pin
from urdf_parser_py import urdf


class QPCBF:
    def __init__(self, nu, update_method, nx=None, alpha_function=lambda x: x):
        self.control = cp.Variable(nu)
        if nx is None:
            nx = nu

        self.update_matrices = update_method
        self.alpha_function = alpha_function

        # Per-call timings populated by get_control(); read by callers for
        # diagnostics (e.g. DistanceCBFNode's throttled rate/latency log).
        self.last_matrix_time = 0.0
        self.last_build_time = 0.0
        self.last_solve_time = 0.0
        self.last_total_time = 0.0

        # Per-call dual variables of the CBF inequality, indexed in the
        # same order as the rows of A / entries of b. Strictly-positive
        # duals mark active (binding) constraints; used by DistanceCBFNode
        # to draw a line from each constrained sphere to its nearest
        # obstacle point. None when the QP was infeasible.
        self.last_cbf_dual = None

        # get_control rebuilds the cvxpy Problem from scratch each call, so
        # we do NOT need to size cp.Parameters here — and doing so eagerly
        # would force an SDF query at construction time, which is fragile
        # when env_sdf is a ROS service that may not be ready yet.

    def get_control(self, current_state, nominal_control_np):

        t0 = time.perf_counter()
        A_matrix, b_vector, M_matrix = self.update_matrices(current_state)
        t1 = time.perf_counter()

        # Per-joint velocity bound — without it the QP can emit wildly
        # unsafe values (40+ rad/s) when M_matrix is near-singular and the
        # constraint is nearly redundant. 1 rad/s is well above what any
        # demo phase asks for and well within the xArm6 joint limits.
        u_max = 1.0

        objective = cp.Minimize(cp.quad_form(self.control - nominal_control_np, M_matrix))
        b_vector = self.alpha_function(b_vector)
        constraints = [
            A_matrix @ self.control + b_vector >= 0,
            self.control <= u_max,
            self.control >= -u_max,
        ]
        problem = cp.Problem(objective, constraints)
        t2 = time.perf_counter()

        problem.solve(solver='osqp')
        t3 = time.perf_counter()

        self.last_matrix_time = t1 - t0
        self.last_build_time = t2 - t1
        self.last_solve_time = t3 - t2
        self.last_total_time = t3 - t0

        # dual_value on a `>= 0` cvxpy constraint is non-negative; entries
        # strictly above the active threshold mean the inequality is tight
        # at the optimum (the corresponding sphere is currently shaping
        # the QP solution). May be None when the solve failed.
        self.last_cbf_dual = constraints[0].dual_value

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

def wall_sdf(x0, query_point):
    dist = query_point[..., 0] - x0
    gradient = np.zeros_like(query_point)
    gradient[..., 0] = 1.

    return dist, gradient


def compose(*sdfs):
    def compose_sdf(query_point):
        dists = []
        grads = []
        for sdf in sdfs:
            dist, grad = sdf(query_point)
            dists.append(dist)
            grads.append(grad)

        dists = np.stack(dists, axis=-1)
        grads = np.stack(grads, axis=-1)

        min_indices = np.argmin(dists, axis=-1)
        min_dists = np.take_along_axis(dists, min_indices[..., np.newaxis], axis=-1)[..., 0]
        min_grads = np.take_along_axis(grads, min_indices[..., np.newaxis, np.newaxis], axis=-1)[..., 0]

        return min_dists, min_grads
    return compose_sdf

class PinocchioFKCBF:

    def __init__(
        self,
        model,
        urdf_model,
        env_sdf,
        env_grad=None,
        controlled_joint_names=None,
        ee_frame_name=None,
        alpha_gain=2.0,
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
        # CBF class-K function: alpha(h) = alpha_gain * h. Larger gain ⇒ less
        # conservative (robot brakes only near contact); smaller ⇒ wider
        # standoff but mushier feel. See build_matrix docstring.
        self.alpha_gain = float(alpha_gain)
        self.controller = QPCBF(
            nu=len(self.controlled_joint_idxs),
            update_method=self.build_matrix,
            nx=len(self.controlled_joint_idxs),
            alpha_function=lambda x: self.alpha_gain * x,
        )

        # Per-call SDF stats populated by build_matrix(); read by callers
        # for diagnostics.
        self.last_sdf_calls = 0
        self.last_sdf_time = 0.0
        self.last_pin_time = 0.0

        # Per-call per-sphere state populated by build_matrix(); read by
        # DistanceCBFNode for the cbf_spheres marker visualization.
        # last_sphere_pos:    (N, 3) world-frame sphere centers
        # last_sphere_radii:  (N,)   sphere radii (same order)
        # last_sphere_h:      (N,)   h = d_env - r  (sphere-surface clearance)
        # last_sphere_dist:   (N,)   raw SDF value at sphere center (no radius subtraction)
        # last_sphere_grad:   (N, 3) SDF gradient at sphere center; nearest
        #                            obstacle point is pos - dist * grad.
        # last_A:             (N, n_u) per-sphere CBF inequality row; the
        #                              full constraint is A·q̇ + b ≥ 0 where
        #                              b is last_sphere_h.
        self.last_sphere_pos = None
        self.last_sphere_radii = None
        self.last_sphere_h = None
        self.last_sphere_dist = None
        self.last_sphere_grad = None
        self.last_A = None

        self.get_control = self.controller.get_control

    @classmethod
    def from_urdf_file(cls, urdf_file, env_sdf, env_grad=None, controlled_joint_names=None, ee_frame_name=None, alpha_gain=2.0):
        return cls(
            pin.buildModelFromUrdf(urdf_file),
            urdf.URDF.from_xml_file(urdf_file),
            env_sdf,
            env_grad=env_grad,
            controlled_joint_names=controlled_joint_names,
            ee_frame_name=ee_frame_name,
            alpha_gain=alpha_gain,
        )

    def get_frame_pose(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[frame_id]

    def get_frame_jacobian(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        return pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.WORLD)

    def build_matrix(self, configuration: np.ndarray):
        r"""Assemble the CBF inequality ``A(q) q_dot + alpha(b(q)) >= 0``.

        Notation
        --------
        q          : joint configuration (model.nq,)
        q_dot      : joint velocity (control input)
        For each collision sphere i attached to a link:
          p_i^L    : sphere center in the parent link frame
          R(q), t(q) : pose of the parent link in the world frame
          p_i(q) = R(q) p_i^L + t(q) : sphere center in world coords
          r_i      : sphere radius
        d_env(.)   : environment signed distance field (>=0 outside obstacles)
        g(.) = grad d_env(.) : its spatial gradient

        Per-sphere barrier
        ------------------
        Define the per-sphere signed distance
            h_i(q) := d_env(p_i(q)) - r_i.
        The CBF safety condition for each sphere is
            h_i_dot(q, q_dot) + alpha(h_i(q)) >= 0,                       (*)
        with alpha a class-K function (here alpha(x) = 5 x, set by
        ``QPCBF(..., alpha_function=...)``). Linearity of (*) in q_dot is
        what makes the per-step QP convex.

        Pinocchio frame convention
        --------------------------
        ``pin.getFrameJacobian(..., pin.ReferenceFrame.WORLD)`` returns the
        spatial Jacobian J(q) of the link frame, where the spatial velocity
            xi(q, q_dot) = [v; w] = J(q) q_dot
        is the twist of the *body coincident with the world origin*,
        expressed in world coordinates. The velocity of a material point on
        the rigid body currently located at world position p is therefore
            p_dot = v + w x p.
        Splitting J row-wise as J = [J_v; J_w] (each 3 x n), the world
        velocity of sphere i is
            p_i_dot = J_v q_dot + w x p_i = J_v q_dot - [p_i]_x J_w q_dot.

        Chain rule on h_i
        -----------------
            h_i_dot = g(p_i)^T p_i_dot
                    = g(p_i)^T (v + w x p_i)
                    = g(p_i)^T v + g(p_i)^T (w x p_i).
        Use the scalar triple product a . (b x c) = c . (a x b):
            g . (w x p_i) = w . (p_i x g) = (p_i x g)^T w.
        Hence
            dh_i/dq (q) = g(p_i)^T J_v(q) + (p_i x g(p_i))^T J_w(q),       (A)
        a row vector of length n_u (controlled joints only). This is the
        i-th row of ``A_matrix``; in code,
            partial_d_partial_theta
                = (p_i x g) @ J_w        # cross-product term
                + g       @ J_v.         # gradient-on-linear-vel term
        Stacking over all spheres gives A(q).

        b vector
        --------
        b_i(q) = h_i(q) = d_env(p_i(q)) - r_i. The alpha is applied later
        in ``QPCBF.get_control`` (``b_vector = self.alpha_function(b_vector)``)
        so that ``build_matrix`` returns the *raw* barrier values; the QP
        constraint is then exactly (*) stacked across all spheres:
            A(q) q_dot + alpha(b(q)) >= 0.

        Notes / assumptions
        -------------------
        * Only links with non-empty collision spheres in the URDF contribute.
        * Columns of J are restricted to ``controlled_joint_idxs`` so that
          the CBF acts only on the joints we command.
        * When ``env_grad`` is None, ``env_sdf`` is assumed to return
          ``(distance, gradient)`` jointly (e.g. ``PointCloudSDF``). The
          radius is subtracted from the distance to get h_i.
        * ``M_matrix`` weights the QP objective ``||q_dot - q_dot_nom||_M``.
          With ``ee_frame_name`` set, M = J_ee^T J_ee + 1e-3 I penalises
          deviations of the end-effector twist (Gauss-Newton-like weighting
          in task space); otherwise M = I.
        """

        self.last_sdf_calls = 0
        self.last_sdf_time = 0.0

        t_pin = time.perf_counter()
        pin.computeJointJacobians(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)
        self.last_pin_time = time.perf_counter() - t_pin

        # Gather per-link spheres + Jacobians first, then make ONE batched
        # env_sdf call on the concatenated point cloud. The SDF backend is
        # typically a service round-trip whose cost is dominated by the
        # round-trip itself, not by the payload size, so collapsing N
        # per-link calls into 1 is a big win.
        per_link = []           # (sphere_pos_world, spatial_jacobian, radii)
        all_sphere_pos = []

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

            per_link.append((sphere_pos_world, spatial_jacobian, self.sphere_radii[link_name]))
            all_sphere_pos.append(sphere_pos_world)

        A_matrix = []
        b_vector = []

        if all_sphere_pos:
            batched_pos = np.concatenate(all_sphere_pos, axis=0)  # (sum_N, 3)
            batched_radii = np.concatenate([r for _, _, r in per_link], axis=0)

            t_sdf = time.perf_counter()
            if self.env_grad is not None:
                batched_dist = self.env_sdf(batched_pos)
                batched_grad = self.env_grad(batched_pos)
                self.last_sdf_calls += 2
            else:
                batched_dist, batched_grad = self.env_sdf(batched_pos)
                self.last_sdf_calls += 1
            self.last_sdf_time += time.perf_counter() - t_sdf

            self.last_sphere_pos = batched_pos
            self.last_sphere_radii = batched_radii
            self.last_sphere_h = batched_dist - batched_radii
            self.last_sphere_dist = batched_dist
            self.last_sphere_grad = batched_grad

            cursor = 0
            for sphere_pos_world, spatial_jacobian, radii in per_link:
                n = sphere_pos_world.shape[0]
                env_distance_values = batched_dist[cursor:cursor + n] - radii
                env_grad_values = batched_grad[cursor:cursor + n]
                cursor += n

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

        A_concat = np.concatenate(A_matrix, axis=0)
        b_concat = np.concatenate(b_vector, axis=0)
        self.last_A = A_concat
        return A_concat, b_concat, M_matrix
