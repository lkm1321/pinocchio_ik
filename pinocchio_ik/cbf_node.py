import os
import tempfile
import threading
import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.distance import cdist

from pinocchio_ik.pointcloud_sdf import PointCloudSDF
from pinocchio_ik.qpcbf import PinocchioFKCBF, table_sdf, wall_sdf, compose


class PointCloudKDTreeNode(Node):
    def __init__(self):
        super().__init__('pointcloud_kdtree_node')

        # PointCloud Subscriber
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            "points",
            self.pointcloud_callback,
            10
        )
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2,
            "filtered_points",
            10
        )
        self.threshold = 0.1
        self.sdf = PointCloudSDF(buffer=0.05)

        hardcoded_urdf_path = '/keti_ws/src/pinocchio_ik/launch/xarm6_with_gripper_spherized.urdf'
        if os.path.exists(hardcoded_urdf_path):
            self.get_logger().info("Using hardcoded path")

            self.model = pin.buildModelFromUrdf(hardcoded_urdf_path)
            self.pin_data = self.model.createData()
            self.collision_model = pin.buildGeomFromUrdf(
                self.model,
                hardcoded_urdf_path,
                pin.GeometryType.COLLISION
            )
            self.collision_data = self.collision_model.createData()
        else:
            urdf_str = self.get_parameter('robot_description').get_parameter_value().string_value

            if urdf_str:
                self.get_logger().info("Loaded URDF from parameter 'robot_description'.")
                self.init_robot_model(urdf_str)
            else:
                qos_profile = QoSProfile(
                    depth=1,  # Queue size
                    durability=DurabilityPolicy.TRANSIENT_LOCAL  # Enable latching
                )
                self.get_logger().warn("Parameter 'robot_description' not set. Waiting for topic '/robot_description'...")
                self.urdf_sub = self.create_subscription(String, '/robot_description', self.urdf_callback, qos_profile=qos_profile)

                self.model = None
                self.pin_data = None
                self.collision_model = None
                self.collision_data = None

        # subscribe to current joint state
        self.current_joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_state = None

    def urdf_callback(self, msg: String):
        urdf_str = msg.data
        self.get_logger().info("Received URDF from /robot_description topic.")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as urdf_file:
            urdf_file.write(urdf_str)
            self.get_logger().info(f"Temporary URDF file created at {urdf_file.name}")
            urdf_file.flush()
            self.model = pin.buildModelFromUrdf(urdf_file.name)
            self.pin_data = self.model.createData()
            self.collision_model = pin.buildGeomFromUrdf(
                self.model,
                urdf_file.name,
                pin.GeometryType.COLLISION
            )
            self.collision_data = self.collision_model.createData()

    def pointcloud_callback(self, msg: PointCloud2):
        # ros2_numpy is only needed for the legacy point-cloud-driven CBF
        # path. Defer the import so the service-call CBF path can run in
        # images that don't ship it.
        import ros2_numpy
        t0 = time.time()

        cloud_arr = ros2_numpy.numpify(msg)

        pts = np.zeros((cloud_arr.size, 3), dtype=np.float32)
        pts[:, 0] = cloud_arr['x'].flatten()
        pts[:, 1] = cloud_arr['y'].flatten()
        pts[:, 2] = cloud_arr['z'].flatten()
        pts = pts[np.isfinite(pts).all(axis=1)]

        if pts.shape[0] == 0:
            self.get_logger().warn("Received empty or invalid point cloud.")
            return

        if self.model is not None and self.current_joint_state is not None:
            pin.updateGeometryPlacements(
                self.model,
                self.pin_data,
                self.collision_model,
                self.collision_data,
                self.current_joint_state
            )

            sphere_positions = np.array([
                self.collision_data.oMg[self.collision_model.getGeometryId(visual.name)].translation
                for visual in self.collision_model.geometryObjects
            ])

            sphere_radii = np.array([
                visual.geometry.radius
                for visual in self.collision_model.geometryObjects
            ])

            distance = np.min(cdist(pts, sphere_positions) - sphere_radii, axis=-1)
            pts = pts[distance > self.threshold]

            cloud_arr_filtered = np.zeros(pts.shape[0], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
            ])
            cloud_arr_filtered['x'] = pts[:, 0]
            cloud_arr_filtered['y'] = pts[:, 1]
            cloud_arr_filtered['z'] = pts[:, 2]

            point_cloud_msg = ros2_numpy.msgify(PointCloud2, cloud_arr_filtered, stamp=msg.header.stamp, frame_id=msg.header.frame_id)

            self.pointcloud_publisher.publish(point_cloud_msg)

        t1 = time.time()

        self.sdf.update_points(pts)
        t2 = time.time()

        print(f"t1 - t0: {t1 - t0}")
        print(f"t2 - t1: {t2 - t1}")

    def joint_state_callback(self, msg: JointState):

        if self.model is None:
            self.get_logger().info("Waiting for URDF")
            return

        for joint_name, position in zip(msg.name, msg.position):
            if joint_name not in self.model.names:
                continue

            if self.current_joint_state is None:
                self.current_joint_state = np.zeros(self.model.nq)

            self.current_joint_state[self.model.getJointId(joint_name) - 1] = position

    def sdf_and_gradient(self, query_point):
        return self.sdf(query_point)


_DEFAULT_URDF_PATH = '/keti_ws/src/pinocchio_ik/launch/xarm6_with_gripper_spherized.urdf'
_DEFAULT_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_DEFAULT_EE_FRAME = 'xarm6_ee_tip'


class DistanceCBFNode(Node):
    def __init__(
        self,
        *controller_args,
        urdf_path=_DEFAULT_URDF_PATH,
        controlled_joint_names=None,
        ee_frame_name=_DEFAULT_EE_FRAME,
        alpha_gain=2.0,
    ):
        super().__init__('distance_cbf_node')
        self.get_logger().info('DistanceCBFNode has been started.')
        self.declare_parameter('robot_description', '')

        self.urdf_path = urdf_path
        self.controlled_joint_names = (
            list(controlled_joint_names)
            if controlled_joint_names is not None
            else list(_DEFAULT_JOINT_NAMES)
        )
        self.ee_frame_name = ee_frame_name
        self.alpha_gain = float(alpha_gain)
        self.get_logger().info(f"CBF alpha_gain = {self.alpha_gain}")

        self.controller_args = controller_args

        if self.urdf_path and os.path.exists(self.urdf_path):
            self.get_logger().info(f"Loading URDF from {self.urdf_path}")

            self.controller = PinocchioFKCBF.from_urdf_file(
                self.urdf_path,
                *self.controller_args,
                controlled_joint_names=self.controlled_joint_names,
                ee_frame_name=self.ee_frame_name,
                alpha_gain=self.alpha_gain,
            )
        else:
            urdf_str = self.get_parameter('robot_description').get_parameter_value().string_value

            if urdf_str:
                self.get_logger().info("Loaded URDF from parameter 'robot_description'.")
                self.init_robot_model(urdf_str)
            else:
                qos_profile = QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL
                )
                self.get_logger().warn("Parameter 'robot_description' not set. Waiting for topic '/robot_description'...")
                self.urdf_sub = self.create_subscription(String, '/robot_description', self.urdf_callback, qos_profile=qos_profile)

            self.controller = None

        # Subscriptions are pure setters: they cache the latest joint state /
        # nominal command and stamp a receive time. The QP is driven by a
        # separate timer on its own callback group so the subscriptions
        # never serialize behind a slow filter cycle. Keeping the two
        # subscriptions in a single MX group is fine — neither callback
        # does meaningful work.
        self._input_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.current_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=self._input_cb_group,
        )
        self.current_joint_state = None
        self._last_js_recv_t = None  # time.monotonic() of last joint_state

        self.nominal_joint_velocity_sub = self.create_subscription(
            Float64MultiArray,
            'nominal_joint_velocity',
            self.nominal_joint_velocity_callback,
            10,
            callback_group=self._input_cb_group,
        )
        self.nominal_velocity = np.zeros(len(self.controller.controlled_joint_idxs))
        self._last_nom_recv_t = None  # time.monotonic() of last nominal cmd

        self.filtered_joint_velocity_pub = self.create_publisher(
            Float64MultiArray,
            'filtered_joint_velocity',
            10
        )

        # Sphere-marker visualization: each CBF collision sphere is published
        # as a Marker.SPHERE coloured by per-sphere clearance h = d_env - r.
        # Mapping uses matplotlib's 'hot' colormap: black (h = d_black, safe)
        # → red (h = d_red, danger) → yellow/white (h < d_red, extrapolated).
        # publish_sphere_markers=False or marker_rate_hz<=0 disables it.
        self.declare_parameter('publish_sphere_markers', True)
        self.declare_parameter('sphere_marker_topic', 'cbf_spheres')
        self.declare_parameter('sphere_marker_d_red', 0.1)
        self.declare_parameter('sphere_marker_d_black', 2.0)
        self.declare_parameter('sphere_marker_rate_hz', 20.0)
        self.publish_sphere_markers = bool(
            self.get_parameter('publish_sphere_markers').value
        )
        self.sphere_marker_d_red = float(
            self.get_parameter('sphere_marker_d_red').value
        )
        self.sphere_marker_d_black = float(
            self.get_parameter('sphere_marker_d_black').value
        )
        marker_rate = float(self.get_parameter('sphere_marker_rate_hz').value)
        self._sphere_marker_period_s = (
            1.0 / marker_rate if (self.publish_sphere_markers and marker_rate > 0.0)
            else None
        )
        self._last_sphere_marker_t = 0.0
        if self.publish_sphere_markers and self._sphere_marker_period_s is not None:
            self.sphere_marker_pub = self.create_publisher(
                MarkerArray,
                self.get_parameter('sphere_marker_topic')
                    .get_parameter_value().string_value,
                10,
            )
        else:
            self.sphere_marker_pub = None

        # Filter timer runs at 100Hz on its own callback group so the
        # subscriptions can dispatch on the executor's other thread while
        # the QP is in flight. Per-cycle staleness check (see _STALE_S)
        # forces a zero command if either input has dropped below 10Hz.
        self._filter_period_s = 0.01
        self._STALE_S = 0.15  # 1.5x the 10Hz nominal-period budget
        self._filter_timer = self.create_timer(
            self._filter_period_s,
            self._filter_timer_cb,
            callback_group=self._timer_cb_group,
        )

        # Rolling instrumentation counters; reset by _log_diag every window.
        self._diag_window_s = 2.0
        self._last_diag_t = time.monotonic()
        self._js_rx = 0           # raw joint_states received
        self._nom_rx = 0          # nominal_joint_velocity received
        self._timer_ticks = 0     # filter timer firings
        self._fp_count = 0        # filter_and_publish invocations in window
        self._stale_pub = 0       # zero-publish events due to stale inputs
        self._js_stale_pub = 0    # of those, stale joint_state
        self._nom_stale_pub = 0   # of those, stale nominal
        self._fp_total = 0.0      # cumulative wall time
        self._fp_max = 0.0
        self._sum_sdf_time = 0.0
        self._sum_sdf_calls = 0
        self._sum_pin_time = 0.0
        self._sum_mat_time = 0.0
        self._sum_build_time = 0.0
        self._sum_solve_time = 0.0

    def urdf_callback(self, msg: String):
        urdf_str = msg.data
        self.get_logger().info("Received URDF from /robot_description topic.")
        self.init_robot_model(urdf_str)

    def init_robot_model(self, urdf_str: str):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as urdf_file:
            urdf_file.write(urdf_str)
            self.get_logger().info(f"Temporary URDF file created at {urdf_file.name}")
            urdf_file.flush()
            self.controller = PinocchioFKCBF.from_urdf_file(
                urdf_file.name,
                *self.controller_args,
                controlled_joint_names=self.controlled_joint_names,
                ee_frame_name=self.ee_frame_name,
                alpha_gain=self.alpha_gain,
            )

    def joint_state_callback(self, msg: JointState):
        """Pure setter: cache the latest joint configuration + recv time."""
        if self.controller is None:
            return

        self._js_rx += 1

        if self.current_joint_state is None:
            self.current_joint_state = np.zeros(len(self.controller.controlled_joint_idxs))

        for joint_name, position in zip(msg.name, msg.position):
            if joint_name not in self.controller.model.names:
                continue
            self.current_joint_state[self.controller.model.getJointId(joint_name) - 1] = position

        self._last_js_recv_t = time.monotonic()

    def nominal_joint_velocity_callback(self, msg: Float64MultiArray):
        """Pure setter: cache the latest nominal velocity + recv time."""
        self._nom_rx += 1
        if self.controller is None:
            return
        self.nominal_velocity = np.array(msg.data)
        self._last_nom_recv_t = time.monotonic()

    def _filter_timer_cb(self):
        """100Hz tick: check input freshness, then run the CBF QP and publish."""
        if self.controller is None:
            return

        self._timer_ticks += 1
        now = time.monotonic()

        js_t = self._last_js_recv_t
        nom_t = self._last_nom_recv_t
        js_stale = js_t is None or (now - js_t) > self._STALE_S
        nom_stale = nom_t is None or (now - nom_t) > self._STALE_S

        if js_stale or nom_stale:
            self._publish_zero_stale(now, js_stale, nom_stale, js_t, nom_t)
        else:
            self.filter_and_publish(self.nominal_velocity)

        self._log_diag()

    def _publish_zero_stale(self, now, js_stale, nom_stale, js_t, nom_t):
        """Emit zero velocity and a throttled warning when inputs go stale."""
        n_dof = len(self.controller.controlled_joint_idxs)
        msg = Float64MultiArray()
        msg.data = [0.0] * n_dof
        self.filtered_joint_velocity_pub.publish(msg)

        self._stale_pub += 1
        if js_stale:
            self._js_stale_pub += 1
        if nom_stale:
            self._nom_stale_pub += 1
        self._last_nom_norm = 0.0
        self._last_filt_norm = 0.0

        last_warn = getattr(self, '_last_stale_warn_t', 0.0)
        if now - last_warn > 1.0:
            self._last_stale_warn_t = now
            js_age = (now - js_t) if js_t is not None else float('inf')
            nom_age = (now - nom_t) if nom_t is not None else float('inf')
            self.get_logger().warn(
                f"cbf: STOP — inputs stale (threshold {self._STALE_S*1e3:.0f}ms): "
                f"joint_state age={js_age*1e3:.0f}ms, "
                f"nominal_cmd age={nom_age*1e3:.0f}ms; publishing zero velocity."
            )

    def filter_and_publish(self, nominal_velocity):
        filtered_msg = Float64MultiArray()
        zero = [0.] * len(self.controller.controlled_joint_idxs)
        t_fp0 = time.perf_counter()
        try:
            control = self.controller.get_control(
                self.current_joint_state,
                nominal_velocity
            )
            filtered_msg.data = control.tolist()
        except ValueError as e:
            self.get_logger().error(
                f"Infeasible CBF QP; publishing zero velocity: {e}"
            )
            filtered_msg.data = zero
        except Exception as e:
            # Covers SDF failures (TimeoutError, RuntimeError, …) raised
            # from inside controller.get_control() via env_sdf, plus any
            # other unexpected error. Fail safe: zero velocity.
            self.get_logger().error(
                f"SDF/controller error; publishing zero velocity: "
                f"{type(e).__name__}: {e}"
            )
            filtered_msg.data = zero
        t_fp1 = time.perf_counter()

        self.filtered_joint_velocity_pub.publish(filtered_msg)

        # Accumulate per-cycle stats; the timer emits a consolidated log
        # line per window via _log_diag().
        self._fp_count += 1
        dt = t_fp1 - t_fp0
        self._fp_total += dt
        if dt > self._fp_max:
            self._fp_max = dt
        self._sum_sdf_time += getattr(self.controller, 'last_sdf_time', 0.0)
        self._sum_sdf_calls += getattr(self.controller, 'last_sdf_calls', 0)
        self._sum_pin_time += getattr(self.controller, 'last_pin_time', 0.0)
        inner = getattr(self.controller, 'controller', None)
        if inner is not None:
            self._sum_mat_time += getattr(inner, 'last_matrix_time', 0.0)
            self._sum_build_time += getattr(inner, 'last_build_time', 0.0)
            self._sum_solve_time += getattr(inner, 'last_solve_time', 0.0)

        self._last_nom_norm = float(np.linalg.norm(np.asarray(nominal_velocity)))
        self._last_filt_norm = float(np.linalg.norm(np.asarray(filtered_msg.data)))

        self._maybe_publish_sphere_markers()

    def _maybe_publish_sphere_markers(self):
        if self.sphere_marker_pub is None:
            return
        now = time.monotonic()
        if now - self._last_sphere_marker_t < self._sphere_marker_period_s:
            return
        pos = getattr(self.controller, 'last_sphere_pos', None)
        radii = getattr(self.controller, 'last_sphere_radii', None)
        h = getattr(self.controller, 'last_sphere_h', None)
        if pos is None or radii is None or h is None or len(pos) == 0:
            return
        self._last_sphere_marker_t = now

        colors = self._hot_colors_for_clearance(h)
        frame_id = self.controller.urdf_model.get_root()
        stamp = self.get_clock().now().to_msg()

        arr = MarkerArray()
        for i in range(len(pos)):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'cbf_spheres'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[i, 0])
            m.pose.position.y = float(pos[i, 1])
            m.pose.position.z = float(pos[i, 2])
            m.pose.orientation.w = 1.0
            d = 2.0 * float(radii[i])
            m.scale.x = m.scale.y = m.scale.z = d
            m.color.r = float(colors[i, 0])
            m.color.g = float(colors[i, 1])
            m.color.b = float(colors[i, 2])
            m.color.a = 0.7
            arr.markers.append(m)
        self.sphere_marker_pub.publish(arr)

    def _hot_colors_for_clearance(self, h):
        """Map per-sphere clearance to RGB via matplotlib's 'hot' colormap.

        h = d_env - r (sphere-surface to nearest obstacle, in metres).
        Linear scale chosen so h=d_red → hot(0.365) = pure red, h=d_black →
        hot(0) = black. Below d_red the scale extrapolates into the
        green→white range (yellow / hot = even more alarming); above d_black
        clipped to black.
        """
        h = np.asarray(h, dtype=float)
        d_red = self.sphere_marker_d_red
        d_black = self.sphere_marker_d_black
        # Avoid divide-by-zero if a user mis-configures the params equal.
        span = max(d_black - d_red, 1e-9)
        s = (d_black - h) / span * 0.365
        s = np.clip(s, 0.0, 1.0)
        # Hot colormap segments:
        #   red:   ramps 0 → 1 over s ∈ [0,    0.365]
        #   green: ramps 0 → 1 over s ∈ [0.365, 0.745]
        #   blue:  ramps 0 → 1 over s ∈ [0.745, 1.000]
        r = np.clip(s / 0.365, 0.0, 1.0)
        g = np.clip((s - 0.365) / (0.745 - 0.365), 0.0, 1.0)
        b = np.clip((s - 0.745) / (1.0 - 0.745), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    def _log_diag(self):
        now = time.monotonic()
        window = now - self._last_diag_t
        if window < self._diag_window_s:
            return

        n = max(self._fp_count, 1)
        js_age_ms = (
            (now - self._last_js_recv_t) * 1e3
            if self._last_js_recv_t is not None else float('inf')
        )
        nom_age_ms = (
            (now - self._last_nom_recv_t) * 1e3
            if self._last_nom_recv_t is not None else float('inf')
        )
        self.get_logger().info(
            f"cbf rates [Hz]: "
            f"js_rx={self._js_rx / window:.1f} "
            f"nom_rx={self._nom_rx / window:.1f} "
            f"tick={self._timer_ticks / window:.1f} "
            f"pub={self._fp_count / window:.1f} "
            f"stale={self._stale_pub / window:.1f} (js={self._js_stale_pub}, nom={self._nom_stale_pub}) | "
            f"filter mean={self._fp_total / n * 1e3:.1f}ms "
            f"max={self._fp_max * 1e3:.1f}ms "
            f"(pin={self._sum_pin_time / n * 1e3:.1f}ms, "
            f"sdf={self._sum_sdf_time / n * 1e3:.1f}ms x{self._sum_sdf_calls / n:.1f}, "
            f"mat={self._sum_mat_time / n * 1e3:.1f}ms, "
            f"cvxpy_build={self._sum_build_time / n * 1e3:.1f}ms, "
            f"solve={self._sum_solve_time / n * 1e3:.1f}ms) | "
            f"||nom||={getattr(self, '_last_nom_norm', 0.0):.3f} "
            f"||filt||={getattr(self, '_last_filt_norm', 0.0):.3f} | "
            f"age js={js_age_ms:.0f}ms nom={nom_age_ms:.0f}ms | "
            f"thread={threading.current_thread().name}"
        )

        self._last_diag_t = now
        self._js_rx = 0
        self._nom_rx = 0
        self._timer_ticks = 0
        self._fp_count = 0
        self._stale_pub = 0
        self._js_stale_pub = 0
        self._nom_stale_pub = 0
        self._fp_total = 0.0
        self._fp_max = 0.0
        self._sum_sdf_time = 0.0
        self._sum_sdf_calls = 0
        self._sum_pin_time = 0.0
        self._sum_mat_time = 0.0
        self._sum_build_time = 0.0
        self._sum_solve_time = 0.0


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()

    table_sdf_z0 = 0.01
    table_sdf_to_use = lambda query_point: table_sdf(table_sdf_z0, query_point)

    wall_sdf_x0 = -0.2
    wall_sdf_to_use = lambda query_point: wall_sdf(wall_sdf_x0, query_point)

    composed_sdf_to_use = compose(table_sdf_to_use, wall_sdf_to_use)


    cbf_node = DistanceCBFNode(composed_sdf_to_use)
    executor.add_node(cbf_node)

    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
