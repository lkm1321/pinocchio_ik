import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as Rot

import pinocchio_ik.triad_openvr as vr

# --- OpenVR frame -> robot frame (forward-left-up) -------------------------
# (x, y, z)_robot = (z, -x, -y)_vr   [taken straight from the original stub]
VR_TO_ROBOT = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
])
VR_TO_ROBOT_ROT = Rot.from_matrix(VR_TO_ROBOT)


class TwistPublisherNode(Node):
    """VR teleop as a virtual coupling.

    A PD law couples the VR controller pose to the robot EE and outputs a
    Twist. The finite PD bandwidth - not an explicit rate limiter - is what
    keeps motion smooth when collision avoidance blocks the arm: while the
    EE is stuck the position error ("spring stretch") grows but is clamped,
    and on release the robot catches up at a bounded speed instead of
    snapping.

    The EE pose is taken from tf: the transform of `ee_frame` expressed in
    `base_frame`. The published Twist is expressed in the EE frame.
    """

    def __init__(self):
        super().__init__('twist_publisher')

        # Frame parameters. ee_frame is the EE link. base_frame is the fixed
        # frame the tf lookup and the PD math run in; the resulting twist is
        # rotated into the EE frame before it is published.
        self.declare_parameter('ee_frame', 'link6')
        self.declare_parameter('base_frame', 'base_link')
        self.ee_frame = self.get_parameter('ee_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # PD gains. Output is a velocity, so kp maps metres -> m/s, etc.
        # Higher kp = stiffer/faster catch-up; kd damps the catch-up.
        self.kp_lin, self.kd_lin = 4.0, 0.8
        self.kp_ang, self.kd_ang = 4.0, 0.8

        # Divergence clamp: the largest "spring stretch" allowed while the
        # EE is collision-blocked. Bounds the catch-up speed on release
        # (max ~ kp * max_err) so the arm never lunges.
        self.max_pos_err = 0.15   # m
        self.max_ang_err = 0.5    # rad

        self.deriv_alpha = 0.6    # low-pass factor on the numerical derivative
        self.dt = 0.1             # s, matches the timer period

        # Watchdog: if timer_callback hasn't successfully published within
        # this many seconds, an independent timer publishes a zero Twist.
        # Default = 3 control periods (300 ms at dt=0.1) so an occasional
        # late tick is tolerated but a real stall (TF unavailable, VR pose
        # missing, exception, executor starvation) falls back to safe zero.
        self.declare_parameter('watchdog_timeout_sec', 0.0)
        wd_param = self.get_parameter('watchdog_timeout_sec').value
        self.watchdog_timeout_sec = wd_param if wd_param > 0.0 else 3.0 * self.dt

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # EE pose feedback comes from tf.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot EE state (filled each tick from the tf lookup).
        self.ee_pos = None
        self.ee_rot = None

        # Relative VR<->EE map, captured once at engage time.
        self.offset_pos = None
        self.offset_rot = None

        # Derivative state.
        self.prev_pos_err = np.zeros(3)
        self.prev_ang_err = np.zeros(3)
        self.pos_err_dot = np.zeros(3)
        self.ang_err_dot = np.zeros(3)

        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Watchdog state — set after each successful publish so a stalled
        # timer_callback can be caught by the separate watchdog timer.
        self._last_publish_t = time.monotonic()
        self._watchdog_warned = False
        self._watchdog_timer = self.create_timer(
            min(0.2 * self.dt, 0.05), self._watchdog_tick,
        )

        self.v = vr.triad_openvr()
        self.controller = self.v.devices["controller_1"]
        self.get_logger().info(
            f'VR virtual-coupling node started '
            f'(EE tf: {self.base_frame} -> {self.ee_frame}).')

    # -- robot feedback: EE pose from tf ------------------------------------
    def get_ee_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'TF {self.base_frame} -> {self.ee_frame} unavailable: {ex}',
                throttle_duration_sec=2.0)
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        pos = np.array([t.x, t.y, t.z])
        rot = Rot.from_quat([q.x, q.y, q.z, q.w])
        return pos, rot

    # -- VR pose expressed in the robot base frame --------------------------
    def get_vr_pose(self):
        pq = self.controller.get_pose_quaternion()   # [x,y,z, w,x,y,z] or None
        if pq is None:
            return None, None
        pos = VR_TO_ROBOT @ np.array(pq[0:3])
        # triad_openvr quaternion order is (w,x,y,z); scipy wants (x,y,z,w).
        quat = Rot.from_quat([pq[4], pq[5], pq[6], pq[3]])
        # Express the controller orientation in the robot frame, using the
        # same R as the stub's position/velocity jumble. A controller
        # orientation C_vr (body -> VR world) becomes R * C_vr (body ->
        # robot world); this is a single left-multiply, not a similarity.
        quat = VR_TO_ROBOT_ROT * quat
        return pos, quat

    @staticmethod
    def _clamp(vec, max_norm):
        n = np.linalg.norm(vec)
        return vec * (max_norm / n) if n > max_norm else vec

    def _capture_offset(self, vr_pos, vr_rot):
        """Anchor the relative map so the robot starts exactly where it is."""
        self.offset_pos = self.ee_pos - vr_pos
        # Right-multiplied offset: with ref_rot = vr_rot * offset_rot, a
        # rotation of the controller by delta in the (robot) world frame
        # maps to the same world-frame rotation of the EE reference.
        #   ref_rot = vr_rot * (vr_rot_0^-1 * ee_rot_0)
        #           = (vr_rot * vr_rot_0^-1) * ee_rot_0   = delta * ee_rot_0
        self.offset_rot = vr_rot.inv() * self.ee_rot
        self.prev_pos_err = np.zeros(3)
        self.prev_ang_err = np.zeros(3)
        self.pos_err_dot = np.zeros(3)
        self.ang_err_dot = np.zeros(3)
        self.get_logger().info('Engaged: captured VR<->EE offset.')

    # -- control loop -------------------------------------------------------
    def timer_callback(self):
        self.ee_pos, self.ee_rot = self.get_ee_pose()
        if self.ee_pos is None:
            return                             # tf not ready; warning logged

        vr_pos, vr_rot = self.get_vr_pose()
        lin_vel = self.controller.get_velocity()
        ang_vel = self.controller.get_angular_velocity()
        if vr_pos is None or lin_vel is None or ang_vel is None:
            return

        # Clutch: hold grip to freeze the robot and reposition your hand;
        # release to re-engage with a fresh offset (workspace re-indexing).
        inputs = self.controller.get_controller_inputs()
        if inputs and inputs.get('grip_button', False):
            self.offset_pos = None             # force re-capture on release
            self._publish(Twist())             # zero velocity while clutched
            return

        if self.offset_pos is None:            # first tick, or just un-clutched
            self._capture_offset(vr_pos, vr_rot)

        # Reference pose = VR pose mapped through the engage-time offset.
        ref_pos = vr_pos + self.offset_pos
        ref_rot = vr_rot * self.offset_rot

        # Tracking error (reference - current), in the robot base frame.
        # Clamping bounds the divergence accumulated while collision-blocked.
        pos_err = self._clamp(ref_pos - self.ee_pos, self.max_pos_err)
        ang_err = self._clamp((ref_rot * self.ee_rot.inv()).as_rotvec(),
                              self.max_ang_err)

        # Filtered numerical derivative of the error -> the D term.
        a = self.deriv_alpha
        self.pos_err_dot = (a * self.pos_err_dot +
                            (1 - a) * (pos_err - self.prev_pos_err) / self.dt)
        self.ang_err_dot = (a * self.ang_err_dot +
                            (1 - a) * (ang_err - self.prev_ang_err) / self.dt)
        self.prev_pos_err, self.prev_ang_err = pos_err, ang_err

        # VR velocity as feedforward, so the robot can hold a *moving*
        # target with zero steady-state lag. Drop these terms if you want a
        # purely spring-like feel (robot lags proportionally to hand speed).
        lin_ff = VR_TO_ROBOT @ np.array(lin_vel)
        ang_ff = VR_TO_ROBOT @ np.array(ang_vel)

        # PD + feedforward. Everything above is computed in the base frame.
        cmd_lin = lin_ff + self.kp_lin * pos_err + self.kd_lin * self.pos_err_dot
        cmd_ang = ang_ff + self.kp_ang * ang_err + self.kd_ang * self.ang_err_dot

        # Express the twist in the EE frame: rotate both 3-vectors by
        # R_ee^-1 (base -> EE). The reference point is unchanged (the EE
        # origin itself), so no linear/angular coupling term is needed.
        cmd_lin = self.ee_rot.inv().apply(cmd_lin)
        cmd_ang = self.ee_rot.inv().apply(cmd_ang)

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = map(float, cmd_lin)
        msg.angular.x, msg.angular.y, msg.angular.z = map(float, cmd_ang)
        self._publish(msg)

    # -- publish + watchdog --------------------------------------------------
    def _publish(self, msg: Twist):
        """Publish a Twist and refresh the watchdog deadline."""
        self.publisher.publish(msg)
        self._last_publish_t = time.monotonic()
        if self._watchdog_warned:
            self.get_logger().info('vr_virtual_coupling: control loop recovered.')
            self._watchdog_warned = False

    def _watchdog_tick(self):
        """Publish a zero Twist if timer_callback hasn't published within
        `watchdog_timeout_sec`.

        Reasons timer_callback might not publish: TF lookup failing, the VR
        controller pose unavailable, an exception in the PD math, or the
        executor being starved by a slow callback elsewhere. In any of
        those cases we'd rather command zero than have the downstream
        controller keep tracking the last (now stale) twist.
        """
        now = time.monotonic()
        if now - self._last_publish_t <= self.watchdog_timeout_sec:
            return

        self.publisher.publish(Twist())  # zero twist
        if not self._watchdog_warned:
            self.get_logger().warn(
                f'vr_virtual_coupling: control loop missed '
                f'{now - self._last_publish_t:.3f}s (timeout '
                f'{self.watchdog_timeout_sec:.3f}s); publishing zero twist.'
            )
            self._watchdog_warned = True


def main(args=None):
    rclpy.init(args=args)
    node = TwistPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
