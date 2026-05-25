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
# (x, y, z)_robot = (z, x, y)_vr
# Verified empirically against the room layout: the original stub had
# negative signs on y/z which only "worked" because vr_virtual_coupling
# also rotated into the EE frame and the xArm6 stick EE happens to sit
# ~180° around x from the base, accidentally cancelling the bad signs.
# This matrix maps the physical room frame to the robot base frame and
# is independent of EE pose, so it's correct for both is_world modes.
VR_TO_ROBOT = np.array([
    [ 0.0,  0.0,  1.0],
    [ 1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
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
    `base_frame`. The published Twist's frame is selected by the `is_world`
    parameter and must match the consumer (`velocity_ik`'s `is_world`):

      * `is_world:=true`  (default) — publish in `base_frame`.
      * `is_world:=false` — rotate by `R_ee^-1` and publish in the EE frame.

    Mixing frames flips the PD sign on whichever axes the EE rotates
    relative to the base, producing instability (a known footgun on the
    xArm6 stick EE where left/right and up/down end up inverted).
    """

    def __init__(self):
        super().__init__('twist_publisher')

        # Frame parameters. ee_frame is the EE link. base_frame is the fixed
        # frame the tf lookup and the PD math run in. is_world selects the
        # frame of the published Twist and MUST match velocity_ik's is_world.
        self.declare_parameter('ee_frame', 'link6')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('is_world', True)
        self.ee_frame = self.get_parameter('ee_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.is_world = bool(self.get_parameter('is_world').value)

        # Trigger deadman: hold the trigger to drive the robot. The trigger
        # axis is analog (0.0 released → 1.0 fully pressed); cross this
        # threshold to count as "engaged". Releasing freezes the robot and
        # re-anchors the VR<->EE map on the next press (workspace
        # re-indexing).
        self.declare_parameter('trigger_threshold', 0.5)
        self.trigger_threshold = float(self.get_parameter('trigger_threshold').value)

        # Trackpad axis-mask: while the trigger is held, *clicking* the
        # trackpad selects which axes the robot tracks.
        #   left half  (trackpad_x < -deadzone) → position-only
        #   right half (trackpad_x > +deadzone) → orientation-only
        # No click, or click in the center band, → both (default).
        # The suppressed axis is re-anchored to the current EE pose each
        # tick, so its tracking error stays at zero — no spring stretch
        # builds up while suppressed and switching back to 'both' is smooth.
        self.declare_parameter('trackpad_deadzone', 0.3)
        self.trackpad_deadzone = float(self.get_parameter('trackpad_deadzone').value)

        # PD gains. Output is a velocity, so kp maps metres -> m/s, etc.
        # Higher kp = stiffer/faster catch-up; kd damps the catch-up.
        self.kp_lin, self.kd_lin = 4.0, 0.8
        self.kp_ang, self.kd_ang = 4.0, 0.8

        # Output scaling + saturation. The fixed gains throttle the overall
        # response (so the operator can dial in a comfortable "feel"
        # independent of the PD law), and the max-velocity clamps bound the
        # commanded twist regardless of error/feedforward. Clipping uses a
        # norm-preserving scale (direction kept, magnitude capped) so the
        # twist direction matches the operator's intent even at saturation.
        self.declare_parameter('gain_lin', 0.5)
        self.declare_parameter('gain_ang', 0.2)
        self.declare_parameter('max_lin_vel', 0.3)   # m/s
        self.declare_parameter('max_ang_vel', 1.0)   # rad/s
        self.gain_lin = float(self.get_parameter('gain_lin').value)
        self.gain_ang = float(self.get_parameter('gain_ang').value)
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)

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
            f'(EE tf: {self.base_frame} -> {self.ee_frame}, '
            f'twist frame: {"world" if self.is_world else "ee"}).')

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

    def _axis_mask(self, inputs):
        """Return (suppress_orient, suppress_lin) from the trackpad click.

        Trackpad clicked on the left half  → position-only (orientation
        suppressed). Clicked on the right half → orientation-only (linear
        suppressed). No click / center click → neither suppressed.
        """
        if not inputs or not inputs.get('trackpad_pressed', False):
            return False, False
        x = float(inputs.get('trackpad_x', 0.0))
        if x < -self.trackpad_deadzone:
            return True, False           # position-only
        if x >  self.trackpad_deadzone:
            return False, True           # orientation-only
        return False, False

    @staticmethod
    def _coerce_vec3(x):
        """Convert triad_openvr's 3-vector return shapes to a (3,) float ndarray.

        get_velocity() / get_angular_velocity() return an openvr.HmdVector3_t,
        a ctypes struct that numpy surfaces as a structured scalar with a
        single field 'v' of dtype ('<f4', (3,)). np.asarray(..., dtype=float)
        can't cast a structured dtype, so unwrap the 'v' field first.
        Also handles the ctypes-attribute form (.v) and plain sequences.
        """
        if isinstance(x, np.ndarray) and x.dtype.names and 'v' in x.dtype.names:
            x = x['v']
        elif hasattr(x, 'v'):
            x = x.v
        return np.asarray(x, dtype=float)

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

        # triad_openvr returns openvr.HmdVector3_t (a ctypes struct numpy
        # lifts as a structured scalar with field 'v'), not a plain
        # 3-tuple, so we have to unwrap before casting to float — otherwise
        # the VR_TO_ROBOT @ ... matmul below dies with
        # "ufunc 'matmul' did not contain signature matching types".
        try:
            lin_vel = self._coerce_vec3(lin_vel)
            ang_vel = self._coerce_vec3(ang_vel)
        except (TypeError, ValueError):
            self.get_logger().warn(
                f'vr_virtual_coupling: non-numeric VR velocity '
                f'(lin={lin_vel!r}, ang={ang_vel!r}); skipping tick.',
                throttle_duration_sec=2.0,
            )
            return
        if (lin_vel.shape != (3,) or ang_vel.shape != (3,)
                or not np.isfinite(lin_vel).all()
                or not np.isfinite(ang_vel).all()):
            self.get_logger().warn(
                f'vr_virtual_coupling: malformed VR velocity '
                f'(lin={lin_vel}, ang={ang_vel}); skipping tick.',
                throttle_duration_sec=2.0,
            )
            return

        # Deadman: hold the trigger to drive. While released we publish
        # zero and drop the engage offset so the next press re-anchors the
        # VR<->EE map at the current EE pose (no jump on re-engage).
        inputs = self.controller.get_controller_inputs()
        trigger_pressed = (
            bool(inputs) and inputs.get('trigger', 0.0) > self.trigger_threshold
        )
        if not trigger_pressed:
            self.offset_pos = None             # force re-capture on next press
            self._publish(Twist())             # zero velocity while released
            return

        if self.offset_pos is None:            # first tick after a fresh press
            self._capture_offset(vr_pos, vr_rot)

        # Trackpad axis-mask: re-anchor the suppressed axis to the current
        # EE pose so its ref==current → zero tracking error → zero PD output
        # on that axis. Feedforward is zeroed below to match.
        suppress_orient, suppress_lin = self._axis_mask(inputs)
        if suppress_orient:
            self.offset_rot = vr_rot.inv() * self.ee_rot
        if suppress_lin:
            self.offset_pos = self.ee_pos - vr_pos

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
        lin_ff = VR_TO_ROBOT @ lin_vel
        ang_ff = VR_TO_ROBOT @ ang_vel

        # Zero the feedforward on suppressed axes so re-anchored zero PD
        # actually yields zero command (FF doesn't bypass the mask).
        if suppress_lin:
            lin_ff = np.zeros(3)
        if suppress_orient:
            ang_ff = np.zeros(3)

        # PD + feedforward. Everything above is computed in the base frame.
        cmd_lin = lin_ff + self.kp_lin * pos_err + self.kd_lin * self.pos_err_dot
        cmd_ang = ang_ff + self.kp_ang * ang_err + self.kd_ang * self.ang_err_dot

        # Output gain + saturation. Norm-preserving clip keeps the twist
        # direction fixed when the magnitude exceeds the limit, so diagonal
        # motion stays diagonal at saturation instead of getting pulled onto
        # whichever axis is saturating first.
        cmd_lin = self._clamp(self.gain_lin * cmd_lin, self.max_lin_vel)
        cmd_ang = self._clamp(self.gain_ang * cmd_ang, self.max_ang_vel)

        # Express the twist in the consumer's frame. is_world=true → keep
        # base-frame; is_world=false → rotate into the EE frame by R_ee^-1
        # (the EE origin is the reference point, so there's no extra
        # linear/angular coupling term).
        if not self.is_world:
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
