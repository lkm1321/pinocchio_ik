import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as Rot
import pinocchio_ik.triad_openvr as vr


# --- OpenVR frame -> robot frame (forward-left-up) -------------------------
# (x, y, z)_robot = (z, -x, -y)_vr
VR_TO_ROBOT = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
])


class TwistPublisherNode(Node):
    # Trigger axis is analog (0.0 released → 1.0 fully pressed); treat any
    # press past this fraction as "deadman engaged".
    TRIGGER_THRESHOLD = 0.5

    def __init__(self):
        super().__init__('twist_publisher')

        # is_world selects the frame the Twist is published in and MUST
        # match velocity_ik's is_world. True → base frame (no tf needed).
        # False → rotate into the EE frame via R_ee^-1 from tf.
        self.declare_parameter('is_world', True)
        self.declare_parameter('ee_frame', 'link6')
        self.declare_parameter('base_frame', 'base_link')
        self.is_world = bool(self.get_parameter('is_world').value)
        self.ee_frame = self.get_parameter('ee_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # Trackpad axis-mask: clicking the left half of the trackpad zeroes
        # the angular component (position-only); clicking the right half
        # zeroes the linear component (orientation-only). Center / no click
        # leaves both components live.
        self.declare_parameter('trackpad_deadzone', 0.3)
        self.trackpad_deadzone = float(self.get_parameter('trackpad_deadzone').value)

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Only spin up the tf listener when we actually need EE rotation.
        if not self.is_world:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = None

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(
            f'Twist Publisher Node has been started '
            f'(twist frame: {"world" if self.is_world else "ee"}).')
        self.v = vr.triad_openvr()
        self.controller = self.v.devices["controller_1"]

    def _ee_rot(self):
        """Look up R_base_ee from tf, or None if unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'TF {self.base_frame} -> {self.ee_frame} unavailable: {ex}',
                throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation
        return Rot.from_quat([q.x, q.y, q.z, q.w])

    def timer_callback(self):
        # Deadman: only forward live twists while the trigger is held.
        # Publish zero on release so the downstream controller halts
        # rather than holding the last non-zero command.
        inputs = self.controller.get_controller_inputs()
        trigger_pressed = bool(inputs) and inputs.get('trigger', 0.0) > self.TRIGGER_THRESHOLD
        if not trigger_pressed:
            self.publisher.publish(Twist())
            return

        linear_vel = self.controller.get_velocity()
        angular_vel = self.controller.get_angular_velocity()
        if linear_vel is None or angular_vel is None:
            self.publisher.publish(Twist())
            return

        # OpenVR frame -> robot base frame.
        lin = VR_TO_ROBOT @ np.asarray([linear_vel[0], linear_vel[1], linear_vel[2]], dtype=float)
        ang = VR_TO_ROBOT @ np.asarray([angular_vel[0], angular_vel[1], angular_vel[2]], dtype=float)

        # Trackpad axis-mask. Apply before any frame rotation so the mask
        # is interpreted in the same physical sense regardless of is_world.
        suppress_orient, suppress_lin = self._axis_mask(inputs)
        if suppress_lin:
            lin = np.zeros(3)
        if suppress_orient:
            ang = np.zeros(3)

        if not self.is_world:
            ee_rot = self._ee_rot()
            if ee_rot is None:
                self.publisher.publish(Twist())
                return
            lin = ee_rot.inv().apply(lin)
            ang = ee_rot.inv().apply(ang)

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = map(float, lin)
        msg.angular.x, msg.angular.y, msg.angular.z = map(float, ang)
        self.publisher.publish(msg)

    def _axis_mask(self, inputs):
        """Return (suppress_orient, suppress_lin) from the trackpad click.

        Left half clicked → position-only (angular suppressed).
        Right half clicked → orientation-only (linear suppressed).
        No click / center click → neither suppressed.
        """
        if not inputs or not inputs.get('trackpad_pressed', False):
            return False, False
        x = float(inputs.get('trackpad_x', 0.0))
        if x < -self.trackpad_deadzone:
            return True, False
        if x >  self.trackpad_deadzone:
            return False, True
        return False, False


def main(args=None):
    rclpy.init(args=args)
    node = TwistPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
