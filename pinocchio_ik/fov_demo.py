"""Phased demo: persistent SDF memory keeps the CBF active even after the
mapped obstacle has left the wrist camera's field of view.

Initial pose: arm folded with wrist pitched ~90° so the wrist camera looks
straight down at the table.

Sequence (all twists in world frame — launch with is_world:=true):

    PUSH_DOWN_1 : EE pushes down toward the table. Camera (still looking
                  down) maps the table top into the GP-SDF. The CBF will
                  start clipping the downward velocity as the EE
                  approaches the mapped table.
    SETTLE      : hold (zero twist) so the SDF mapping integrates the last
                  frames.
    LOOK_FWD    : pitch the EE +90° around world Y. The camera rotates from
                  looking down (-Z) to looking forward (+X). The table
                  leaves the FOV.
    PUSH_DOWN_2 : repeat the downward push. The camera no longer sees the
                  table, but the SDF map remembers it — the CBF must still
                  clip. This is the property we are demonstrating.

The node logs nominal/filtered velocity magnitudes during each phase so the
operator can see the CBF biting.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# Phase entry is (label, duration_sec, kind, payload).
#   kind='twist'     : payload = ((lx,ly,lz),(ax,ay,az)) world-frame twist
#                      published on /desired_twist (goes through velocity_ik
#                      and the CBF).
#   kind='joint_vel' : payload = (v1,v2,v3,v4,v5,v6) joint velocity published
#                      DIRECTLY on /xarm6_velocity_controller/commands,
#                      bypassing both velocity_ik and the CBF. Used for the
#                      rotation phase so the CBF doesn't clip the wrist-pitch
#                      that physically can't bring any sphere closer to the
#                      mapped SDF surface.
_PHASES = [
    # 1. Push down toward the table while the camera (looking down) maps it.
    ('PUSH_DOWN_1', 6.0,  'twist',     ((0.0, 0.0, -0.04), (0.0, 0.0, 0.0))),
    ('SETTLE',      2.0,  'twist',     ((0.0, 0.0,  0.0),  (0.0, 0.0, 0.0))),
    # 2. Pull the EE back up for clearance. Twice as long + twice as fast so
    #    the stick gets WELL above the mapped table before we rotate.
    ('PULL_UP',     12.0, 'twist',     ((0.0, 0.0, +0.08), (0.0, 0.0, 0.0))),
    ('SETTLE2',     2.0,  'twist',     ((0.0, 0.0,  0.0),  (0.0, 0.0, 0.0))),
    # 3. Rotate the wrist via direct joint5 command going through the CBF.
    #    Longer phase so even if the CBF clips, the integrated rotation is
    #    large enough to swing the camera all the way to looking forward.
    ('LOOK_FWD',    8.0,  'joint_vel', (0.0, 0.0, 0.0, 0.0, +0.6, 0.0)),
    ('SETTLE3',     2.0,  'twist',     ((0.0, 0.0,  0.0),  (0.0, 0.0, 0.0))),
    # 4. Push down again. Camera no longer sees the table, but the SDF map
    #    remembers it -> CBF must still clip the downward velocity.
    ('PUSH_DOWN_2', 12.0, 'twist',     ((0.0, 0.0, -0.04), (0.0, 0.0, 0.0))),
    ('DONE',        1e9,  'twist',     ((0.0, 0.0,  0.0),  (0.0, 0.0, 0.0))),
]


class FovDemo(Node):
    def __init__(self):
        super().__init__('fov_demo')

        self.declare_parameter('startup_delay_sec', 8.0)
        self.declare_parameter('rate_hz', 50.0)
        self._startup_delay = self.get_parameter('startup_delay_sec') \
            .get_parameter_value().double_value
        rate = self.get_parameter('rate_hz').get_parameter_value().double_value

        self._pub = self.create_publisher(Twist, 'desired_twist', 10)
        # Direct joint velocity publisher to the CBF's input (nominal) topic,
        # bypassing the IK but NOT the CBF. The CBF still filters whatever we
        # send here against the persistent SDF; we just skip the IK because
        # for pure-wrist motions the pseudo-inverse IK fights the constraint.
        self._joint_pub = self.create_publisher(
            Float64MultiArray, 'nominal_joint_velocity', 10,
        )
        self._nom_sub = self.create_subscription(
            Float64MultiArray, 'nominal_joint_velocity',
            self._nominal_cb, 10,
        )
        self._filt_sub = self.create_subscription(
            Float64MultiArray, 'filtered_joint_velocity',
            self._filtered_cb, 10,
        )
        self._last_nominal_norm = 0.0
        self._last_filtered_norm = 0.0

        self._t0 = self.get_clock().now()
        self._phase_idx = 0
        self._phase_start = self._t0
        self._current_label = None

        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(1.0, self._report)

        self.get_logger().info(
            f"fov_demo armed; first phase begins {self._startup_delay:.1f}s after start."
        )

    def _now(self):
        return self.get_clock().now()

    def _elapsed_total(self):
        return (self._now() - self._t0).nanoseconds * 1e-9

    def _elapsed_phase(self):
        return (self._now() - self._phase_start).nanoseconds * 1e-9

    def _nominal_cb(self, msg: Float64MultiArray):
        self._last_nominal_norm = math.sqrt(sum(v * v for v in msg.data))

    def _filtered_cb(self, msg: Float64MultiArray):
        self._last_filtered_norm = math.sqrt(sum(v * v for v in msg.data))

    def _tick(self):
        if self._elapsed_total() < self._startup_delay:
            return

        label, dur, kind, payload = _PHASES[self._phase_idx]
        if label != self._current_label:
            self._current_label = label
            self._phase_start = self._now()
            self.get_logger().info(
                f"=== PHASE {label} ({kind}={payload}, duration={dur}s) ==="
            )
            if label == 'PULL_UP':
                self.get_logger().info(
                    "Pulling the EE back up to give the stick room to rotate."
                )
            elif label == 'LOOK_FWD':
                self.get_logger().info(
                    "Rotating joint5 directly (bypassing IK+CBF) so the "
                    "wrist physically pitches and the camera swings from "
                    "looking down to looking forward."
                )
            elif label == 'PUSH_DOWN_2':
                self.get_logger().info(
                    "Commanding the same downward push as PUSH_DOWN_1, but "
                    "the table is no longer in the camera FOV. If the SDF "
                    "is persistent the CBF will still clip the velocity."
                )

        if self._elapsed_phase() >= dur:
            self._phase_idx = min(self._phase_idx + 1, len(_PHASES) - 1)
            return

        if kind == 'twist':
            lin, ang = payload
            t = Twist()
            t.linear.x, t.linear.y, t.linear.z = lin
            t.angular.x, t.angular.y, t.angular.z = ang
            self._pub.publish(t)
        elif kind == 'joint_vel':
            msg = Float64MultiArray()
            msg.data = list(payload)
            self._joint_pub.publish(msg)
        else:
            self.get_logger().error(f"Unknown phase kind: {kind}")

    def _report(self):
        if self._current_label is None or self._current_label == 'DONE':
            return
        clip_ratio = (
            self._last_filtered_norm / self._last_nominal_norm
            if self._last_nominal_norm > 1e-6 else 1.0
        )
        self.get_logger().info(
            f"[{self._current_label} t+{self._elapsed_phase():.1f}s] "
            f"||nominal||={self._last_nominal_norm:.3f}  "
            f"||filtered||={self._last_filtered_norm:.3f}  "
            f"clip={clip_ratio:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FovDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
