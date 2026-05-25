"""Publish a constant desired EE twist on /desired_twist.

Tiny helper used by the CBF demo to drive velocity_ik with a fixed Cartesian
velocity. The intended motion is set via ROS parameters (linear/angular x,y,z
in the same frame velocity_ik is configured for — pass is_world:=true in the
launch if you want these in the world frame).
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CannedTwist(Node):
    def __init__(self):
        super().__init__('canned_twist')

        self.declare_parameter('linear_x', 0.0)
        self.declare_parameter('linear_y', 0.0)
        self.declare_parameter('linear_z', 0.0)
        self.declare_parameter('angular_x', 0.0)
        self.declare_parameter('angular_y', 0.0)
        self.declare_parameter('angular_z', 0.0)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('startup_delay_sec', 5.0)

        p = lambda n: self.get_parameter(n).get_parameter_value().double_value
        self._twist = Twist()
        self._twist.linear.x = p('linear_x')
        self._twist.linear.y = p('linear_y')
        self._twist.linear.z = p('linear_z')
        self._twist.angular.x = p('angular_x')
        self._twist.angular.y = p('angular_y')
        self._twist.angular.z = p('angular_z')

        rate = p('rate_hz')
        delay = p('startup_delay_sec')
        self._pub = self.create_publisher(Twist, 'desired_twist', 10)

        self._t0 = self.get_clock().now()
        self._delay_sec = delay
        self._timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"canned_twist: linear=({self._twist.linear.x:.3f},{self._twist.linear.y:.3f},"
            f"{self._twist.linear.z:.3f}) angular=({self._twist.angular.x:.3f},"
            f"{self._twist.angular.y:.3f},{self._twist.angular.z:.3f}) @ {rate:.0f} Hz "
            f"(starting after {delay:.1f}s)"
        )

    def _tick(self):
        dt = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        if dt < self._delay_sec:
            return
        self._pub.publish(self._twist)


def main(args=None):
    rclpy.init(args=args)
    node = CannedTwist()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
