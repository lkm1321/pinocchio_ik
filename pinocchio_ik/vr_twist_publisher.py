import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import pinocchio_ik.triad_openvr as vr

class TwistPublisherNode(Node):
    def __init__(self):
        super().__init__('twist_publisher')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Twist Publisher Node has been started.')
        self.v = vr.triad_openvr()
        self.controller = self.v.devices["controller_1"]


    def timer_callback(self):
        msg = Twist()
        linear_vel = self.controller.get_velocity()
        angular_vel = self.controller.get_angular_velocity()
        if linear_vel is None or angular_vel is None:
            return

        # msg.linear.x = linear_vel[0]
        # msg.linear.y = linear_vel[1]
        # msg.linear.z = linear_vel[2]
        # msg.angular.x = angular_vel[0]
        # msg.angular.y = angular_vel[1]
        # msg.angular.z = angular_vel[2]

        # jumble coordinate from openvr to forward - left - up
        msg.linear.x =  linear_vel[2]
        msg.linear.y = -linear_vel[0]
        msg.linear.z = -linear_vel[1]

        msg.angular.x =  angular_vel[2]
        msg.angular.y = -angular_vel[0]
        msg.angular.z = -angular_vel[1]

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TwistPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()