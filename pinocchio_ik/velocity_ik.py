#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String

import numpy as np
np.set_printoptions(precision=3)
import pinocchio as pin
from rclpy.qos import QoSProfile, DurabilityPolicy
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import tempfile


class VelocityIKNode(Node):
    def __init__(self):
        super().__init__('velocity_ik_node')

        # === Declare parameters ===
        self.declare_parameter('end_effector_link', 'tool_link')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('robot_description', '')
        self.declare_parameter('is_world', True)

        self.end_effector_link = self.get_parameter('end_effector_link').get_parameter_value().string_value
        self.base_link = self.get_parameter('base_link').get_parameter_value().string_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        is_world = self.get_parameter('is_world').get_parameter_value().bool_value
        self.reference_frame = pin.ReferenceFrame.WORLD if is_world else pin.ReferenceFrame.LOCAL
        self.force_orientation = False

        # === Load URDF ===
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
            # Temporarily disable control until model is ready
            self.model = None
            self.visual_model = None
            self.collision_model = None

        # === Subscribers and Publishers ===
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.desired_twist_sub = self.create_subscription(
            Twist, 'desired_twist', self.desired_twist_callback, 10)

        self.joint_velocity_pub = self.create_publisher(
            Float64MultiArray, 'joint_velocity_cmd', 10)

        self.joint_position_pub = self.create_publisher(
            Float64MultiArray, 'joint_position_cmd', 10)

        # === Internal state ===
        self.q = None
        self.v = None
        self.q_pin = None
        self.v_des = np.zeros(6)  # Desired end-effector twist
        self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
        self.last_command_time = self.get_clock().now()

    # ------------------------------------------------------
    # URDF setup
    # ------------------------------------------------------
    def urdf_callback(self, msg: String):
        urdf_str = msg.data
        self.get_logger().info("Received URDF from /robot_description topic.")
        self.init_robot_model(urdf_str)
        # self.destroy_subscription(self.urdf_sub)

    def init_robot_model(self, urdf_str: str):
        """Initialize Pinocchio model from URDF string."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as urdf_file:
                urdf_file.write(urdf_str)
                self.get_logger().info(f"Temporary URDF file created at {urdf_file.name}")
                urdf_file.flush()
                self.model, self.visual_model, self.collision_model = pin.buildModelsFromUrdf(
                    urdf_file.name
                )
            self.data = self.model.createData()
            self.nq = self.model.nq
            self.nv = self.model.nv
            self.frame_id = self.model.getFrameId(self.end_effector_link)
            self.get_logger().info(f"✅ Model loaded with {self.nq} joints and {self.nv} tangents. EE frame: {self.end_effector_link}")
            self.get_logger().info(f"Model joint names: {list(self.model.names)}")
            self.get_logger().info(f"Number of joint names: {len(self.model.names)}")



            # self.get_logger().info(f"Model frame names: {list(self.model.frames)}")
            self.get_logger().info(f"End-effector frame ID: {self.frame_id}")

            # self.get_logger().info(f"{len(list(self.model.names))}")

            try:
                self.viz = Visualizer(self.model, collision_model=self.collision_model, visual_model=self.visual_model)
                self.viz.initViewer(loadModel=True)
                self.viz.displayVisuals(True)
                self.viz.displayCollisions(False)
            except:
                self.viz = None


        except Exception as e:
            self.get_logger().error(f"Failed to build model from URDF: {e}")
            self.model = None


    # ------------------------------------------------------
    # Joint & Twist callbacks
    # ------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        if self.model is None:
            return  # Model not ready yet
        
        for joint_name, position, velocity in zip(msg.name, msg.position, msg.velocity):
            if not self.model.existJointName(joint_name):
                self.get_logger().warn(f"Joint '{joint_name}' not found in model.")
                return
            jointId = self.model.getJointId(joint_name)
            if self.q is None:
                self.q = np.zeros(self.nq)
                self.v = np.zeros(self.nv)
            # Important! jointId 0 is universe
            self.q[jointId - 1] = position
            self.v[jointId - 1] = velocity

        # Hack: initialize q_pin on first callback
        # if self.q_pin is None:
        #     self.q_pin = self.q.copy()
        self.q_pin = self.q.copy()

        # self.get_logger().info(f"Current joints: {self.q}")

        # if len(msg.position) == self.model.nq:
        #     self.q = np.array(msg.position)
        #     self.have_joint_state = True
        # else:
        #     self.get_logger().warn(f"JointState size mismatch: expected {self.model.nq}, got {len(msg.position)}")

    def desired_twist_callback(self, msg: Twist):
        self.v_des = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ])
        self.last_command_time = self.get_clock().now()

    # ------------------------------------------------------
    # Control loop
    # ------------------------------------------------------
    def control_loop(self):
        if not (self.model and self.q is not None and self.v is not None):
            return
        if self.get_clock().now() - self.last_command_time > rclpy.duration.Duration(seconds=0.1):
            # No recent command, stop
            self.v_des = np.zeros(6)

        # TODO: fix hardcode
        controlled_joints = [
            f"joint{i}" for i in range(1, 7)
        ]

        controlled_joint_idxs = [self.model.getJointId(name) - 1 for name in controlled_joints]

        # Forward kinematics and Jacobian
        data_source = self.q
        pin.computeJointJacobians(self.model, self.data, data_source)
        J_frame_world = pin.getFrameJacobian(self.model, self.data, self.frame_id, self.reference_frame)

        if self.force_orientation:
            pin.forwardKinematics(self.model, self.data, data_source)
            pin.updateFramePlacements(self.model, self.data)

            current_frame_placement = self.data.oMf[self.frame_id]

            # HACK! need to move to hand set code
            # this code tries to maintain a constant end effector heading
            desired_rotation = np.array([
                [1, 0., 0.],
                [0., -1, 0.],
                [0., 0., -1]
            ])
            rotation_error_matrix = desired_rotation @ current_frame_placement.rotation.T
            rotation_error_vector = pin.log3(rotation_error_matrix)
            # self.get_logger().info(f"rotation error vector: {rotation_error_vector}")

            if self.reference_frame == pin.ReferenceFrame.LOCAL:
                self.v_des[3:] = rotation_error_vector * 0.1  # P gain on orientation error

            else:
                self.v_des[3:] = current_frame_placement.rotation @ rotation_error_vector * 0.1  # P gain on orientation error


        J_frame_world_controlled = J_frame_world[:, controlled_joint_idxs]
        J = J_frame_world_controlled
        control, residuals, rank, s = np.linalg.lstsq(J, self.v_des, rcond=None)
        # self.get_logger().info(f"singular value for ik: {s}")

        #TODO: check the next state instead
        # if s[0] / s[-1] > 1e2:
        #     self.get_logger().warn(f"Nearing singular configuration")
        #     control = np.zeros(len(controlled_joints))

        dq = np.zeros(self.nq)
        for i, jointIdx in enumerate(controlled_joint_idxs):
            dq[jointIdx] = control[i]

        if self.viz is not None:
            self.viz.display(data_source)
            self.viz.drawFrameVelocities(self.frame_id, v_scale=0.5, color=0xff0000)


        # Hack: integrate to get new q
        self.q_pin = pin.integrate(self.model, self.q_pin, dq * (1.0 / self.rate))

        # Publish joint velocity command
        velocity_msg = Float64MultiArray()

        velocity_msg.data = control.tolist()
        # msg.data = control.tolist().reverse()
        self.joint_velocity_pub.publish(velocity_msg)

        position_msg = Float64MultiArray()
        position_msg.data = self.q_pin[controlled_joint_idxs].tolist()
        self.joint_position_pub.publish(position_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()