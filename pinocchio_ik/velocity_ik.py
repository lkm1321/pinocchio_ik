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
import time


class VelocityIKNode(Node):
    def __init__(self):
        super().__init__('velocity_ik_node')

        # === Declare parameters ===
        self.declare_parameter('end_effector_link', 'tool_link')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('robot_description', '')
        self.declare_parameter('is_world', True)
        self.declare_parameter(
            'controlled_joint_names',
            [f"joint{i}" for i in range(1, 7)],
        )
        # Watchdog: if the control_loop hasn't successfully published in this
        # many seconds, an independent timer publishes a zero joint-velocity
        # command. Default is 3 nominal periods (=3/rate) so an occasional
        # late tick doesn't trip the watchdog but a real stall does.
        self.declare_parameter('watchdog_timeout_sec', 0.0)

        self.end_effector_link = self.get_parameter('end_effector_link').get_parameter_value().string_value
        self.base_link = self.get_parameter('base_link').get_parameter_value().string_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        is_world = self.get_parameter('is_world').get_parameter_value().bool_value
        self.controlled_joint_names = list(
            self.get_parameter('controlled_joint_names')
            .get_parameter_value().string_array_value
        ) or [f"joint{i}" for i in range(1, 7)]
        wd_param = self.get_parameter('watchdog_timeout_sec').get_parameter_value().double_value
        self.watchdog_timeout_sec = wd_param if wd_param > 0.0 else 3.0 / self.rate
        # is_world=true maps to LOCAL_WORLD_ALIGNED, NOT WORLD: the operator
        # wants the angular velocity expressed in the world frame *with the
        # EE as the centre of rotation*, not the spatial (screw) twist where
        # v is the velocity of a body-fixed point at the world origin. The
        # spatial-twist reading turns a pure ω command into a large EE
        # translation: v_EE = ω × p_EE (the lever-arm effect). LWA gives
        # [v_EE; ω] in world frame, which is the usual intuitive meaning.
        self.reference_frame = (
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED if is_world
            else pin.ReferenceFrame.LOCAL
        )
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
        # Watchdog: timestamp (monotonic seconds) of the last successful
        # joint-velocity publish from control_loop. Initialised to NOW so
        # the watchdog doesn't fire while the node is still booting up.
        self._last_publish_t = time.monotonic()
        self._watchdog_warned = False
        # Watch at 5x the nominal rate so we catch stalls quickly.
        self._watchdog_timer = self.create_timer(
            min(0.2 / self.rate, 0.05), self._watchdog_tick,
        )

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

        controlled_joints = self.controlled_joint_names
        # Skip names that aren't in this URDF (pinocchio returns njoints for
        # unknown names, which would index out-of-bounds below).
        controlled_joint_idxs = [
            self.model.getJointId(name) - 1
            for name in controlled_joints
            if self.model.existJointName(name)
        ]
        if len(controlled_joint_idxs) != len(controlled_joints):
            missing = [n for n in controlled_joints if not self.model.existJointName(n)]
            self.get_logger().warn(
                f"controlled_joint_names contained joints not in the URDF: {missing}"
            )

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

        # Successful publish — refresh the watchdog deadline.
        self._last_publish_t = time.monotonic()
        if self._watchdog_warned:
            self.get_logger().info("velocity_ik: control loop recovered.")
            self._watchdog_warned = False

    # ------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------
    def _watchdog_tick(self):
        """Publish a zero joint-velocity command if control_loop hasn't
        successfully published within `watchdog_timeout_sec`.

        Reasons control_loop might be late: model/joint_state still missing,
        IK exception, the executor being starved by a long-running callback
        elsewhere. In any of those cases we'd rather command zero than have
        the robot keep tracking the last (now stale) velocity.
        """
        now = time.monotonic()
        if now - self._last_publish_t <= self.watchdog_timeout_sec:
            return

        n = len(self.controlled_joint_names)
        zero_msg = Float64MultiArray()
        zero_msg.data = [0.0] * n
        self.joint_velocity_pub.publish(zero_msg)

        if not self._watchdog_warned:
            self.get_logger().warn(
                f"velocity_ik: control loop missed "
                f"{now - self._last_publish_t:.3f}s (timeout "
                f"{self.watchdog_timeout_sec:.3f}s); publishing zero."
            )
            self._watchdog_warned = True


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