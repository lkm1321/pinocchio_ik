import os
import tempfile
import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState, PointCloud2
from scipy.spatial.distance import cdist

# You may need to install this: pip install ros2-numpy
import ros2_numpy

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


class DistanceCBFNode(Node):
    def __init__(self, *controller_args):
        super().__init__('distance_cbf_node')
        self.get_logger().info('DistanceCBFNode has been started.')
        self.declare_parameter('robot_description', '')

        self.controller_args = controller_args

        hardcoded_urdf_path = '/keti_ws/src/pinocchio_ik/launch/xarm6_with_gripper_spherized.urdf'
        if os.path.exists(hardcoded_urdf_path):
            self.get_logger().info("Using hardcoded path")

            self.controller = PinocchioFKCBF.from_urdf_file(
                hardcoded_urdf_path,
                *self.controller_args,
                controlled_joint_names=[f"joint{i}" for i in range(1, 7)],
                ee_frame_name="xarm6_ee_tip",
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

        self.current_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_state = None

        self.nominal_joint_velocity_sub = self.create_subscription(
            Float64MultiArray,
            'nominal_joint_velocity',
            self.nominal_joint_velocity_callback,
            10
        )
        self.nominal_velocity = [0.] * len(self.controller.controlled_joint_idxs)

        self.filtered_joint_velocity_pub = self.create_publisher(
            Float64MultiArray,
            'filtered_joint_velocity',
            10
        )

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
                controlled_joint_names=[f"joint{i}" for i in range(1, 7)],
                ee_frame_name="xarm6_ee_tip",
            )

    def joint_state_callback(self, msg: JointState):
        if self.controller is None:
            return

        for joint_name, position in zip(msg.name, msg.position):
            if joint_name not in self.controller.model.names:
                continue

            if self.current_joint_state is None:
                self.current_joint_state = np.zeros(len(self.controller.controlled_joint_idxs))

            self.current_joint_state[self.controller.model.getJointId(joint_name) - 1] = position

        self.filter_and_publish(self.nominal_velocity)

    def nominal_joint_velocity_callback(self, msg: Float64MultiArray):
        if self.controller is None or self.current_joint_state is None:
            return

        nominal_velocity = np.array(msg.data)
        self.nominal_velocity = nominal_velocity
        self.filter_and_publish(self.nominal_velocity)

    def filter_and_publish(self, nominal_velocity):
        filtered_msg = Float64MultiArray()
        zero = [0.] * len(self.controller.controlled_joint_idxs)
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

        self.filtered_joint_velocity_pub.publish(filtered_msg)


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
