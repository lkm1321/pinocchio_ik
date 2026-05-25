"""DistanceCBF node driven by a service-call SDF (erl_gp_sdf_msgs/SdfQuery).

Mirrors ``cbf_node.main`` but swaps the analytic ``table_sdf`` for a
``RosSdfClient`` that fetches signed distance + gradient from an external
service.

The service name and the URDF/joint config are declared on a dedicated
``sdf_client_node`` (separate from the CBF node so the SDF client lives in
its own mutually-exclusive callback group). The launch file sets parameters
on this node.
"""

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pinocchio_ik.cbf_node import (
    DistanceCBFNode,
    _DEFAULT_EE_FRAME,
    _DEFAULT_JOINT_NAMES,
    _DEFAULT_URDF_PATH,
)
from pinocchio_ik.ros_sdf_client import RosSdfClient


def main(args=None):
    rclpy.init(args=args)

    # Dedicated node for the SDF service client. Kept separate from the CBF
    # node so the client lives in its own mutually-exclusive callback group,
    # isolated from the CBF node's subscriber callbacks. Under
    # MultiThreadedExecutor this lets the blocking SDF request inside the
    # CBF callback be served on a different executor thread without
    # deadlocking.
    sdf_node = Node('sdf_client_node')
    sdf_node.declare_parameter('sdf_srv_pkg', 'erl_gp_sdf_msgs')
    sdf_node.declare_parameter('sdf_timeout_sec', 2.0)
    sdf_node.declare_parameter('sdf_wait_for_service_sec', 5.0)
    sdf_node.declare_parameter('urdf_path', _DEFAULT_URDF_PATH)
    sdf_node.declare_parameter('controlled_joint_names', _DEFAULT_JOINT_NAMES)
    sdf_node.declare_parameter('ee_frame_name', _DEFAULT_EE_FRAME)

    sdf_cb_group = MutuallyExclusiveCallbackGroup()

    sdf_client = RosSdfClient(
        sdf_node,
        service_name='sdf_query',  # remap in the launch file
        srv_pkg=sdf_node.get_parameter('sdf_srv_pkg').get_parameter_value().string_value,
        timeout_sec=sdf_node.get_parameter('sdf_timeout_sec').get_parameter_value().double_value,
        wait_for_service_sec=sdf_node.get_parameter('sdf_wait_for_service_sec').get_parameter_value().double_value,
        callback_group=sdf_cb_group,
    )

    urdf_path = sdf_node.get_parameter('urdf_path').get_parameter_value().string_value
    controlled_joint_names = list(
        sdf_node.get_parameter('controlled_joint_names').get_parameter_value().string_array_value
    ) or list(_DEFAULT_JOINT_NAMES)
    ee_frame_name = sdf_node.get_parameter('ee_frame_name').get_parameter_value().string_value

    cbf_node = DistanceCBFNode(
        sdf_client,
        urdf_path=urdf_path,
        controlled_joint_names=controlled_joint_names,
        ee_frame_name=ee_frame_name,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(sdf_node)
    executor.add_node(cbf_node)
    try:
        executor.spin()
    finally:
        cbf_node.destroy_node()
        sdf_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
