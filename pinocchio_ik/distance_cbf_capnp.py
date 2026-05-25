"""DistanceCBF node driven by a Cap'n Proto SDF client.

The Cap'n Proto twin of ``distance_cbf_service``: same CBF logic, but the SDF
query goes over the capnp RPC server hosted by ``erl_gp_sdf_ros``'s
``sdf_mapping_node`` (enabled via ``capnp_rpc.enabled: true`` in the mapping
node's YAML config) instead of a ROS2 service. Avoids the DDS round-trip and
the cross-image ``erl_gp_sdf_msgs`` dependency.

Parameters are declared on a dedicated ``sdf_client_node`` to keep the
parameter / logging surface consistent with ``distance_cbf_service``; capnp
itself doesn't need rclpy. The CBF runs alongside it under a
MultiThreadedExecutor; capnp's own reactor thread services the RPC.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pinocchio_ik.capnp_sdf_client import CapnpSdfClient
from pinocchio_ik.cbf_node import (
    DistanceCBFNode,
    _DEFAULT_EE_FRAME,
    _DEFAULT_JOINT_NAMES,
    _DEFAULT_URDF_PATH,
)


def main(args=None):
    rclpy.init(args=args)

    sdf_node = Node('sdf_client_node')
    sdf_node.declare_parameter('sdf_host', '127.0.0.1')
    sdf_node.declare_parameter('sdf_port', 51111)
    sdf_node.declare_parameter('sdf_timeout_sec', 2.0)
    sdf_node.declare_parameter('sdf_wait_for_server_sec', 5.0)
    sdf_node.declare_parameter('sdf_schema_path', '')
    sdf_node.declare_parameter('urdf_path', _DEFAULT_URDF_PATH)
    sdf_node.declare_parameter('controlled_joint_names', _DEFAULT_JOINT_NAMES)
    sdf_node.declare_parameter('ee_frame_name', _DEFAULT_EE_FRAME)
    sdf_node.declare_parameter('alpha_gain', 2.0)

    schema_path = sdf_node.get_parameter(
        'sdf_schema_path'
    ).get_parameter_value().string_value or None

    sdf_client = CapnpSdfClient(
        host=sdf_node.get_parameter('sdf_host').get_parameter_value().string_value,
        port=sdf_node.get_parameter('sdf_port').get_parameter_value().integer_value,
        timeout_sec=sdf_node.get_parameter(
            'sdf_timeout_sec'
        ).get_parameter_value().double_value,
        wait_for_server_sec=sdf_node.get_parameter(
            'sdf_wait_for_server_sec'
        ).get_parameter_value().double_value,
        schema_path=schema_path,
        logger=sdf_node.get_logger(),
    )

    urdf_path = sdf_node.get_parameter('urdf_path').get_parameter_value().string_value
    controlled_joint_names = list(
        sdf_node.get_parameter('controlled_joint_names').get_parameter_value().string_array_value
    ) or list(_DEFAULT_JOINT_NAMES)
    ee_frame_name = sdf_node.get_parameter('ee_frame_name').get_parameter_value().string_value
    alpha_gain = sdf_node.get_parameter('alpha_gain').get_parameter_value().double_value

    cbf_node = DistanceCBFNode(
        sdf_client,
        urdf_path=urdf_path,
        controlled_joint_names=controlled_joint_names,
        ee_frame_name=ee_frame_name,
        alpha_gain=alpha_gain,
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
