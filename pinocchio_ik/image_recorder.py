"""Subscribe to a sensor_msgs/Image topic and write each frame to an mp4.

Bypasses host-side screen capture (which doesn't work for NVIDIA direct
rendering): the frames we record are the ones gz-sim already rendered into
the sensor pipeline. ROS parameters:

    image_topic    : the topic to record (required, no default)
    output_path    : where to write the mp4 (default /tmp/cbf_recording.mp4)
    fps            : encoder fps (default 25)
    duration_sec   : how long to record after the first frame, then exit
                     (default 60.0)
"""

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class ImageRecorder(Node):
    def __init__(self):
        super().__init__('image_recorder')

        self.declare_parameter('image_topic', '/scene_cam/image')
        self.declare_parameter('output_path', '/tmp/cbf_recording.mp4')
        self.declare_parameter('fps', 25.0)
        self.declare_parameter('duration_sec', 60.0)
        # Cyclone-DDS + ros_gz_bridge gives mixed QoS in practice: scene_cam
        # publishes RELIABLE, but the wrist camera bridge (shared with the
        # SDF mapping node's paint subscriber) shows up as BEST_EFFORT once
        # any subscriber requests it. Allow forcing the recorder's QoS so
        # we can match whichever side is publishing.
        self.declare_parameter('reliability', 'best_effort')

        self.topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        self.duration_sec = float(self.get_parameter('duration_sec').get_parameter_value().double_value)
        rel_str = self.get_parameter('reliability').get_parameter_value().string_value.lower()
        rel = (
            ReliabilityPolicy.RELIABLE if rel_str == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )

        self._bridge = CvBridge()
        self._writer = None
        self._first_frame_t = None
        self._frame_count = 0

        qos = QoSProfile(
            reliability=rel,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, self.topic, self._on_image, qos)

        self.get_logger().info(
            f"image_recorder: subscribing to {self.topic}, writing to "
            f"{self.output_path} for {self.duration_sec:.1f}s @ {self.fps:.0f} fps."
        )

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (w, h),
            )
            if not self._writer.isOpened():
                self.get_logger().error(
                    f"cv2.VideoWriter failed to open {self.output_path} "
                    f"({w}x{h} @ {self.fps} fps, fourcc=mp4v)"
                )
                rclpy.shutdown()
                return
            self._first_frame_t = time.monotonic()
            self.get_logger().info(
                f"image_recorder: first frame {w}x{h}; recording started."
            )

        self._writer.write(frame)
        self._frame_count += 1

        if (time.monotonic() - self._first_frame_t) >= self.duration_sec:
            self._writer.release()
            self.get_logger().info(
                f"image_recorder: wrote {self._frame_count} frames to "
                f"{self.output_path} (~{self.duration_sec:.1f}s). "
                f"Size: {os.path.getsize(self.output_path)} bytes."
            )
            self._writer = None
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node._writer is not None:
        node._writer.release()
    node.destroy_node()


if __name__ == '__main__':
    main()
