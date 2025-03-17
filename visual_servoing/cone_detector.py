#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation, detect_apriltags


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.declare_parameter("line_follower", True)
        self.LineFollower = self.get_parameter("line_follower").value
        
        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        ### AprilTag Detection Dictionary ###
        # self.tag_positions = {}
        # for tag_id in range(9):
        #     self.tag_positions[tag_id] = {
        #         'total_center_u': 0,
        #         'total_center_v': 0,
        #         'total_X': 0,
        #         'total_Y': 0,
        #         'count': 0
        #     }

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        """
        Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        From your bounding box, take the center pixel on the bottom
        (We know this pixel corresponds to a point on the ground plane)
        publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        convert it to the car frame.
        """

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        bottom_pixel = ConeLocationPixel()

        if self.LineFollower is True:
            y_win = 30
            y_mid = 210

            # Black mask
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[y_mid-y_win: y_mid+y_win, :] = 255
            image = cv2.bitwise_and(image, mask)

        bounding_box = cd_color_segmentation(image, "placeholder", True, self.LineFollower)

        ### AprilTag Detection (Remeber to comment out) ###
        # self.april_tag_distances(detect_apriltags(image))

        (x1, y1), (x2, y2) = bounding_box
        bottom_pixel.u = float(x1 + (x2 - x1) / 2)
        bottom_pixel.v = float(y2)
        self.cone_pub.publish(bottom_pixel)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def april_tag_distances(self, positions):
        for tag in positions:
            tag_id = tag['id']
            u, v = tag['center']
            X, Y = tag['X'], tag['Y']
            tag_position = self.tag_positions[tag_id]
            tag_position['total_center_u'] += u
            tag_position['total_center_v'] += v
            tag_position['total_X'] += X
            tag_position['total_Y'] += Y
            tag_position['count'] += 1
        
        if self.tag_positions[0]['count'] == 100:
            for tag_id, data in self.tag_positions.items():
                count = data['count']
                if count > 0:
                    avg_center_u = data['total_center_u'] / count
                    avg_center_v = data['total_center_v'] / count
                    avg_X = data['total_X'] / count
                    avg_Y = data['total_Y'] / count

                    print(f"Tag {tag_id} Averaged Center: ({avg_center_u:.2f}, {avg_center_v:.2f})")
                    print(f"Tag {tag_id} Averaged Position: (X: {avg_X:.2f}, Y: {avg_Y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
