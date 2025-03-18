#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult
from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        # Declaring parameters #
        self.declare_parameter("drive_topic", "/vesc/high_level/input/nav_0")
        self.declare_parameter("distance_error_thresholds", [-0.2, 0.2]) # Boundaries for when the car is considered [too close, too far] from the cone for angle adjustment manuevers
        self.declare_parameter("drive_velocity", 1.0)
        self.declare_parameter("parking_distance", 0.5)
        self.declare_parameter("line_follower", True)

        # Getting parameters #
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # Set in launch file; different for simulator vs racecar
        self.distance_error_threshold = self.get_parameter("distance_error_thresholds").get_parameter_value().double_array_value
        self.parking_distance = self.get_parameter("parking_distance").value
        self.line_follower = self.get_parameter("line_follower").value
        self.VELOCITY = self.get_parameter("drive_velocity").value

        # Publishers and subscribers #
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)
        self.create_subscription(ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        # Constants #
        self.L = 0.35
        self.camera_offset = 0.06 # Offset of left eye relative to the middle of the camera
        self.relative_x = 0
        self.relative_y = 0
        
        self.lookahead_distance = 2.1
        if self.line_follower:
            self.parking_distance = 0.1
            self.lookahead_distance = 2.1
            
        self.get_logger().info("Parking Controller Initialized")

    def compute_steering_angle(self, x_robot, y_robot, x_wp, y_wp, yaw_robot):
        """
        Compute the steering angle for pure pursuit.
        """
        dx = x_wp - x_robot
        dy = y_wp - y_robot
        angle_to_wp = np.arctan2(dy, dx)

        angle_diff = angle_to_wp
        steering_angle = np.arctan2(2.0 * self.L * np.sin(angle_diff), self.lookahead_distance)
        
        return steering_angle

    def relative_cone_callback(self, msg):
        x_robot = 0
        y_robot = 0
        self.relative_x = msg.x_pos 
        self.relative_y = msg.y_pos + self.camera_offset
        yaw_robot = np.arctan2(self.relative_y, self.relative_x)
        
        target_dist = np.sqrt(self.relative_x**2+self.relative_y**2)
        dist_error = target_dist - self.parking_distance
        
        x_wp, y_wp = (self.relative_x, self.relative_y)

        steering_angle = self.compute_steering_angle(x_robot, y_robot, x_wp, y_wp, yaw_robot)
        velocity = self.VELOCITY
        
        if not self.line_follower:
            if (dist_error < self.distance_error_threshold[1] and dist_error > self.distance_error_threshold[0]):
                steering_angle = 0.0
                velocity = 0.0
            elif (dist_error < self.distance_error_threshold[0]):
                steering_angle *= -1
                velocity *= -1
        
        self.get_logger().info(f'dist_error={dist_error}')
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle  # Steering angle in radians
        drive_msg.drive.speed = velocity  # Linear velocity in m/s
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'  # Frame ID
        self.error_publisher()
        self.drive_pub.publish(drive_msg)

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error =  self.relative_y
        error_msg.distance_error = np.hypot(error_msg.x_error, error_msg.y_error)

        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
