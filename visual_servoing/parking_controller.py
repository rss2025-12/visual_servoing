#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

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

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone",
            self.relative_cone_callback, 1)

        self.parking_distance = .75 # meters; try playing with this number!
        self.drive_velocity = 0.8
        self.velocity_scale = 1
        self.deceleration_distance = self.parking_distance + self.park_velocity**2 - self.drive_velocity**2 / (2 * 0.6 * 9.8)
        self.min_radius = self.velocity**2 / (9.8 * 0.8) # mu = [0.7, 0.9]

        self.relative_x = 0
        self.relative_y = 0
        self.e_cross = 0
        self.e_head = 0
        self.gamma = None # reference heading relative to the heading of the car (positive if car heading is greater than reference)
        self.k = None
        self.ksoft = 1 # m/s
        self.kdsteer = 0.0
        self.Kdyaw = 0.0
        self.v = None
        self.L = None # wheelbase
        self.d_prev = None
        self.d_current = None

        self.get_logger().info("Parking Controller Initialized")


    def stanley():
        """
        Given crosstrack and heading error, apply the Stanley
        controller formula.
        """
        pass


    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        if distance > self.threshold:
            # Forward logic
            distance = np.sqrt(self.relative_x**2 + self.relative_y**2)
            x_goal = self.relative_x - (self.deceleration_distance + self.relative_x / distance)
            y_goal= self.relative_y - (self.deceleration_distance +  self.relative_y / distance)
            theta_goal = np.arctan2(y_goal, x_goal)
            velocity = min(self.drive_velocity, self.velocity_scale * np.sqrt(x_goal**2 + y_goal**2))
            steering_angle = np.arctan2(np.sin(theta_goal), np.cos(theta_goal))

            self.ackerman_publisher(drive_cmd, velocity, steering_angle)
        else:
            # Reverse path planning
            pass

        self.error_publisher()


    def ackerman_publisher(self, drive_cmd, velocity, steering_angle):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.drive.speed = velocity
        drive_cmd.drive.acceleration = 0.0
        drive_cmd.drive.jerk = 0.0
        drive_cmd.drive.steering_angle = steering_angle
        drive_cmd.drive.steering_angle_velocity = 0.0

        self.drive_pub.publish(drive_cmd)


    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error = float(self.relative_x)
        error_msg.y_error = float(self.relative_y)
        error_msg.distance_error = float(np.sqrt(self.relative_x**2 + self.relative_y**2))

        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
