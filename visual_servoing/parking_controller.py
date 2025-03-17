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
        self.declare_parameter("anglular_error_threshold", np.pi/12) # Boundary for when the car is considered pointed in the right direction
        self.declare_parameter("distance_error_thresholds", [-0.2, 0.2]) # Boundaries for when the car is considered [too close, too far] from the cone for angle adjustment manuevers
        self.declare_parameter("parking_velocity", 0.8)
        self.declare_parameter("drive_velocity", 1.0)
        self.declare_parameter("parking_distance", 0.75)
        self.declare_parameter("line_follower", True)

        # Getting parameters #
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # Set in launch file; different for simulator vs racecar
        self.angular_error_threshold = self.get_parameter("anglular_error_threshold").value
        self.distance_error_threshold = self.get_parameter("distance_error_thresholds").get_parameter_value().double_array_value
        self.parking_velocity = self.get_parameter("parking_velocity").value
        self.parking_distance = self.get_parameter("parking_distance").value
        self.line_follower = self.get_parameter("line_follower").value
        self.VELOCITY = self.get_parameter("drive_velocity").value
        self.lookahead_distance = 2
        self.parking_distance = self.parking_distance if not self.line_follower else 0.1

        # Publishers and subscribers #
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)
        self.create_subscription(ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        # Constants #
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0.1
        self.camera_offset = 0.06 # Offset of left eye relative to the middle of the camera
        self.relative_x = 0
        self.relative_y = 0
        self.reverse = False
        self.previous_x_error = 0
        self.previous_heading_error = 0
        self.previous_time = Time()
        self.integrated_dist_error = 0
        self.integral_max, self.integral_min = -10, 10
        self.add_on_set_parameters_callback(self.parameters_callback) # Enable setting parking_velocity and parking_distance from the command line

        self.get_logger().info("Parking Controller Initialized")

    def get_yaw_from_quaternion(self, quaternion):
        """
        Convert quaternion to yaw (z-axis rotation).
        """
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def compute_steering_angle(self, x_robot, y_robot, x_wp, y_wp, yaw_robot):
        """
        Compute the steering angle for pure pursuit.
        """
        dx = x_wp - x_robot
        dy = y_wp - y_robot
        angle_to_wp = np.arctan2(dy, dx)

        # Calculate the steering angle (geometry of the pursuit)
        angle_diff = angle_to_wp
        # Pure Pursuit formula for steering angle (in radians)
        steering_angle = np.arctan2(2.0 * np.sin(angle_diff), self.lookahead_distance)
        return steering_angle

    def relative_cone_callback(self, msg):
        # self.relative_x = msg.x_pos 
        # self.relative_y = msg.y_pos + self.camera_offset
        # drive_cmd = AckermannDriveStamped()

        # # Calculating errors #
        # distance = np.sqrt(self.relative_x**2 + self.relative_y**2)
        # distance_error = distance - self.parking_distance
        # angle_error = np.arctan2(self.relative_y, self.relative_x)
        # x_error = self.relative_x - self.parking_distance

        # # PID #
        # current_time = self.get_clock().now()
        # dt = (current_time.nanoseconds - self.previous_time.nanoseconds) / (10**9)

        # # Logic and booleans #
        # if (distance_error <= self.distance_error_threshold[0]):
        #     self.reverse = True
        # elif distance_error >= self.distance_error_threshold[1]:
        #     self.reverse = False

        # heading = -angle_error if self.reverse else angle_error
        # high_angular_error = (np.abs(angle_error) > self.angular_error_threshold)

        # # Set velocity and heading #
        # if not high_angular_error and distance_error > self.distance_error_threshold[0] and distance_error < self.distance_error_threshold[1]:
        #     velocity = self.parking_velocity * ((x_error)/self.parking_distance)
        #     heading /= 2
        # else:
        #     velocity = -self.parking_velocity if self.reverse else self.parking_velocity

        # # PID #
        # p_error = self.Kp*heading
        # d_error = self.Kd*(heading - self.previous_heading_error)/dt
        # self.integrated_dist_error += max(min(dt * self.previous_x_error, self.integral_max), self.integral_min)
        # i_error = self.Ki*self.integrated_dist_error
        
        # self.previous_x_error = x_error
        # self.previous_heading_error = heading
        # self.previous_time = current_time
        
        # heading = p_error + d_error + i_error

        # # Publishing Ackermann #
        # drive_cmd.header.frame_id = "base_link"
        # drive_cmd.header.stamp = current_time.to_msg()
        # drive_cmd.drive.steering_angle = heading
        # drive_cmd.drive.speed = velocity
        # self.drive_pub.publish(drive_cmd)
        # self.error_publisher()
        
        
        x_robot = 0
        y_robot = 0
        self.relative_x = msg.x_pos 
        self.relative_y = msg.y_pos + self.camera_offset
        yaw_robot = np.arctan2(self.relative_y, self.relative_x)

        # Find the closest waypoint
        x_wp, y_wp = (self.relative_x, self.relative_y)

        # Compute the lookahead point
        dx = x_wp - x_robot
        dy = y_wp - y_robot
        # distance = np.sqrt(dx**2 + dy**2)
        self.get_logger().info(f'{x_wp,y_wp}')

        # Compute the curvature and velocity command
        steering_angle = self.compute_steering_angle(x_robot, y_robot, x_wp, y_wp, yaw_robot)
        self.get_logger().info(f'{steering_angle}')
        speed = self.VELOCITY  # Forward velocity in m/s (You can adjust this as needed)

        # Create an AckermannDriveStamped message to publish the steering and speed
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle  # Steering angle in radians
        drive_msg.drive.speed = speed  # Linear velocity in m/s
        drive_msg.header.stamp = self.get_clock().now().to_msg()  # Timestamp for the message
        drive_msg.header.frame_id = 'base_link'  # Frame ID

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


    def parameters_callback(self, params):
        for param in params:
            if param.name == 'parking_velocity':
                self.parking_velocity = param.value
                self.get_logger().info(f"Updated parking velocity to {self.parking_velocity}")
            elif param.name == 'parking_distance':
                self.parking_distance = param.value
                self.get_logger().info(f"Updated parking distance to {self.parking_distance}")
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
