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
        self.relative_x = 0
        self.relative_y = 0

        self.get_logger().info("Parking Controller Initialized")

    # def get_R_point(robot_loc, target_loc, R):
    #     """
    #     Given the current location of the robot, 
    #     get the point on the direct path from the
    #     robot to the goal at a distance R
    #     """
    #     # TODO: revise this to get the R radius circle intersection with the shortest path to the goal
    #     # x,y = robot_loc
    #     # x_goal, y_goal = target_loc
    #     # theta = np.arctan2(y_goal-y, x_goal-x)
    #     # return (x+R*np.cos(theta), y+R*np.sin(theta))
    #     pass


    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd
        # using pure pursuit
        
        # x_target, y_target = get_R_point((0,0), self.relative_x, self.relative_y)
        
        # L = 13 # in
        # lfw = L*2/3
        # Lfw = np.linalg.norm([(x_target+0+lfw), (y_target)])
        # R = 0.5 # m
        # eta = np.arctan2(y_target, x_target, R)
        # drive_cmd = -np.arctan(L*np.sin(eta))/(Lfw/2+lfw*np.cos(eta))
        x, y = self.relative_x, self.relative_y
        L = 0.5 # lookahed distance
        # r = L**2/(2*np.abs(self.relative_y))
        g = abs(y/x)
        r = L**2/(2*abs(g*y))
        drive_cmd = 1/r

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################
        
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()