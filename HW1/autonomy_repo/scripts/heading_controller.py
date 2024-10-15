#!/usr/bin/env python3

import numpy as np 
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState  # Corrected import statement

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__('HeadingController')
        self.kp = 2.0  

    def compute_control_with_goal(self, h_curr: TurtleBotState, h_des: TurtleBotState) -> TurtleBotControl:
        """
        Takes in the current and desired state of type TurtleBotState,
        and returns control message of type TurtleBotControl.
        """
        # print(h_curr)
        err = wrap_angle(h_des.theta - h_curr.theta)
        
        msg = TurtleBotControl()
        msg.omega = self.kp * err
        return msg

def main():
    rclpy.init(args=None)
    heading_controller = HeadingController()
    rclpy.spin(heading_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
