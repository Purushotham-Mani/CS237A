#!/usr/bin/env python3
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from nav_msgs.msg import OccupancyGrid
from asl_tb3_msgs.msg import TurtleBotState
from std_msgs.msg import Bool
from asl_tb3_lib.grids import StochOccupancyGrid2D

import numpy as np
import typing as T
from scipy.signal import convolve2d

class Frontier_Explorer(Node):
    def __init__(self):
        super().__init__("Explorer")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.nav_success = None
        self.state = None

        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_cb, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
    
        
    def explore(self):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.
        """

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        current_state = np.array([self.state.x, self.state.y])
        ########################### Code starts here ###########################
        
        occupied_mask = np.where(self.occupancy.probs >= 0.5, 1, 0)
        unknown_mask = np.where(self.occupancy.probs == -1, 1, 0)
        unoccupied_mask = np.where((self.occupancy.probs < 0.5) & (self.occupancy.probs>=0), 1, 0)

        kernel = np.ones((window_size, window_size)) / window_size**2
        occupied=convolve2d(occupied_mask, kernel, mode='same')
        unoccupied=convolve2d(unoccupied_mask, kernel, mode='same')
        unknown=convolve2d(unknown_mask, kernel, mode='same')

        frontier_mask = np.where((occupied==0) & (unoccupied>=0.3) & (unknown>=0.2),1,0)
        frontier_states = np.transpose(np.nonzero(np.transpose(frontier_mask)))
        frontier_states = self.occupancy.grid2state(frontier_states)

        frontier_state  = np.argmin(np.linalg.norm(frontier_states-current_state,axis=1))
        frontier_state = frontier_states[frontier_state,:]
        ########################### Code ends here ###########################

        msg = TurtleBotState()
        msg.x, msg.y = frontier_state
        self.cmd_nav_pub.publish(msg)

    
    def nav_success_cb(self, msg: Bool) -> None:
        """ Callback triggered when nav_success is updated

        Args:
            msg (Bool): updated nav_success message
        """
        self.explore()
        print("i got a message")
        self.nav_success = msg

    def state_callback(self, msg: TurtleBotState) -> None:
        """ Callback triggered when nav_success is updated

        Args:
            msg (Bool): updated nav_success message
        """
        self.state=msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        i=0
        if self.occupancy is None:
            i=1
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
        if i==1:
            self.explore()

def main():
    rclpy.init(args=None)
    print("main")
    node = Frontier_Explorer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()