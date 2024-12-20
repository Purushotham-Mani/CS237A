# CS237A

In this repository, I maintain my Homework submissions for the course CS237A: Principles of Robot Autonomy I (Stanford). Read this in top-down order to see how the each code fits into the robot autonomy stack

## A* Motion Planning & Path Smoothing (using Cubic Spline)

![Screenshot from 2024-11-03 01-07-22](https://github.com/user-attachments/assets/27b082bb-68ec-4b11-b5e3-06c36926e411)

Code can be found in HW1/P1_astar.py and corresponding visualisation can be done using HW1/sim_astar.ipynb

## Sampling Based Algorithms: 
### Rapidly-exploring Random Trees:

![Screenshot from 2024-11-03 01-10-25](https://github.com/user-attachments/assets/41be230e-65c7-44bc-8522-8b9b4714669f)

Code can be found in HW1/P2_rrt.py and corresponding visualisation can be done using HW1/sim_rrt.ipynb

## Trajectory Optimzation for Turtlebot:

Computing dynamically feasible trajectory given final and initial states (Nominal control u and state x)

![Screenshot from 2024-11-03 01-15-46](https://github.com/user-attachments/assets/1e413171-3d7a-4771-8985-bf1272b3b049)

Code can be found in HW1/P3_trajectory_optimization.ipynb


## Differential Flatness of differential drive bot:

Exploiting differential flatness of a drifferential drive robot to compute output trajectory and the corresponding control required to achieve the trajectory.

![Screenshot from 2024-11-03 01-31-43](https://github.com/user-attachments/assets/d3002719-e643-4612-8204-8a64777f8d30)

Code can be found in HW2/P2_differential_flatness.py and HW2/P2_trajectory_tracking.py and corresponding visualisation can be done using HW2/sim_trajectory.ipynb

## Gain-Scheduled LQR to a Planar Quadrotor:

![image](https://github.com/user-attachments/assets/8a1d10dc-5593-4613-99f6-86a1da84113d)

Code can be found in HW2/P3_gain_scheduled_LQR.ipynb

## Camera Caliberation: Extrinsics Caliberation

![image](https://github.com/user-attachments/assets/3863c5a8-eb65-4e17-bcb4-4ab0a2baad7c)


## Object Detection: Using Torchvision

![image](https://github.com/user-attachments/assets/ef677154-aa89-48c5-856a-d013eee6bba5)

## Iterative Closest Point Algorithm for point Cloud Registration:

![image](https://github.com/user-attachments/assets/be85aed7-cacb-460a-86cd-3f31c129b22f)

## Bundle Adjustment SLAM using GTSAM

![Screenshot from 2024-11-20 15-58-28](https://github.com/user-attachments/assets/7d85742e-fd95-4a98-87e8-a504e1c8d154)

## Frontier Exploration

Robot builds a StochOccupancyGrid using the LiDAR measurements and explores the environment its placed in until it has mapped the entire space.

Controller (flatness-based), Planning (A* and path smoothening)

[frontier_exploration.webm](https://github.com/user-attachments/assets/0c842465-11bb-421b-8808-8dfbf9d3683e)

Code can be found in autonomy_repo/scripts/frontier_explorer.py and autonomy_repo/scripts/navigator.py 






