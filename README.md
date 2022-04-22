# DeepRL_595_FinalProject
Karter Krueger, Pratyush Kumar Sahoo, Himanshu Gautam, Steven Hyland

## WPI DS595 - Reinforcement Learning

This project aims to use DQN (Deep Q-Learning Network) to learn how to pick objects in a simulated environment using a Franka robotic arm. 
The simulation is done in Gazebo and communicates through ROS with our network. We have used the DeepAI Atari DQN environment and file structure 
as a basis for our own network.

## Notes:
karterk@karterk-ms-7c37:~/Documents/Classes/DS595_Project4/DeepRL_595_FinalProject/catkin_rlp/src$ rosservice call /gazebo/reset_world 

karterk@karterk-ms-7c37:~/Documents/Classes/DS595_Project4/DeepRL_595_FinalProject/catkin_rlp/src$ rostopic pub /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: cube_red, pose: { position: { x: .5, y: 0, z: .07 }, orientation: {x: 0, y: 0.0, z: 0, w: 0.0 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'
publishing and latching message. Press ctrl-C to terminate



##TO DO
