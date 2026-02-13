# llmotion
LLM-driven motion planning and HIL system


# Running a Hardware-in-the-Loop (HIL) LLM Driven Autonomous Driving Framework with CARLA and ROS2

## Purpose
Guiding you through the steps to set up, run, and validate the HIL simulated tests using CARLA and ROS2.

## Context
This SOP will need to be followed anytime someone conducts HIL tests that involve hardware (Jetson Xavier and sensors) connecting to CARLA in order to evaluate planning and control algorithms.

## Definitions
HIL: Hardware-in-the-Loop
ROS: Robotic Operating System

## Tools and Equipment
Jetson Xavier/Orin (with ROS2 Foxy and llm models files)
Desktop：Ubuntu 22.04, ROS2 ‘Humble’, Python 3.8, Carla 9.15
LLMs：gpt3.5-turbo,  llama3.1, llama3.2, qwen
Power supply for Xavier
Ethernet cable / WiFi connection
HIL: Hardware-in-the-Loop
A testing method where real hardware (Jetson Xavier, sensors, or planners) interacts with a simulated environment (CARLA). HIL allows validation of planning, control, and decision-making algorithms in real time before deploying to a real vehicle.

ROS: Robot Operating System
A middleware framework that enables communication between modular components (nodes). In this HIL setup, ROS2 is used to exchange planning trajectories, odometry, and vehicle control commands between Xavier and the Desktop.

CARLA-ROS Bridge
A ROS2 interface layer that connects CARLA simulator topics to ROS topics.
It translates simulation data (e.g., /carla/hero/odometry) into ROS standard messages and forwards control commands back to the simulator.
Local Planner (Lattice / Hybrid A*)
A ROS2 planning node running on Jetson Xavier using real-time feedback (odometry) to generate a feasible trajectory published on /planning/trajectory.
Controller Node
A ROS2 node running on the Desktop that consumes /planning/trajectory and computes throttle, brake, and steering commands to control the simulated vehicle in CARLA.
Talk2Drive LLM Interface
A lightweight Python interface (running on Xavier) that processes natural-language commands using an LLM model. Commands modify goal states or constraints for the Local Planner in real time.

## How To Run Hardware-in-the-Loop with CARLA and ROS
1.	Power on hardware and verify that there is a network connection.
2.	On the Jetson Xavier, source the correct conda environment and ROS2.
(LLM + Local Planner environment)
3.	On the Desktop, start the CARLA simulator.
4.	Source ROS2 Humble and the CARLA workspace on the Desktop.
5.	Launch and verify the CARLA–ROS2 bridge.
6.	Start the Controller Node on the Desktop.
(Consumes /planning/trajectory and publishes /carla/hero/vehicle_control_cmd)
7.	Once the CARLA environment is set up, run the LLM interface on the Jetson Xavier.
Example:
time python main.py --no_mic --llm_model_name LLM_MODEL_NAME -I "YOUR PROMPT HERE"
8.	Start the Local Planner on the Jetson Xavier.
(Subscribes to /carla/hero/odometry, publishes /planning/trajectory)
9.	Confirm ROS2 communication across devices.
Ensure topics are visible on both machines:

●	/carla/hero/odometry
●	/planning/trajectory
●	/carla/hero/vehicle_control_cmd
10.	Validate the Full HIL Loop.
11.	Once all components are running:
●	CARLA publishes odometry.
●	ROS2 bridge forwards odometry to Xavier.
●	Local Planner generates a trajectory.
●	Controller consumes the trajectory and commands CARLA.
●	LLM modifies planner goals based on user input.
●	The vehicle in CARLA should now drive along the planner’s path.
12.	Logging and Data Recording
13.	Ending the HIL Test

Documentation
After each test, log:
1.	Date and time
2.	Test types
3.	System execution 
4.	Commands executed 
5.	Action command
6.	Execution time and detailed time
7.	How does the car action in Carla 
8.	Any failures or unexpected outcomes

