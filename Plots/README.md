# TurtleBot3 RL Navigation - Testing & Training

Follow these steps to run the simulation and the RL agent using the provided Docker environment.

# Prerequisites
Clone this repository and open a terminal inside the downloaded project folder.

#Available Curriculum Stages
When running the commands below, replace `<model_name>` with one of the following available stages:
free;
medium;
forest;
final_easy.


# 1. Build the Docker Image (First Time Only)
Before running the container, you must build the Docker image which includes ROS2, Gazebo, and all required Python libraries.

cd docker_ws
./build_robot.sh
cd ..


# 2. Terminal 1 — Launch Gazebo
Start the Docker container, build the workspace, and launch the simulation environment:
./run_container.sh

 Inside the container:
cd /app
colcon build --symlink-install
source install/setup.bash
ros2 launch bot arena.launch_<model_name>.py

# 3. Terminal 2 — Training & Testing
Open a second terminal in the same project folder, execute into the running container, and run either the training or the testing script:
./exec_container.sh

Inside the container (to train a new model):
python3 src/bot/bot/train_<model_name>.py

Inside the container (to test the trained model):
python3 src/bot/bot/test_<model_name>.py
