#!/bin/bash
xhost +local:docker

docker run -it --rm \
  --name robot_rl \
  --env="DISPLAY=$DISPLAY" \
  --env="TURTLEBOT3_MODEL=burger" \
  --env="ROS_DOMAIN_ID=0" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $(pwd)/workspace:/app \
  -p 6006:6006 \
  --privileged \
  immagine_robot_rl \
  bash -c "
    source /opt/ros/jazzy/setup.bash &&
    source /app/install/setup.bash 2>/dev/null || true &&
    tensorboard --logdir=/app/ppo_tensorboard/ --host=0.0.0.0 --port=6006 &
    echo '✅ TensorBoard avviato su http://localhost:6006' &&
    exec bash
  "