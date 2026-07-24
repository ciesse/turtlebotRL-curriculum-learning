docker exec -it robot_rl bash -c \
  "source /opt/ros/jazzy/setup.bash && \
   source /app/install/setup.bash 2>/dev/null || true && \
   exec bash"
