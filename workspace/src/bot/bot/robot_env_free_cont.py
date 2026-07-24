import os
import time
import math
import threading
import subprocess
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from ament_index_python.packages import get_package_share_directory


class TurtleBotEnv(gym.Env, Node):
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, "rl_environment_node")

        # Osservazioni: 48 raggi LiDAR normalizzati + distanza goal + angolo goal
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 48 + [0.0, -1.0], dtype=np.float32),
            high=np.array([1.0] * 48 + [1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Azioni continue (Box): [0] = lineare (freno/acceleratore), [1] = angolare (sterzo)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_sensor)
        self.sub_robot_pose = self.create_subscription(Pose, "/model/burger/pose", self.robot_pose_callback, 10)

        self.latest_scan = np.full(48, 3.5, dtype=np.float32)
        self._scan_received = threading.Event()
        self._scan_lock = threading.Lock()

        self._pose_received = threading.Event()
        self._pose_lock = threading.Lock()

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.goal_x = 1.5
        self.goal_y = 1.5

        self.step_count = 0
        self.grace_period = 5
        self.max_steps = 1000  # margine per dare tempo all'esplorazione continua

        # Logica di collisione filtrata (LiDAR)
        self.collision_distance = 0.15
        self.min_collision_rays = 2
        self.hard_collision_distance = 0.10

        self.goal_reached_distance = 0.25
        self.arena_limit = 2.5
        self.max_distance = math.sqrt(2) * (self.arena_limit * 2)

        self.min_goal_distance_from_robot = 1.0
        self.wall_goal_margin = 0.45

        # Lettura automatica ostacoli da SDF
        self.world_sdf_name = "free.sdf"
        self.goal_marker_half_size = 0.25
        self.goal_obstacle_margin = 0.1

        self.forbidden_goal_circles = []
        self.forbidden_goal_rectangles = []
        self._load_obstacles_from_sdf()

        self.abort_on_bad_coordinates = True
        self.max_reasonable_abs_pose = 3.0

        self.last_distance = 0.0
        self.episode_reward = 0.0
        self.last_episode_success = False
        self.first_reset = True

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self.get_logger().info("Attendo primo scan LiDAR...")
        self._scan_received.wait(timeout=10.0)

        self.get_logger().info("Attendo posa Gazebo del robot...")
        self._pose_received.wait(timeout=10.0)

        self.get_logger().info("Ambiente pronto a spazio continuo.")

    # Parsing SDF e geometria

    def _read_pose_xy_yaw(self, element):
        if element is None or element.text is None:
            return 0.0, 0.0, 0.0
        values = [float(v) for v in element.text.strip().split()]
        values += [0.0] * (6 - len(values))
        return values[0], values[1], values[5]

    def _compose_pose_2d(self, parent_pose, child_pose):
        px, py, pyaw = parent_pose
        cx, cy, cyaw = child_pose
        c, s = math.cos(pyaw), math.sin(pyaw)
        x = px + c * cx - s * cy
        y = py + s * cx + c * cy
        yaw = math.atan2(math.sin(pyaw + cyaw), math.cos(pyaw + cyaw))
        return x, y, yaw

    def _load_obstacles_from_sdf(self):
        package_share = get_package_share_directory("bot")
        world_path = os.path.join(package_share, "world", self.world_sdf_name)
        ignored_models = {"ground_plane", "goal_marker", "burger"}

        tree = ET.parse(world_path)
        root = tree.getroot()

        for model in root.findall(".//model"):
            model_name = model.attrib.get("name", "")
            if model_name in ignored_models: continue

            model_pose = self._read_pose_xy_yaw(model.find("pose"))
            for link in model.findall("link"):
                link_pose = self._read_pose_xy_yaw(link.find("pose"))
                link_world_pose = self._compose_pose_2d(model_pose, link_pose)

                for collision in link.findall("collision"):
                    collision_pose = self._read_pose_xy_yaw(collision.find("pose"))
                    world_pose = self._compose_pose_2d(link_world_pose, collision_pose)
                    geometry = collision.find("geometry")
                    if geometry is None: continue

                    cylinder = geometry.find("cylinder")
                    box = geometry.find("box")

                    if cylinder is not None:
                        radius = float(cylinder.find("radius").text)
                        forbidden_radius = radius + self.goal_marker_half_size + self.goal_obstacle_margin
                        self.forbidden_goal_circles.append((world_pose[0], world_pose[1], forbidden_radius))
                    elif box is not None:
                        sx, sy, _ = [float(v) for v in box.find("size").text.strip().split()]
                        half_x = sx / 2.0 + self.goal_marker_half_size + self.goal_obstacle_margin
                        half_y = sy / 2.0 + self.goal_marker_half_size + self.goal_obstacle_margin
                        self.forbidden_goal_rectangles.append((world_pose[0], world_pose[1], half_x, half_y, world_pose[2]))

    def _point_inside_rotated_rectangle(self, px, py, cx, cy, half_x, half_y, yaw):
        dx, dy = px - cx, py - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        return abs(local_x) <= half_x and abs(local_y) <= half_y

    def _is_goal_position_valid(self, gx, gy):
        if abs(gx) > self.arena_limit - self.wall_goal_margin: return False
        if abs(gy) > self.arena_limit - self.wall_goal_margin: return False

        for ox, oy, radius in self.forbidden_goal_circles:
            if math.sqrt((gx - ox) ** 2 + (gy - oy) ** 2) < radius: return False
        for cx, cy, half_x, half_y, yaw in self.forbidden_goal_rectangles:
            if self._point_inside_rotated_rectangle(gx, gy, cx, cy, half_x, half_y, yaw): return False
        return True

    # Callback e sensori

    def robot_pose_callback(self, msg):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        with self._pose_lock:
            self.robot_x = msg.position.x
            self.robot_y = msg.position.y
            self.robot_yaw = math.atan2(siny, cosy)
        self._pose_received.set()

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)

        # Filtro anti-fantasma
        ranges[ranges < 0.05] = 3.5

        ranges = np.clip(ranges, 0.0, 3.5)
        indices = np.linspace(0, len(ranges) - 1, 48, dtype=int)

        with self._scan_lock:
            self.latest_scan = ranges[indices].astype(np.float32)
        self._scan_received.set()

    def _get_pose(self):
        with self._pose_lock: return self.robot_x, self.robot_y, self.robot_yaw

    def _check_coordinates_or_abort(self, distance_to_goal):
        if not self.abort_on_bad_coordinates: return
        robot_x, robot_y, _ = self._get_pose()
        if abs(robot_x) > self.max_reasonable_abs_pose or abs(robot_y) > self.max_reasonable_abs_pose or distance_to_goal > self.max_distance + 1.0:
            msg = f"Coordinate sballate: robot=({robot_x:.3f},{robot_y:.3f}) goal=({self.goal_x:.3f},{self.goal_y:.3f}) dist={distance_to_goal:.3f}"
            self.get_logger().error(msg)
            raise RuntimeError(msg)

    def _move_goal_gazebo(self, goal_x, goal_y):
        result = subprocess.run(
            ["gz", "service", "-s", "/world/arena/set_pose", "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", "--timeout", "1000", "--req", f'name: "goal_marker" position: {{x: {goal_x}, y: {goal_y}, z: 0.005}}'],
            env=os.environ.copy(), capture_output=True, text=True
        )
        if result.returncode != 0: return False
        time.sleep(0.15)
        return True

    def _generate_random_goal(self):
        robot_x, robot_y, _ = self._get_pose()
        for _ in range(100):
            gx, gy = np.random.uniform(-self.arena_limit, self.arena_limit), np.random.uniform(-self.arena_limit, self.arena_limit)
            if math.sqrt((gx - robot_x) ** 2 + (gy - robot_y) ** 2) <= self.min_goal_distance_from_robot: continue
            if not self._is_goal_position_valid(gx, gy): continue
            return gx, gy
        raise RuntimeError("Impossibile generare goal valido.")

    def _set_random_goal(self):
        for _ in range(5):
            gx, gy = self._generate_random_goal()
            if self._move_goal_gazebo(gx, gy):
                self.goal_x, self.goal_y = gx, gy
                return
            time.sleep(0.2)
        raise RuntimeError("Impossibile spostare goal_marker.")

    def _get_goal_obs(self):
        robot_x, robot_y, robot_yaw = self._get_pose()
        dx, dy = self.goal_x - robot_x, self.goal_y - robot_y
        distance, angle_to_goal = math.sqrt(dx**2 + dy**2), math.atan2(dy, dx)
        relative_angle = math.atan2(math.sin(angle_to_goal - robot_yaw), math.cos(angle_to_goal - robot_yaw))
        return distance, relative_angle

    def _get_obs(self):
        with self._scan_lock: scan = self.latest_scan.copy()
        distance, angle = self._get_goal_obs()
        return np.append(scan / 3.5, [min(distance / self.max_distance, 1.0), angle / math.pi]).astype(np.float32)

    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        raw_weights = np.cos(relative_angles) ** 6 + 0.1
        scaled_weights = raw_weights * (max_weight / np.max(raw_weights))
        return scaled_weights / np.sum(scaled_weights)

    def compute_weighted_obstacle_reward(self):
        with self._scan_lock: scan = self.latest_scan.copy()
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        angles[angles > np.pi] -= 2 * np.pi

        valid_mask = scan <= 0.8

        if not np.any(valid_mask): return 0.0

        weights = self.compute_directional_weights(angles[valid_mask])
        safe_dists = np.clip(scan[valid_mask] - 0.25, 1e-2, 3.5)
        decay = np.exp(-3.0 * safe_dists)

        # Fattore di decadimento ostacoli, tarato per non penalizzare eccessivamente l'esplorazione
        return -(1.0 + 4.0 * np.dot(weights, decay))

    # Step continuo con freno integrato

    def step(self, action):
        twist = Twist()

        # Mappatura azione continua: lineare [0.0 -> 0.22] | angolare [-1.0 -> 1.0]
        raw_linear = float(action[0])
        twist.linear.x = float(np.clip((raw_linear + 1.0) / 2.0 * 0.22, 0.0, 0.22))
        twist.angular.z = float(np.clip(action[1], -1.0, 1.0))

        self.pub_cmd_vel.publish(twist)

        self._scan_received.clear()
        self._scan_received.wait(timeout=1.0)

        obs = self._get_obs()
        scan_m = obs[:48] * 3.5
        min_distance = float(np.min(scan_m))
        near_count = int(np.sum(scan_m < self.collision_distance))
        hard_collision = min_distance < self.hard_collision_distance

        distance_to_goal = float(obs[-2]) * self.max_distance
        angle_to_goal = float(obs[-1]) * math.pi

        self._check_coordinates_or_abort(distance_to_goal)
        self.step_count += 1

        terminated = False
        truncated = False
        info = {"is_success": False, "episode_success": 0, "collision": 0, "timeout": 0}

        # Reward shaping

        distance_reward = (self.last_distance - distance_to_goal) * 30.0
        self.last_distance = distance_to_goal

        # Disabilita il termine di allineamento all'obiettivo in prossimità di ostacoli o del goal,
        # per permettere manovre di aggiramento più fluide
        if min_distance < 0.80 or distance_to_goal < 0.60:
            yaw_reward = 0.0
        else:
            yaw_reward = 0.10 - (0.30 * abs(angle_to_goal) / math.pi)

        obstacle_reward = self.compute_weighted_obstacle_reward()

        # Premio per spazio libero frontale
        front_indices = [0, 1, 2, 46, 47]
        avg_front_dist = float(np.mean(scan_m[front_indices]))
        front_clearance_reward = 0.15 * (avg_front_dist / 3.5)

        reward = distance_reward + yaw_reward + obstacle_reward + front_clearance_reward

        # Penalità di collisione, ridotta per non scoraggiare l'esplorazione nelle fasi iniziali
        if (near_count >= self.min_collision_rays or hard_collision) and self.step_count > self.grace_period:
            reward = -50.0
            terminated = True
            self.last_episode_success = False
            info["collision"] = 1
            self.get_logger().info(f"Collisione | dist_goal={distance_to_goal:.3f} | min_scan={min_distance:.3f}")

        elif distance_to_goal <= self.goal_reached_distance:
            reward = 100.0
            terminated = True
            self.last_episode_success = True
            info["is_success"] = True
            info["episode_success"] = 1
            self.get_logger().info(f"GOAL RAGGIUNTO | dist={distance_to_goal:.3f}")

        elif self.step_count >= self.max_steps:
            reward = -50.0
            truncated = True
            self.last_episode_success = False
            info["timeout"] = 1

        self.episode_reward += reward

        if terminated or truncated:
            self.pub_cmd_vel.publish(Twist())
        return obs, reward, terminated, truncated, info

    def _respawn_robot_at_origin(self):
        env_vars = os.environ.copy()
        self.pub_cmd_vel.publish(Twist())

        subprocess.run(
            ["gz", "service", "-s", "/world/arena/remove", "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean", "--timeout", "1000", "--req", 'name: "burger", type: 2'],
            env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)

        package_share = get_package_share_directory("bot")
        model_path = os.path.join(package_share, "models", "turtlebot3_burger_pose", "model.sdf")

        req_spawn = f'sdf_filename: "{model_path}", name: "burger", pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.05 }} }}'
        self._pose_received.clear()

        subprocess.run(
            ["gz", "service", "-s", "/world/arena/create", "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean", "--timeout", "1000", "--req", req_spawn],
            env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)

        if not self._pose_received.wait(timeout=2.0):
            with self._pose_lock:
                self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
            self.get_logger().warn("Posa non ricevuta. Uso fallback (0,0,0).")

        robot_x, robot_y, robot_yaw = self._get_pose()
        self.get_logger().info(f"Robot dopo respawn | pose=({robot_x:.2f},{robot_y:.2f},{robot_yaw:.2f})")

    # Reset dell'episodio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_reward = 0.0
        self.pub_cmd_vel.publish(Twist())
        time.sleep(0.5)

        # Il respawn del robot avviene al primo avvio oppure dopo un fallimento;
        # in caso di successo il robot resta fermo e viene generato solo un nuovo target
        if self.first_reset:
            self.get_logger().info("Primo avvio: respawn robot e nuovo target.")
            self._respawn_robot_at_origin()
        elif self.last_episode_success:
            self.get_logger().info("Successo: il robot rimane fermo, genero nuovo target.")
        else:
            self.get_logger().info("Fallimento: respawn del robot.")
            self._respawn_robot_at_origin()

        self.first_reset = False

        # Viene sempre generato un nuovo goal per massimizzare l'esplorazione
        self._set_random_goal()

        with self._scan_lock:
            self.latest_scan = np.full(48, 3.5, dtype=np.float32)

        self._scan_received.clear()
        self._scan_received.wait(timeout=3.0)

        obs = self._get_obs()
        self.last_distance = float(obs[-2]) * self.max_distance
        return obs, {"is_success": False}

    def close(self):
        try: self.pub_cmd_vel.publish(Twist())
        except Exception: pass
        try:
            self._executor.shutdown()
            self._spin_thread.join(timeout=1.0)
        except Exception: pass
        try: self.destroy_node()
        except Exception: pass