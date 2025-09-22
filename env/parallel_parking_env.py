from __future__ import annotations
import numpy as np
import logging
from typing import Optional
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import LineType, StraightLane
from gymnasium.envs.registration import register
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark
from highway_env.envs.parking_env import ParkingEnv
from highway_env.envs.common.action import ContinuousAction

# Enable debug logging for underlying systems
logging.basicConfig(level=logging.WARNING)
logging.getLogger("highway_env").setLevel(logging.INFO)
logging.getLogger("gymnasium").setLevel(logging.INFO)

class ParallelParkingEnv(ParkingEnv):
    """
    OPTIMIZED VERSION - Fixed episode length issue
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "see_vehicles": True,
                "vehicles_count": 4,
                "normalize": False,
                "absolute": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": (-5.0, 5.0),
                "steering_range": (-np.pi/4, np.pi/4),
            },
            "success_bonus": 100.0,
            "collision_penalty": -20.0,
            "steering_penalty_weight": 0.001,
            "success_position_threshold": 2.0,
            "success_heading_threshold": np.deg2rad(30),
            "success_lateral_threshold": 0.8,
            "parallel_spots_count": 5,
            "spot_length": 8.0,
            "spot_width": 3.0,
            "road_length": 60.0,
            "road_width": 4.0,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 200,
            "spawn_x_range": [5.0, 9.0],
            "spawn_y_range": [-0.5, 0.5],
            "spawn_heading_range": [-0.1, 0.1],
            "spawn_speed": 0.0,
            "max_goal_index": 2,
            "max_goal_distance": 10.0,
            "min_goal_distance": 1.0,
            "enable_success_termination": True,
            "success_rate_threshold": 0.05,
            "disable_obstacles_until_success_rate": 0.1,
        })
        return config

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        self.cumulative_success_count = 0
        self.episode_count = 0
        self.last_lateral_error = 100.0
        self.last_longitudinal_error = 100.0
        self.last_speed = 0.0
        self.success = False
        self.just_printed_goal = False
        self.steps = 0
        self.last_action = [0.0, 0.0]
        super().__init__(config, render_mode)

    def define_spaces(self) -> None:
        if self.action_type is None:
            self.action_type = ContinuousAction(self, self.config["action"])
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])
        self.observation_type = self.observation_type_parking
        super().define_spaces()

    def _reset(self) -> None:
        self._create_road()
        self.steps = 0
        self.success = False
        self.just_printed_goal = False
        self._create_vehicles()

    def _create_road(self) -> None:
        net = RoadNetwork()
        road_width = self.config["road_width"]
        spot_length = self.config["spot_length"]
        road_length = self.config["road_length"]
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        net.add_lane("a", "b", StraightLane([0, 0], [road_length, 0], width=road_width, line_types=lt))
        y_offset = road_width / 2 + self.config["spot_width"] / 2
        for i in range(self.config["parallel_spots_count"]):
            x_start = i * (spot_length + 2) + 5
            net.add_lane(
                f"park_{i}", f"park_{i}_end",
                StraightLane(
                    [x_start, y_offset],
                    [x_start + spot_length, y_offset],
                    width=self.config["spot_width"],
                    line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
                )
            )
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        if self.action_type is None:
            raise RuntimeError("action_type is None in _create_vehicles.")

        # Get max_goal_index from config
        max_goal_index = self.config["max_goal_index"]
        
        # More gradual curriculum - allow more variation earlier
        if self.episode_count < 50:
            spawn_x_range = [7.0, 8.5]
            spawn_y_range = [1.0, 2.5]
            spawn_heading_range = [-0.05, 0.05]
        else:
            spawn_x_range = self.config["spawn_x_range"]
            spawn_y_range = self.config["spawn_y_range"]
            spawn_heading_range = self.config["spawn_heading_range"]

        spawn_x = self.np_random.uniform(*spawn_x_range)
        spawn_y = self.np_random.uniform(*spawn_y_range)
        spawn_heading = self.np_random.uniform(*spawn_heading_range)

        vehicle = self.action_type.vehicle_class(
            self.road,
            [spawn_x, spawn_y],
            heading=spawn_heading,
            speed=self.config["spawn_speed"]
        )
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(vehicle)
        self.controlled_vehicles = [vehicle]

        spot_length = self.config["spot_length"]
        spot_width = self.config["spot_width"]
        road_width = self.config["road_width"]
        y_offset = road_width / 2 + spot_width / 2
        start_pos = np.array([spawn_x, spawn_y], dtype=float)
        max_dist = self.config["max_goal_distance"]
        min_dist = self.config["min_goal_distance"]

        spot_centers = []
        for i in range(self.config["parallel_spots_count"]):
            x_start = i * (spot_length + 2) + 5
            center_x = x_start + spot_length / 2
            center_y = y_offset
            spot_centers.append((center_x, center_y))

        occupied_spots = set()
        current_success_rate = self.cumulative_success_count / max(self.episode_count, 1) if self.episode_count > 20 else 0.0

        if current_success_rate >= self.config["disable_obstacles_until_success_rate"]:
            all_spot_indices = list(range(self.config["parallel_spots_count"]))
            self.np_random.shuffle(all_spot_indices)
            num_to_occupy = min(int(len(all_spot_indices) * 0.4), len(all_spot_indices) - 2)
            for i in all_spot_indices[:num_to_occupy]:
                if i > max_goal_index:
                    lane_id = (f"park_{i}", f"park_{i}_end", 0)
                    lane = self.road.network.get_lane(lane_id)
                    if lane:
                        parked_vehicle = Vehicle.make_on_lane(self.road, lane_id, longitudinal=spot_length/2, speed=0)
                        parked_vehicle.color = (100, 100, 100)
                        self.road.vehicles.append(parked_vehicle)
                        occupied_spots.add(i)

        candidate_indices = [i for i in range(max_goal_index + 1) if i not in occupied_spots]
        if not candidate_indices:
            candidate_indices = [0]

        valid_indices = [
            i for i in candidate_indices
            if (spot_centers[i][0] > start_pos[0]
                and (spot_centers[i][0] - start_pos[0]) < max_dist
                and (spot_centers[i][0] - start_pos[0]) > min_dist)
        ]

        if not valid_indices:
            ahead_indices = [i for i in candidate_indices if spot_centers[i][0] > start_pos[0]]
            if ahead_indices:
                distances = [spot_centers[i][0] - start_pos[0] for i in ahead_indices]
                closest_i = ahead_indices[np.argmin(distances)]
                valid_indices = [closest_i]
            else:
                valid_indices = [candidate_indices[0]]

        goal_index = self.np_random.choice(valid_indices)
        goal_lane_id = (f"park_{goal_index}", f"park_{goal_index}_end", 0)
        goal_lane = self.road.network.get_lane(goal_lane_id)

        if goal_lane is None:
            goal_position = [spot_centers[goal_index][0], spot_centers[goal_index][1]]
            goal_heading = 0.0
        else:
            goal_position = goal_lane.position(goal_lane.length * 0.75, 0)
            goal_heading = goal_lane.heading

        goal = Landmark(
            self.road,
            position=goal_position,
            heading=goal_heading
        )
        self.road.objects.append(goal)
        vehicle.goal = goal

    def _reward(self, action: np.ndarray) -> float:
        obs_parking = self.observation_type_parking.observe()
        obs_list = obs_parking if isinstance(obs_parking, (tuple, list)) else [obs_parking]
        reward = sum(
            self.compute_reward(agent_obs["achieved_goal"], agent_obs["desired_goal"], {})
            for agent_obs in obs_list
        )
        self.last_action = action
        return reward

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        vehicle = self.controlled_vehicles[0]
        goal = getattr(vehicle, "goal", None)

        if goal is None:
            pos_diff = achieved_goal[:2] - desired_goal[:2]
            distance = np.linalg.norm(pos_diff)
            lateral_error = abs(pos_diff[1])
            heading_error = 0.0
        else:
            vehicle_pos = np.array(vehicle.position, dtype=float)
            goal_pos = np.array(goal.position, dtype=float)
            pos_diff = goal_pos - vehicle_pos
            distance = np.linalg.norm(pos_diff)
            lateral_error = abs(pos_diff[1])
            heading_error = abs((vehicle.heading - goal.heading + np.pi) % (2 * np.pi) - np.pi)

        reward = 0.0

        # 1. Distance reward (more generous)
        distance_reward = -distance * 0.3  # Reduced penalty
        reward += distance_reward

        # 2. Lateral error penalty
        lateral_penalty = -lateral_error * 1.5  # Reduced penalty
        reward += lateral_penalty

        # 3. Heading alignment reward
        heading_reward = -heading_error * 0.8  # Reduced penalty
        reward += heading_reward

        # 4. Speed control (more lenient)
        speed = abs(vehicle.velocity[0])
        if distance < 3.0:
            speed_penalty = -speed * 1.0 if speed > 0.5 else 0.0  # More lenient
            reward += speed_penalty

        # 5. Reversing bonus when close to goal
        if distance < 4.0 and vehicle.velocity[0] < -0.1:
            reverse_bonus = 1.0
            reward += reverse_bonus

        # 6. Progress bonus - reward getting closer
        if hasattr(self, 'last_distance'):
            progress = self.last_distance - distance
            if progress > 0:
                reward += progress * 2.0  # Bonus for making progress
        self.last_distance = distance

        # 7. SUCCESS BONUS
        is_actually_successful = (distance < self.config["success_position_threshold"]
            and heading_error < self.config["success_heading_threshold"]
            and lateral_error < self.config["success_lateral_threshold"]
            and abs(vehicle.velocity[0]) < 0.2)  # More lenient speed requirement
        if is_actually_successful:
            success_bonus = self.config["success_bonus"]
            reward += success_bonus

        # 8. Penalties
        if any(v.crashed for v in self.controlled_vehicles):
            collision_penalty = self.config["collision_penalty"]
            reward += collision_penalty

        # 9. Action penalties (very small)
        if len(self.last_action) > 1:
            steering_penalty = self.config["steering_penalty_weight"] * np.square(self.last_action[1])
            reward -= steering_penalty
        if len(self.last_action) > 0:
            accel_penalty = 0.0005 * np.square(self.last_action[0])  # Even smaller penalty
            reward -= accel_penalty

        # 10. Living reward - small positive reward for each step (encourages exploration)
        reward += 0.1

        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        vehicle = self.controlled_vehicles[0]
        goal = getattr(vehicle, "goal", None)
        if goal is None:
            return False

        vehicle_pos = np.array(vehicle.position, dtype=float)
        goal_pos = np.array(goal.position, dtype=float)
        pos_error = np.linalg.norm(vehicle_pos[:2] - goal_pos[:2])
        heading_error = abs((vehicle.heading - goal.heading + np.pi) % (2 * np.pi) - np.pi)
        lateral_error = abs(vehicle_pos[1] - goal_pos[1])
        stopped = abs(vehicle.velocity[0]) < 0.2  # More lenient

        success = (pos_error < self.config["success_position_threshold"]
                and heading_error < self.config["success_heading_threshold"]
                and lateral_error < self.config["success_lateral_threshold"]
                and stopped)
        return success

    def _is_terminated(self) -> bool:
        # Only terminate on collision
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        if crashed:
            return True

        # Early termination on success (but only after some learning)
        if self.episode_count > 20:  # Allow some exploration first
            success = self._check_success()
            self.success = success
            if success:
                self.cumulative_success_count += 1
                return True  # Terminate on success

        self.success = False
        return False

    def _check_success(self) -> bool:
        obs_parking = self.observation_type_parking.observe()
        obs_list = obs_parking if isinstance(obs_parking, (tuple, list)) else [obs_parking]
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs_list
        )
        return success

    def _is_truncated(self) -> bool:
        # This is the key fix - make sure we respect our duration setting
        return self.steps >= self.config["duration"]

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        success = self._check_success()
        info["is_success"] = success
        info["episode_length"] = self.steps  # Add episode length to info
        if hasattr(self, 'controlled_vehicles') and len(self.controlled_vehicles) > 0:
            v = self.controlled_vehicles[0]
            goal = getattr(v, "goal", None)
            if goal:
                dist = np.linalg.norm(np.array(v.position[:2]) - np.array(goal.position[:2]))
                info["distance_to_goal"] = dist
        return info

    def _observation(self) -> dict:
        obs = super()._observation()
        if isinstance(obs, dict) and "observation" in obs:
            vehicle = self.controlled_vehicles[0]
            goal = getattr(vehicle, "goal", None)
            if goal is not None:
                # Transform goal to vehicle's coordinate system
                vehicle_pos = np.array(vehicle.position, dtype=float)
                goal_pos = np.array(goal.position, dtype=float)
                
                # Calculate relative position
                delta = goal_pos - vehicle_pos
                cos_h = np.cos(-vehicle.heading)
                sin_h = np.sin(-vehicle.heading)
                rel_x = cos_h * delta[0] - sin_h * delta[1]
                rel_y = sin_h * delta[0] + cos_h * delta[1]
                rel_heading = goal.heading - vehicle.heading
                
                # Create desired goal
                desired_goal = np.array([
                    rel_x,
                    rel_y,
                    0.0,
                    0.0,
                    np.cos(rel_heading),
                    np.sin(rel_heading)
                ], dtype=np.float32)
                
                # Update observation
                obs["desired_goal"] = desired_goal
                obs["achieved_goal"] = obs["observation"][:6]
        return obs

    def step(self, action):
        self.steps += 1
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_count += 1
        self.last_distance = 100.0  # Reset distance tracking
        obs, info = super().reset(**kwargs)
        return obs, info

# Register without max_episode_steps to avoid TimeLimit wrapper
register(
    id="parallel-parking-v0",
    entry_point="parallel_parking_env:ParallelParkingEnv",
)