from __future__ import annotations
import numpy as np
import logging
from typing import Optional
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import LineType, StraightLane, CircularLane
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

class AdvancedParallelParkingEnv(ParkingEnv):
    """
    ADVANCED VERSION - Harder scenarios with tighter spaces, more obstacles, and different orientations
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
                "vehicles_count": 8,  # Increased for more obstacles
                "normalize": False,
                "absolute": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": (-5.0, 5.0),
                "steering_range": (-np.pi/3, np.pi/3),  # Wider steering range
            },
            "success_bonus": 200.0,  # Higher bonus for harder task
            "collision_penalty": -50.0,  # Higher penalty
            "steering_penalty_weight": 0.002,
            "success_position_threshold": 1.5,  # Tighter threshold
            "success_heading_threshold": np.deg2rad(20),  # Tighter heading
            "success_lateral_threshold": 0.5,  # Much tighter lateral alignment
            "parallel_spots_count": 8,  # More spots
            "spot_length": 6.5,  # Shorter spots (tighter parking)
            "spot_width": 2.5,  # Narrower spots
            "road_length": 80.0,  # Longer road
            "road_width": 4.5,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,  # Longer episodes for harder task
            "spawn_x_range": [5.0, 15.0],  # Wider spawn range
            "spawn_y_range": [-1.0, 1.0],
            "spawn_heading_range": [-0.2, 0.2],
            "spawn_speed": 0.0,
            "max_goal_index": 4,  # More goal options
            "max_goal_distance": 15.0,
            "min_goal_distance": 3.0,
            "enable_success_termination": True,
            "success_rate_threshold": 0.1,
            "obstacle_density": 0.8,  # High obstacle density
            "angled_parking": True,  # Enable angled parking spots
            "dynamic_obstacles": False,  # Start with static obstacles
            "curriculum_learning": True,  # Progressive difficulty
        })
        return config

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        self.cumulative_success_count = 0
        self.episode_count = 0
        self.last_lateral_error = 100.0
        self.last_longitudinal_error = 100.0
        self.last_speed = 0.0
        self.success = False
        self.steps = 0
        self.last_action = [0.0, 0.0]
        self.difficulty_level = 0  # Start easy
        super().__init__(config, render_mode)

    def define_spaces(self) -> None:
        if self.action_type is None:
            self.action_type = ContinuousAction(self, self.config["action"])
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])
        self.observation_type = self.observation_type_parking
        super().define_spaces()

    def _reset(self) -> None:
        self._update_difficulty()
        self._create_road()
        self.steps = 0
        self.success = False
        self._create_vehicles()

    def _update_difficulty(self) -> None:
        """Progressive curriculum learning - increase difficulty over time"""
        if not self.config["curriculum_learning"]:
            return
            
        success_rate = self.cumulative_success_count / max(self.episode_count, 1) if self.episode_count > 20 else 0.0
        
        # Level 0: Basic (first 100 episodes or until 30% success rate)
        if self.episode_count < 100 or success_rate < 0.3:
            self.difficulty_level = 0
            self.config["obstacle_density"] = 0.3
            self.config["spot_length"] = 7.5
            self.config["angled_parking"] = False
            
        # Level 1: More obstacles (until 50% success rate)
        elif success_rate < 0.5:
            self.difficulty_level = 1
            self.config["obstacle_density"] = 0.6
            self.config["spot_length"] = 7.0
            self.config["angled_parking"] = False
            
        # Level 2: Tighter spaces + obstacles (until 70% success rate)
        elif success_rate < 0.7:
            self.difficulty_level = 2
            self.config["obstacle_density"] = 0.7
            self.config["spot_length"] = 6.5
            self.config["angled_parking"] = True
            
        # Level 3: Maximum difficulty
        else:
            self.difficulty_level = 3
            self.config["obstacle_density"] = 0.8
            self.config["spot_length"] = 6.0
            self.config["angled_parking"] = True

    def _create_road(self) -> None:
        net = RoadNetwork()
        road_width = self.config["road_width"]
        spot_length = self.config["spot_length"]
        spot_width = self.config["spot_width"]
        road_length = self.config["road_length"]
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        
        # Main road
        net.add_lane("a", "b", StraightLane([0, 0], [road_length, 0], width=road_width, line_types=lt))
        
        y_offset = road_width / 2 + spot_width / 2
        
        for i in range(self.config["parallel_spots_count"]):
            x_start = i * (spot_length + 1) + 10  # Tighter spacing
            
            if self.config["angled_parking"] and i % 2 == 1:  # Every other spot is angled
                # Create angled parking spot (30 degrees)
                angle = np.deg2rad(30)
                x_end = x_start + spot_length * np.cos(angle)
                y_end = y_offset + spot_length * np.sin(angle)
                
                net.add_lane(
                    f"park_{i}", f"park_{i}_end",
                    StraightLane(
                        [x_start, y_offset],
                        [x_end, y_end],
                        width=spot_width,
                        line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
                    )
                )
            else:
                # Regular parallel parking spot
                net.add_lane(
                    f"park_{i}", f"park_{i}_end",
                    StraightLane(
                        [x_start, y_offset],
                        [x_start + spot_length, y_offset],
                        width=spot_width,
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

        # Curriculum-based spawn positioning
        if self.difficulty_level == 0:
            spawn_x_range = [8.0, 12.0]
            spawn_y_range = [0.5, 1.5]
            spawn_heading_range = [-0.05, 0.05]
        else:
            spawn_x_range = self.config["spawn_x_range"]
            spawn_y_range = self.config["spawn_y_range"]
            spawn_heading_range = self.config["spawn_heading_range"]

        spawn_x = self.np_random.uniform(*spawn_x_range)
        spawn_y = self.np_random.uniform(*spawn_y_range)
        spawn_heading = self.np_random.uniform(*spawn_heading_range)

        # Create ego vehicle
        vehicle = self.action_type.vehicle_class(
            self.road,
            [spawn_x, spawn_y],
            heading=spawn_heading,
            speed=self.config["spawn_speed"]
        )
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(vehicle)
        self.controlled_vehicles = [vehicle]

        # Create goal
        self._create_goal(vehicle)
        
        # Create obstacle vehicles with higher density
        self._create_obstacle_vehicles()

    def _create_goal(self, vehicle):
        """Create goal with support for angled parking"""
        spot_length = self.config["spot_length"]
        spot_width = self.config["spot_width"]
        road_width = self.config["road_width"]
        y_offset = road_width / 2 + spot_width / 2
        start_pos = np.array([vehicle.position[0], vehicle.position[1]], dtype=float)
        
        # Calculate all spot centers
        spot_centers = []
        spot_angles = []
        for i in range(self.config["parallel_spots_count"]):
            x_start = i * (spot_length + 1) + 10
            
            if self.config["angled_parking"] and i % 2 == 1:
                # Angled spot
                angle = np.deg2rad(30)
                center_x = x_start + (spot_length * np.cos(angle)) / 2
                center_y = y_offset + (spot_length * np.sin(angle)) / 2
                spot_angles.append(angle)
            else:
                # Regular spot
                center_x = x_start + spot_length / 2
                center_y = y_offset
                spot_angles.append(0.0)
                
            spot_centers.append((center_x, center_y))

        # Choose goal spot (avoid occupied spots)
        occupied_spots = self._get_occupied_spots()
        max_goal_index = min(self.config["max_goal_index"], len(spot_centers) - 1)
        candidate_indices = [i for i in range(max_goal_index + 1) if i not in occupied_spots]
        
        if not candidate_indices:
            candidate_indices = [0]

        # Filter by distance
        valid_indices = [
            i for i in candidate_indices
            if (spot_centers[i][0] > start_pos[0]
                and (spot_centers[i][0] - start_pos[0]) < self.config["max_goal_distance"]
                and (spot_centers[i][0] - start_pos[0]) > self.config["min_goal_distance"])
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
        goal_position = spot_centers[goal_index]
        goal_heading = spot_angles[goal_index]

        goal = Landmark(
            self.road,
            position=goal_position,
            heading=goal_heading
        )
        self.road.objects.append(goal)
        vehicle.goal = goal

    def _get_occupied_spots(self):
        """Get list of spots that will be occupied by obstacle vehicles"""
        occupied_spots = set()
        obstacle_density = self.config["obstacle_density"]
        total_spots = self.config["parallel_spots_count"]
        
        # Determine how many spots to occupy
        num_obstacles = int(total_spots * obstacle_density)
        
        # Randomly select spots to occupy (avoiding first few spots for goal placement)
        available_spots = list(range(self.config["max_goal_index"] + 1, total_spots))
        if len(available_spots) < num_obstacles:
            # If not enough spots after goal area, use some from goal area too
            available_spots.extend(range(self.config["max_goal_index"] + 1))
        
        if available_spots:
            occupied_indices = self.np_random.choice(
                available_spots, 
                size=min(num_obstacles, len(available_spots)), 
                replace=False
            )
            occupied_spots.update(occupied_indices)
        
        return occupied_spots

    def _create_obstacle_vehicles(self):
        """Create obstacle vehicles in parking spots and on road"""
        occupied_spots = self._get_occupied_spots()
        spot_length = self.config["spot_length"]
        
        # Place vehicles in parking spots
        for spot_i in occupied_spots:
            lane_id = (f"park_{spot_i}", f"park_{spot_i}_end", 0)
            lane = self.road.network.get_lane(lane_id)
            if lane:
                # Random position within the spot
                longitudinal_pos = self.np_random.uniform(0.2, 0.8) * spot_length
                parked_vehicle = Vehicle.make_on_lane(
                    self.road, 
                    lane_id, 
                    longitudinal=longitudinal_pos, 
                    speed=0
                )
                # Different colors for different vehicle types
                colors = [(100, 100, 100), (150, 150, 150), (80, 80, 80), (120, 120, 120)]
                parked_vehicle.color = colors[spot_i % len(colors)]
                self.road.vehicles.append(parked_vehicle)

        # Add some vehicles on the main road (moving obstacles for future dynamic version)
        if self.difficulty_level >= 2:
            main_lane_id = ("a", "b", 0)
            main_lane = self.road.network.get_lane(main_lane_id)
            if main_lane:
                for _ in range(2):  # 2 additional vehicles on main road
                    road_pos = self.np_random.uniform(20, 60)
                    road_vehicle = Vehicle.make_on_lane(
                        self.road,
                        main_lane_id,
                        longitudinal=road_pos,
                        speed=self.np_random.uniform(0, 2)  # Slow moving
                    )
                    road_vehicle.color = (200, 100, 100)  # Red for road vehicles
                    self.road.vehicles.append(road_vehicle)

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

        # 1. Distance reward (scaled by difficulty)
        distance_penalty_scale = 0.5 + (self.difficulty_level * 0.2)  # Harder at higher levels
        distance_reward = -distance * distance_penalty_scale
        reward += distance_reward

        # 2. Lateral alignment (more important for tighter spaces)
        lateral_penalty_scale = 3.0 + (self.difficulty_level * 1.0)
        lateral_penalty = -lateral_error * lateral_penalty_scale
        reward += lateral_penalty

        # 3. Heading alignment (more important for angled parking)
        heading_penalty_scale = 1.5 + (self.difficulty_level * 0.5)
        heading_reward = -heading_error * heading_penalty_scale
        reward += heading_reward

        # 4. Speed control (more strict for tighter spaces)
        speed = abs(vehicle.velocity[0])
        if distance < 5.0:
            max_allowed_speed = 0.5 - (self.difficulty_level * 0.1)
            speed_penalty = -speed * 3.0 if speed > max_allowed_speed else 0.0
            reward += speed_penalty

        # 5. Precision bonus (reward for being very close)
        if distance < 2.0:
            precision_bonus = (2.0 - distance) * 5.0
            reward += precision_bonus

        # 6. Collision penalties (higher for harder levels)
        if any(v.crashed for v in self.controlled_vehicles):
            collision_penalty = self.config["collision_penalty"] * (1 + self.difficulty_level * 0.5)
            reward += collision_penalty

        # 7. Success bonus (much higher for harder levels)
        is_actually_successful = self._is_success_check(vehicle, goal)
        if is_actually_successful:
            success_bonus = self.config["success_bonus"] * (1 + self.difficulty_level)
            reward += success_bonus

        # 8. Action penalties
        if len(self.last_action) > 1:
            steering_penalty = self.config["steering_penalty_weight"] * np.square(self.last_action[1])
            reward -= steering_penalty
        if len(self.last_action) > 0:
            accel_penalty = 0.001 * np.square(self.last_action[0])
            reward -= accel_penalty

        # 9. Living penalty (encourage efficiency)
        reward -= 0.05

        return reward

    def _is_success_check(self, vehicle, goal):
        """Check if parking is successful with difficulty-adjusted thresholds"""
        if goal is None:
            return False

        vehicle_pos = np.array(vehicle.position, dtype=float)
        goal_pos = np.array(goal.position, dtype=float)
        pos_error = np.linalg.norm(vehicle_pos[:2] - goal_pos[:2])
        heading_error = abs((vehicle.heading - goal.heading + np.pi) % (2 * np.pi) - np.pi)
        lateral_error = abs(vehicle_pos[1] - goal_pos[1])
        stopped = abs(vehicle.velocity[0]) < 0.15

        # Adjust thresholds based on difficulty
        pos_threshold = self.config["success_position_threshold"] - (self.difficulty_level * 0.2)
        heading_threshold = self.config["success_heading_threshold"] - (self.difficulty_level * np.deg2rad(5))
        lateral_threshold = self.config["success_lateral_threshold"] - (self.difficulty_level * 0.1)

        success = (pos_error < max(pos_threshold, 1.0)
                and heading_error < max(heading_threshold, np.deg2rad(10))
                and lateral_error < max(lateral_threshold, 0.3)
                and stopped)
        return success

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        vehicle = self.controlled_vehicles[0]
        goal = getattr(vehicle, "goal", None)
        return self._is_success_check(vehicle, goal)

    def _is_terminated(self) -> bool:
        # Terminate on collision
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        if crashed:
            return True

        # Terminate on success
        success = self._check_success()
        self.success = success
        if success:
            self.cumulative_success_count += 1
            return True

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
        return self.steps >= self.config["duration"]

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        success = self._check_success()
        info["is_success"] = success
        info["difficulty_level"] = self.difficulty_level
        info["episode_length"] = self.steps
        if hasattr(self, 'controlled_vehicles') and len(self.controlled_vehicles) > 0:
            v = self.controlled_vehicles[0]
            goal = getattr(v, "goal", None)
            if goal:
                dist = np.linalg.norm(np.array(v.position[:2]) - np.array(goal.position[:2]))
                info["distance_to_goal"] = dist
        return info

    def _reward(self, action: np.ndarray) -> float:
        obs_parking = self.observation_type_parking.observe()
        obs_list = obs_parking if isinstance(obs_parking, (tuple, list)) else [obs_parking]
        reward = sum(
            self.compute_reward(agent_obs["achieved_goal"], agent_obs["desired_goal"], {})
            for agent_obs in obs_list
        )
        self.last_action = action
        return reward

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
        self.last_distance = 100.0
        obs, info = super().reset(**kwargs)
        return obs, info

# Register the advanced environment
register(
    id="advanced-parallel-parking-v0",
    entry_point="advanced_parallel_parking_env:AdvancedParallelParkingEnv",
)