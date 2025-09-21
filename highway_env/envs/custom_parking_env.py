from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class CustomParkingEnv(AbstractEnv):
    """
    A DQN-friendly parallel parking environment.

    The vehicle must parallel park between two stationary vehicles.
    Optimized for Deep Q-Network training with:
    - Simplified discrete action space
    - Dense reward shaping
    - Normalized observations
    - Clear success/failure states
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # Observation configuration - optimized for DQN
                "observation": {
                    "type": "KinematicsGoal",
                    "vehicles_count": 5,  # Ego + 2 parked + 2 traffic
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1,1],
                    "features_range": {
                        "x": [-50, 150],
                        "y": [-10, 10],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": False,
                    "normalize": True,
                    "see_behind": True,
                },
                
                # Action configuration - simplified for DQN
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],  # Reduced speed options
                },
                
                # Environment setup
                "lanes_count": 2,
                "vehicles_count": 4,  # Reduced for simpler state space
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 45,  # Reasonable time limit
                "ego_spacing": 3,
                "vehicles_density": 0.3,
                
                # Parking space configuration
                "parking_space_length": 7.0,
                "parking_space_width": 2.8,
                "parked_vehicle_length": 4.5,
                
                # DQN-optimized rewards - dense and shaped
                "collision_reward": -10,     # Strong negative for crashes
                "success_reward": 20,        # Strong positive for success
                "distance_reward_scale": 2.0,  # Scale for distance-based reward
                "angle_reward_scale": 1.5,     # Scale for orientation reward
                "progress_reward": 0.1,        # Small reward for making progress
                "idle_penalty": -0.05,         # Penalty for not moving
                "reverse_bonus": 0.02,         # Small bonus for using reverse (realistic)
                "final_position_bonus": 5.0,   # Bonus for ending in good position
                
                # Success criteria - slightly relaxed for easier learning
                "parking_tolerance": 1.2,
                "angle_tolerance": 0.3,
                "max_parking_speed": 3.0,
                
                # Training optimizations
                "normalize_reward": False,  # Keep raw rewards for DQN
                "reward_scale": 0.1,       # Scale down rewards if needed
                "offroad_terminal": True,   # Terminate if going off road
                "collision_terminal": True, # Terminate on collision
                "success_terminal": True,   # Terminate on success
                
                # Simulation settings
                "simulation_frequency": 15,  # Higher frequency for better control
                "policy_frequency": 5,       # Reasonable decision frequency
                "show_trajectories": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._define_parking_space()
        self._create_vehicles()
        self._reset_tracking_variables()

    def _reset_tracking_variables(self) -> None:
        """Reset variables used for reward calculation."""
        ego_pos = self.vehicle.position
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        self.initial_distance = np.linalg.norm(ego_pos - target_pos)
        self.best_distance = self.initial_distance
        self.last_distance = self.initial_distance
        self.steps_without_progress = 0
        self.has_attempted_reverse = False

    def _create_road(self) -> None:
        """Create a road with a driving lane and a parking lane."""
        net = RoadNetwork()
        
        lane_width = 4.0
        parking_width = 3.2
        road_length = 120.0
        
        # Driving lane
        driving_lane = StraightLane(
            start=[0, 0], 
            end=[road_length, 0],
            width=lane_width,
            line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]
        )
        
        # Parking lane
        parking_lane = StraightLane(
            start=[0, -lane_width], 
            end=[road_length, -lane_width],
            width=parking_width,
            line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE]
        )
        
        net.add_lane("street", "driving", driving_lane)
        net.add_lane("street", "parking", parking_lane)
        
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _define_parking_space(self) -> None:
        """Define the target parking space location."""
        road_length = 120.0
        self.parking_spot_x = road_length * 0.65  # Position along road
        self.parking_spot_y = -4.0  # In parking lane
        
        # Parking space boundaries
        space_length = self.config["parking_space_length"]
        space_width = self.config["parking_space_width"]
        
        self.parking_space_bounds = {
            'x_min': self.parking_spot_x - space_length/2,
            'x_max': self.parking_spot_x + space_length/2,
            'y_min': self.parking_spot_y - space_width/2,
            'y_max': self.parking_spot_y + space_width/2,
        }

    def _create_vehicles(self) -> None:
        """Create vehicles with deterministic positioning for consistent training."""
        # Create ego vehicle starting before the parking space
        initial_x = self.parking_spot_x - 15  # Start 15m before parking spot
        ego_position = [initial_x, 0]  # In driving lane
        
        ego_vehicle = Vehicle(
            road=self.road,
            position=ego_position,
            heading=0,
            speed=8.0
        )
        
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        # Create parked vehicles with consistent positioning
        self._create_parked_vehicles()
        
        # Add minimal traffic
        self._create_traffic_vehicles()

    def _create_parked_vehicles(self) -> None:
        """Create parked vehicles that define the parking space."""
        vehicle_length = self.config["parked_vehicle_length"]
        space_length = self.config["parking_space_length"]
        
        # Front parked vehicle
        front_pos = [
            self.parking_spot_x + space_length/2 + vehicle_length/2 + 0.3,
            self.parking_spot_y
        ]
        front_vehicle = Vehicle(
            road=self.road,
            position=front_pos,
            heading=0,
            speed=0
        )
        front_vehicle.color = (0.7, 0.7, 0.7)
        self.road.vehicles.append(front_vehicle)
        
        # Rear parked vehicle
        rear_pos = [
            self.parking_spot_x - space_length/2 - vehicle_length/2 - 0.3,
            self.parking_spot_y
        ]
        rear_vehicle = Vehicle(
            road=self.road,
            position=rear_pos,
            heading=0,
            speed=0
        )
        rear_vehicle.color = (0.7, 0.7, 0.7)
        self.road.vehicles.append(rear_vehicle)
        
        self.front_parked_vehicle = front_vehicle
        self.rear_parked_vehicle = rear_vehicle

    def _create_traffic_vehicles(self) -> None:
        """Create minimal traffic for realism without complicating learning."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Only create 1-2 traffic vehicles
        num_traffic = min(2, self.config["vehicles_count"] - 3)
        
        for i in range(num_traffic):
            # Place traffic vehicles away from parking area
            vehicle = other_vehicles_type.create_random(
                self.road,
                spacing=1 / self.config["vehicles_density"],
                lane_id=("street", "driving", 0)
            )
            # Ensure traffic doesn't spawn too close to parking area
            while abs(vehicle.position[0] - self.parking_spot_x) < 20:
                vehicle = other_vehicles_type.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                    lane_id=("street", "driving", 0)
                )
            
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        DQN-optimized reward function with dense rewards and clear signals.
        """
        rewards = self._rewards(action)
        
        # Weighted sum of rewards
        total_reward = (
            self.config["collision_reward"] * rewards["collision_reward"] +
            self.config["success_reward"] * rewards["success_reward"] +
            self.config["distance_reward_scale"] * rewards["distance_reward"] +
            self.config["angle_reward_scale"] * rewards["angle_reward"] +
            self.config["progress_reward"] * rewards["progress_reward"] +
            self.config["idle_penalty"] * rewards["idle_penalty"] +
            self.config["reverse_bonus"] * rewards["reverse_bonus"] +
            self.config["final_position_bonus"] * rewards["final_position_bonus"]
        )
        
        # Scale reward if configured
        if self.config["reward_scale"] != 1.0:
            total_reward *= self.config["reward_scale"]
            
        return total_reward

    def _rewards(self, action: Action) -> dict[str, float]:
        """Calculate dense reward components for effective DQN training."""
        ego_pos = np.array(self.vehicle.position)
        ego_heading = self.vehicle.heading
        ego_speed = self.vehicle.speed
        
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        distance_to_target = np.linalg.norm(ego_pos - target_pos)
        
        # Distance-based reward (dense)
        distance_reward = 0
        if distance_to_target < self.last_distance:
            distance_reward = (self.last_distance - distance_to_target) / self.initial_distance
            self.best_distance = min(self.best_distance, distance_to_target)
        
        # Progress tracking
        progress_reward = 0
        if distance_to_target < self.last_distance - 0.1:  # Meaningful progress
            progress_reward = 1.0
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        
        # Angle reward (dense)
        target_heading = 0.0  # Parallel to road
        heading_diff = abs(utils.wrap_to_pi(ego_heading - target_heading))
        angle_reward = max(0, (np.pi - heading_diff) / np.pi)
        
        # Encourage use of reverse when appropriate
        reverse_bonus = 0
        if hasattr(action, 'longitudinal') and action.longitudinal < 0:  # Reverse action
            if distance_to_target < 10.0:  # Only when close to parking
                reverse_bonus = 1.0
                self.has_attempted_reverse = True
        
        # Idle penalty
        idle_penalty = 0
        if abs(ego_speed) < 0.5:  # Nearly stationary
            idle_penalty = 1.0
        
        # Success and collision
        success_reward = 1.0 if self._is_parked_successfully() else 0.0
        collision_reward = 1.0 if self.vehicle.crashed else 0.0
        
        # Final position bonus (when episode ends successfully)
        final_position_bonus = 0
        if success_reward > 0:
            # Bonus based on how well positioned
            center_distance = abs(ego_pos[0] - self.parking_spot_x)
            if center_distance < 1.0:
                final_position_bonus = 1.0
        
        # Update tracking variables
        self.last_distance = distance_to_target
        
        return {
            "collision_reward": collision_reward,
            "success_reward": success_reward,
            "distance_reward": distance_reward,
            "angle_reward": angle_reward,
            "progress_reward": progress_reward,
            "idle_penalty": idle_penalty,
            "reverse_bonus": reverse_bonus,
            "final_position_bonus": final_position_bonus,
        }

    def _is_parked_successfully(self) -> bool:
        """Check if vehicle is successfully parked with relaxed criteria for DQN learning."""
        ego_pos = self.vehicle.position
        ego_heading = self.vehicle.heading
        ego_speed = abs(self.vehicle.speed)
        
        # Position check
        in_parking_space = (
            self.parking_space_bounds['x_min'] <= ego_pos[0] <= self.parking_space_bounds['x_max'] and
            self.parking_space_bounds['y_min'] <= ego_pos[1] <= self.parking_space_bounds['y_max']
        )
        
        # Orientation check (relaxed)
        heading_diff = abs(utils.wrap_to_pi(ego_heading))
        correct_orientation = heading_diff <= self.config["angle_tolerance"]
        
        # Speed check (relaxed)
        reasonable_speed = ego_speed <= self.config["max_parking_speed"]
        
        return in_parking_space and correct_orientation and reasonable_speed

    def _is_terminated(self) -> bool:
        """Clear termination conditions for DQN learning."""
        # Success termination
        if self.config["success_terminal"] and self._is_parked_successfully():
            return True
        
        # Collision termination
        if self.config["collision_terminal"] and self.vehicle.crashed:
            return True
            
        # Off-road termination
        if self.config["offroad_terminal"] and not self.vehicle.on_road:
            return True
            
        # Stuck detection (optional)
        if self.steps_without_progress > 100:  # No progress for too long
            return True
            
        return False

    def _is_truncated(self) -> bool:
        """Episode truncation based on time limit."""
        return self.time >= self.config["duration"]

    def get_normalized_state(self) -> dict:
        """Get normalized state information useful for DQN analysis."""
        ego_pos = np.array(self.vehicle.position)
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        distance_to_target = np.linalg.norm(ego_pos - target_pos)
        
        # Normalize values to [0, 1] or [-1, 1]
        normalized_distance = min(distance_to_target / 20.0, 1.0)  # Max 20m distance
        normalized_heading = self.vehicle.heading / np.pi  # [-1, 1]
        normalized_speed = min(abs(self.vehicle.speed) / 15.0, 1.0)  # Max 15 m/s
        
        # Position relative to parking space
        rel_x = (ego_pos[0] - self.parking_spot_x) / 10.0  # Normalize by 10m
        rel_y = (ego_pos[1] - self.parking_spot_y) / 5.0   # Normalize by 5m
        
        return {
            "distance_to_target": normalized_distance,
            "relative_x": np.clip(rel_x, -1, 1),
            "relative_y": np.clip(rel_y, -1, 1),
            "heading": normalized_heading,
            "speed": normalized_speed,
            "in_parking_space": float(self._is_in_parking_bounds()),
            "successfully_parked": float(self._is_parked_successfully()),
        }

    def _is_in_parking_bounds(self) -> bool:
        """Check if vehicle is within parking space bounds."""
        ego_pos = self.vehicle.position
        return (
            self.parking_space_bounds['x_min'] <= ego_pos[0] <= self.parking_space_bounds['x_max'] and
            self.parking_space_bounds['y_min'] <= ego_pos[1] <= self.parking_space_bounds['y_max']
        )


class CustomParkingEnvFast(CustomParkingEnv):
    """
    Fast variant optimized for DQN training with higher throughput.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 10,  # Lower for faster training
                "policy_frequency": 5,
                "vehicles_count": 3,         # Minimal vehicles
                "duration": 30,              # Shorter episodes
                "ego_spacing": 2.0,
                # More lenient success criteria for faster learning
                "parking_tolerance": 1.5,
                "angle_tolerance": 0.4,
                "max_parking_speed": 4.0,
            }
        )
        return cfg