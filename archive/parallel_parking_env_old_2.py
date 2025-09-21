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


# class SACOptimizedParkingEnv(AbstractEnv):
class ParallelParkingEnv(AbstractEnv):
    """
    A parallel parking environment optimized for SAC following highway-env standards.

    Reward function follows the goal environment pattern:
    R(s,a) = -||s - s_g||_{W,p}^p - b * collision
    
    Key features:
    - Simple reward function (goal distance + collision penalty)
    - Rewards bounded in [0, 1] range
    - Continuous action space for SAC
    - Minimal complexity following highway-env philosophy
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # Observation configuration
                "observation": {
                    "type": "KinematicsGoal",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "features_range": {
                        "x": [-50, 150],
                        "y": [-10, 10],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                    "absolute": False,
                    "normalize": True,
                    "see_behind": True,
                },
                
                # Action configuration - continuous for SAC
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                    "clip": True,
                },
                
                # Environment setup
                "lanes_count": 2,
                "vehicles_count": 4,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 50,
                "ego_spacing": 3,
                "vehicles_density": 0.2,
                
                "other_vehicles_type": None,
                
                # Parking space configuration
                "parking_space_length": 7.5,
                "parking_space_width": 3.0,
                "parked_vehicle_length": 4.5,
                
                # Reward parameters (following highway-env standards)
                "collision_penalty": 1.0,      # b coefficient in R = -||s-sg|| - b*collision
                "goal_weight": 1.0,            # Weight for goal distance term
                "p_norm": 2,                   # p-norm order (2 = Euclidean)
                "reward_weights": [1.0, 1.0, 0.3, 0.3, 0.5, 0.5],  # W weights for [x,y,vx,vy,cos,sin]
                
                # Success criteria
                "success_distance_threshold": 1.0,
                "success_angle_threshold": 0.2,
                "success_speed_threshold": 2.0,
                
                # Termination settings
                "collision_terminal": True,
                "success_terminal": True,
                "offroad_terminal": False,
                
                # Simulation settings
                "simulation_frequency": 20,
                "policy_frequency": 10,
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
        """Reset tracking variables."""
        ego_pos = self.vehicle.position
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        self.initial_distance = np.linalg.norm(ego_pos - target_pos)

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
        self.parking_spot_x = road_length * 0.6 + self.np_random.uniform(-3, 3)
        self.parking_spot_y = -4.0
        
        # Goal state: [x_g, y_g, 0, 0, cos(psi_g), sin(psi_g)]
        # Target heading is 0 (parallel to road)
        self.goal_state = np.array([
            self.parking_spot_x,  # x_g
            self.parking_spot_y,  # y_g
            0.0,                  # vx_g = 0 (stationary)
            0.0,                  # vy_g = 0 (stationary)
            1.0,                  # cos(0) = 1
            0.0                   # sin(0) = 0
        ])
        
        # Parking space boundaries for success check
        space_length = self.config["parking_space_length"]
        space_width = self.config["parking_space_width"]
        
        self.parking_space_bounds = {
            'x_min': self.parking_spot_x - space_length/2,
            'x_max': self.parking_spot_x + space_length/2,
            'y_min': self.parking_spot_y - space_width/2,
            'y_max': self.parking_spot_y + space_width/2,
        }

    def _create_vehicles(self) -> None:
        """Create vehicles."""
        # Create ego vehicle
        initial_offset = self.np_random.uniform(-10, -5)
        initial_x = self.parking_spot_x + initial_offset
        initial_y = self.np_random.uniform(-0.5, 0.5)
        ego_position = [initial_x, initial_y]
        
        ego_vehicle = Vehicle(
            road=self.road,
            position=ego_position,
            heading=self.np_random.uniform(-0.1, 0.1),
            speed=self.np_random.uniform(6.0, 10.0)
        )
        
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        # Create parked vehicles
        self._create_parked_vehicles()
        
        # Add minimal traffic
        # self._create_traffic_vehicles()

    def _create_parked_vehicles(self) -> None:
        """Create parked vehicles defining the parking space."""
        vehicle_length = self.config["parked_vehicle_length"]
        space_length = self.config["parking_space_length"]
        
        spacing = 0.4
        
        # Front parked vehicle
        front_pos = [
            self.parking_spot_x + space_length/2 + vehicle_length/2 + spacing,
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
            self.parking_spot_x - space_length/2 - vehicle_length/2 - spacing,
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

    def _create_traffic_vehicles(self) -> None:
        """Create minimal traffic vehicles."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Minimal traffic to avoid overcomplicating the task
        num_traffic = 1
        
        for i in range(num_traffic):
            vehicle = other_vehicles_type.create_random(
                self.road,
                spacing=1 / self.config["vehicles_density"],
                lane_id=("street", "driving", 0)
            )
            # Ensure traffic is far from parking area
            while abs(vehicle.position[0] - self.parking_spot_x) < 25:
                vehicle = other_vehicles_type.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                    lane_id=("street", "driving", 0)
                )
            
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        Reward function following highway-env goal environment standard:
        R(s,a) = -||s - s_g||_{W,p}^p - b * collision
        
        Bounded in [0, 1] range as per highway-env standards.
        """
        # Get current state: [x, y, vx, vy, cos(psi), sin(psi)]
        current_state = self._get_vehicle_state()
        
        # Compute weighted p-norm distance to goal
        weighted_diff = (current_state - self.goal_state) * self.config["reward_weights"]
        p_norm_distance = np.linalg.norm(weighted_diff, ord=self.config["p_norm"])
        
        # Normalize distance term to [0, 1] range
        # Use initial distance as normalization factor
        normalized_distance = min(p_norm_distance / (self.initial_distance + 1e-6), 1.0)
        
        # Collision penalty
        collision_penalty = self.config["collision_penalty"] if self.vehicle.crashed else 0.0
        
        # Combined reward: maximize closeness to goal, minimize collisions
        # Transform to [0, 1] range where 1 is best (at goal, no collision)
        reward = (1.0 - normalized_distance) * (1.0 - collision_penalty)
        
        # Ensure reward is in [0, 1] as per highway-env standards
        return float(np.clip(reward, 0.0, 1.0))

    def _get_vehicle_state(self) -> np.ndarray:
        """Get current vehicle state as [x, y, vx, vy, cos(psi), sin(psi)]."""
        pos = self.vehicle.position
        vel = self.vehicle.velocity
        heading = self.vehicle.heading
        
        return np.array([
            pos[0],           # x
            pos[1],           # y
            vel[0],           # vx
            vel[1],           # vy
            np.cos(heading),  # cos(psi)
            np.sin(heading)   # sin(psi)
        ])

    def _is_parked_successfully(self) -> bool:
        """Check if vehicle is successfully parked."""
        ego_pos = self.vehicle.position
        ego_heading = self.vehicle.heading
        ego_speed = np.linalg.norm([self.vehicle.velocity[0], self.vehicle.velocity[1]])
        
        # Distance to center of parking space
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        distance_to_center = np.linalg.norm(ego_pos - target_pos)
        
        # Check success criteria
        position_ok = distance_to_center <= self.config["success_distance_threshold"]
        
        heading_diff = abs(utils.wrap_to_pi(ego_heading))
        orientation_ok = heading_diff <= self.config["success_angle_threshold"]
        
        speed_ok = ego_speed <= self.config["success_speed_threshold"]
        
        # Must be within parking space bounds
        in_bounds = (
            self.parking_space_bounds['x_min'] <= ego_pos[0] <= self.parking_space_bounds['x_max'] and
            self.parking_space_bounds['y_min'] <= ego_pos[1] <= self.parking_space_bounds['y_max']
        )
        
        return position_ok and orientation_ok and speed_ok and in_bounds

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Success termination
        if self.config["success_terminal"] and self._is_parked_successfully():
            return True
        
        # Collision termination  
        if self.config["collision_terminal"] and self.vehicle.crashed:
            return True
            
        return False

    def _is_truncated(self) -> bool:
        """Episode truncation based on time limit."""
        return self.time >= self.config["duration"]

    def get_parking_info(self) -> dict:
        """Get parking-specific information for analysis."""
        ego_pos = np.array(self.vehicle.position)
        target_pos = np.array([self.parking_spot_x, self.parking_spot_y])
        distance_to_target = np.linalg.norm(ego_pos - target_pos)
        
        return {
            "distance_to_target": distance_to_target,
            "progress_ratio": max(0, (self.initial_distance - distance_to_target) / self.initial_distance),
            "in_parking_space": self._is_in_parking_bounds(),
            "successfully_parked": self._is_parked_successfully(),
            "goal_state": self.goal_state.tolist(),
            "current_state": self._get_vehicle_state().tolist(),
        }

    def _is_in_parking_bounds(self) -> bool:
        """Check if vehicle is within parking space bounds."""
        ego_pos = self.vehicle.position
        return (
            self.parking_space_bounds['x_min'] <= ego_pos[0] <= self.parking_space_bounds['x_max'] and
            self.parking_space_bounds['y_min'] <= ego_pos[1] <= self.parking_space_bounds['y_max']
        )


class ParallelParkingEnvCompact(ParallelParkingEnv):
    """
    Compact variant for faster training with even simpler setup.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "duration": 40,              # Shorter episodes
                "vehicles_count": 3,         # Minimal vehicles (ego + 2 parked)
                "vehicles_density": 0.1,     # Minimal traffic
                "simulation_frequency": 15,  # Lower frequency for speed
                "policy_frequency": 8,
                
                # Slightly more lenient for faster convergence
                "success_distance_threshold": 1.2,
                "success_angle_threshold": 0.25,
                "success_speed_threshold": 2.5,
                
                # Simpler reward weights (equal weighting)
                "reward_weights": [1.0, 1.0, 0.2, 0.2, 0.3, 0.3],
            }
        )
        return cfg