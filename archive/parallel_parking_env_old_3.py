from __future__ import annotations

from abc import abstractmethod

import numpy as np
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParallelParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment for parallel parking.

    The agent must park parallel to the road between two parked vehicles.
    """

    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.15,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 100,
                "screen_width": 800,
                "screen_height": 400,
                "centering_position": [0.5, 0.5],
                "scaling": 8,
                "controlled_vehicles": 1,
                "parked_vehicles_count": 6,
                "parking_space_length": 7.0,  # Length of parking space
                "road_length": 100,
                "lane_width": 4.0,
            }
        )
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Create a straight road with parking spaces along the side.
        """
        net = RoadNetwork()
        lane_width = self.config["lane_width"]
        road_length = self.config["road_length"]
        
        # Main driving lane
        net.add_lane(
            "main", "main_end",
            StraightLane(
                [-road_length/2, lane_width], [road_length/2, lane_width], 
                width=lane_width,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED)
            )
        )
        
        # Parking lane (parallel to main lane)
        net.add_lane(
            "parking", "parking_end",
            StraightLane(
                [-road_length/2, 0], [road_length/2, 0], 
                width=lane_width,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS)
            )
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create vehicles and define parking scenario."""
        # Start ego vehicle in the main driving lane
        ego_x = -self.config["road_length"]/3
        ego_y = self.config["lane_width"]
        vehicle = self.action_type.vehicle_class(
            self.road, [ego_x, ego_y], 0.0, 5.0  # Start with some forward velocity
        )
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(vehicle)
        self.controlled_vehicles = [vehicle]

        # Create parked vehicles along the parking lane, leaving a gap for parking
        parking_space_length = self.config["parking_space_length"]
        vehicle_length = 5.0  # Standard vehicle length
        
        # Calculate positions for parked vehicles
        gap_center = 0  # Center of the parking space
        gap_start = gap_center - parking_space_length/2
        gap_end = gap_center + parking_space_length/2
        
        parked_positions = []
        
        # Vehicles before the gap
        x_pos = gap_start - vehicle_length/2 - 1.0  # 1m spacing
        while x_pos > -self.config["road_length"]/2 + vehicle_length/2:
            parked_positions.append(x_pos)
            x_pos -= (vehicle_length + 2.0)  # 2m between vehicles
            if len(parked_positions) >= self.config["parked_vehicles_count"]//2:
                break
                
        # Vehicles after the gap
        x_pos = gap_end + vehicle_length/2 + 1.0
        while x_pos < self.config["road_length"]/2 - vehicle_length/2:
            parked_positions.append(x_pos)
            x_pos += (vehicle_length + 2.0)
            if len(parked_positions) >= self.config["parked_vehicles_count"]:
                break

        # Create parked vehicles
        for x_pos in parked_positions:
            parked_vehicle = Vehicle(
                self.road, 
                [x_pos, 0.0],  # In parking lane
                heading=0.0, 
                speed=0.0
            )
            parked_vehicle.color = VehicleGraphics.BLUE
            self.road.vehicles.append(parked_vehicle)

        # Set goal position in the center of the parking gap
        goal_x = gap_center
        goal_y = 0.0  # In the parking lane
        goal_heading = 0.0  # Parallel to the road
        
        self.controlled_vehicles[0].goal = Landmark(
            self.road, [goal_x, goal_y], heading=goal_heading
        )
        self.road.objects.append(self.controlled_vehicles[0].goal)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Compute reward based on proximity to goal and proper parallel alignment.
        """
        # Basic distance reward
        distance_reward = -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )
        
        # Additional reward for proper orientation (parallel to road)
        vehicle = self.controlled_vehicles[0]
        heading_diff = abs(vehicle.heading % (2*np.pi))
        if heading_diff > np.pi:
            heading_diff = 2*np.pi - heading_diff
        
        # Reward for being parallel (heading close to 0 or π)
        parallel_reward = 0
        if heading_diff < np.pi/4 or heading_diff > 3*np.pi/4:
            parallel_reward = 0.1 * (1 - min(heading_diff, np.pi - heading_diff) / (np.pi/4))
            
        return distance_reward + parallel_reward

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        # Check if vehicle is close enough to goal
        distance_success = (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )
        
        # Check if vehicle is properly aligned (parallel)
        vehicle = self.controlled_vehicles[0]
        heading_diff = abs(vehicle.heading % (2*np.pi))
        if heading_diff > np.pi:
            heading_diff = 2*np.pi - heading_diff
        
        alignment_success = heading_diff < np.pi/6  # Within 30 degrees of parallel
        
        return distance_success and alignment_success

    def _is_terminated(self) -> bool:
        """Episode ends if crashed or successfully parked."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """Episode truncated if time limit exceeded."""
        return self.time >= self.config["duration"]


class ParallelParkingEnvActionRepeat(ParallelParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParallelParkingEnvCompact(ParallelParkingEnv):
    """Compact version with tighter parking space for more challenging scenarios."""
    def __init__(self):
        super().__init__({"parking_space_length": 6.0, "parked_vehicles_count": 8})