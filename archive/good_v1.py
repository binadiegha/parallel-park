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

    The agent must learn to parallel park into a designated parking spot along the side of a road.
    The car starts in the driving lane and must maneuver into the parallel parking space.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
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
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 3,   # Lower = slower simulation (try 1-3 for very slow)
                "policy_frequency": 1,       # Lower = agent acts less frequently (try 1 for slowest)
                "duration": 600,             # Increased duration to compensate
                "screen_width": 800,  # Wider screen for parallel parking view
                "screen_height": 400,
                "centering_position": [0.5, 0.5],
                "scaling": 12,  # Adjusted scaling for better view
                "controlled_vehicles": 1,
                "vehicles_count": 0,  # No parked vehicles for easier task
                "add_walls": True,
                "parking_spots": 5,  # Number of parallel parking spots
                "road_length": 100,  # Length of the main road
                "parking_spot_length": 8,  # Length of each parking spot
                "parking_spot_width": 3,   # Width of parking spots
                # Vehicle speed control parameters
                "initial_vehicle_speed": 0.0,  # Starting speed (m/s)
                "max_vehicle_speed": 0.5,      # Maximum speed limit (m/s)
                "min_vehicle_speed": -5.0,     # Maximum reverse speed (m/s)
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
        Create a road with parallel parking spots along the side.
        """
        net = RoadNetwork()
        
        # Main driving lane
        road_length = self.config["road_length"]
        lane_width = 4.0
        
        # Create main driving lane (horizontal)
        net.add_lane(
            "main", "main_end",
            StraightLane(
                [-road_length/2, 0], [road_length/2, 0], 
                width=lane_width, 
                line_types=(LineType.CONTINUOUS, LineType.STRIPED)
            )
        )
        
        # Create parallel parking spots along the bottom side of the road
        spots = self.config["parking_spots"]
        spot_length = self.config["parking_spot_length"]
        spot_width = self.config["parking_spot_width"]
        
        # Calculate spacing between spots
        total_spots_length = spots * spot_length
        spacing = (road_length - total_spots_length) / (spots + 1)
        
        for i in range(spots):
            # Position each parking spot
            x_start = -road_length/2 + spacing + i * (spot_length + spacing/spots)
            x_end = x_start + spot_length
            y_pos = -lane_width/2 - spot_width/2
            
            # Create parking spot lane
            net.add_lane(
                f"parking_{i}", f"parking_{i}_end",
                StraightLane(
                    [x_start, y_pos], [x_end, y_pos],
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
        """Create the controlled vehicle and set up the parking goal."""
        # Get all parking spot lanes
        all_lanes = list(self.road.network.lanes_dict().keys())
        parking_lanes = [lane_id for lane_id in all_lanes 
                        if lane_id[0].startswith("parking_")]
        
        # Controlled vehicle - start it in the main driving lane
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            # Start vehicle in driving lane, offset from center
            start_x = -self.config["road_length"]/4 + i * 10
            start_y = 0  # Center of driving lane
            
            vehicle = self.action_type.vehicle_class(
                self.road, [start_x, start_y], 0.0, self.config["initial_vehicle_speed"]
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            
            # Set vehicle speed limits
            vehicle.MAX_SPEED = self.config["max_vehicle_speed"]
            vehicle.MIN_SPEED = self.config["min_vehicle_speed"]
            
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Set up parking goal
        for vehicle in self.controlled_vehicles:
            # Randomly select a parking spot
            target_lane_id = parking_lanes[self.np_random.choice(len(parking_lanes))]
            target_lane = self.road.network.get_lane(target_lane_id)
            
            # Place goal in the middle of the parking spot
            goal_position = target_lane.position(target_lane.length / 2, 0)
            goal_heading = target_lane.heading
            
            vehicle.goal = Landmark(
                self.road, goal_position, heading=goal_heading
            )
            self.road.objects.append(vehicle.goal)

        # Add boundary walls if enabled
        if self.config["add_walls"]:
            road_length = self.config["road_length"]
            road_width = 15  # Total width including parking area
            
            # Top and bottom walls
            for y in [-road_width/2 - 2, road_width/2 + 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (road_length + 10, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            
            # Left and right walls
            for x in [-road_length/2 - 5, road_length/2 + 5]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (road_width + 4, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded with emphasis on position and orientation.

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

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
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParallelParkingEnvEasy(ParallelParkingEnv):
    """Easier version with more time and larger parking spots."""
    def __init__(self):
        super().__init__({
            "duration": 600,                # Increased from 300
            "parking_spot_length": 10,
            "parking_spot_width": 4,
            "success_goal_reward": 0.15,
            "simulation_frequency": 4,      # Even slower for easy mode
            "policy_frequency": 1
        })


class ParallelParkingEnvHard(ParallelParkingEnv):
    """Harder version with tighter spots and less time."""
    def __init__(self):
        super().__init__({
            "duration": 250,                # Adjusted for slower speed
            "parking_spot_length": 6,
            "parking_spot_width": 2.5,
            "success_goal_reward": 0.08,
            "simulation_frequency": 6,      # Slightly faster for hard mode
            "policy_frequency": 3
        })