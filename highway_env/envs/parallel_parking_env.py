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
        self.parked_vehicles = []  # Track parked cars

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
                "reward_weights": [1, 0.3, 0.3, 0.3, 0.05, 0.05],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "step_penalty": -0.01,  # Small penalty per step to encourage efficiency
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 3,   # Lower = slower simulation (try 1-3 for very slow)
                "policy_frequency": 1,       # Lower = agent acts less frequently (try 1 for slowest)
                "duration": 600,             # Increased duration to compensate
                "screen_width": 800,  # Wider screen for parallel parking view
                "screen_height": 400,
                "centering_position": [0.5, 0.5],
                "scaling": 12,  # Adjusted scaling for better view
                "controlled_vehicles": 1,
                "vehicles_count": 0,  # This will be overridden by parked car config
                "add_walls": True,
                "parking_spots": 5,  # Number of parallel parking spots
                "road_length": 100,  # Length of the main road
                "parking_spot_length": 8,  # Length of each parking spot
                "parking_spot_width": 3,   # Width of parking spots
                # Vehicle speed control parameters
                "initial_vehicle_speed": 0.0,  # Starting speed (m/s)
                "max_vehicle_speed": 0.5,      # Maximum speed limit (m/s)
                "min_vehicle_speed": -5.0,     # Maximum reverse speed (m/s)
                # Orientation tolerance for success
                "orientation_tolerance": 0.2,  # ±0.2 radians tolerance for final orientation
                # Parked car configuration
                "max_parked_cars": 2,          # Maximum number of parked cars (0-2)
                "min_parked_cars": 0,          # Minimum number of parked cars
                "parked_car_probability": 0.8, # Probability of placing a parked car in an available spot
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
            x_start = -road_length/2 + spacing + i * (spot_length + spacing)
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

    def _get_parking_spot_info(self, spot_idx: int) -> dict:
        """Get the boundaries and center position of a parking spot."""
        road_length = self.config["road_length"]
        spots = self.config["parking_spots"]
        spot_length = self.config["parking_spot_length"]
        spot_width = self.config["parking_spot_width"]
        lane_width = 4.0
        
        total_spots_length = spots * spot_length
        spacing = (road_length - total_spots_length) / (spots + 1)
        
        x_start = -road_length/2 + spacing + spot_idx * (spot_length + spacing)
        x_end = x_start + spot_length
        x_center = (x_start + x_end) / 2
        y_center = -lane_width/2 - spot_width/2
        
        return {
            'x_start': x_start,
            'x_end': x_end,
            'x_center': x_center,
            'y_center': y_center,
            'width': spot_width,
            'length': spot_length
        }

    def _create_parked_vehicles(self, target_spot_idx: int) -> list:
        """Create parked vehicles in random spots (excluding the target spot)."""
        parked_cars = []
        spots = self.config["parking_spots"]
        max_parked = min(self.config["max_parked_cars"], spots - 1)  # Leave at least target spot free
        min_parked = min(self.config["min_parked_cars"], max_parked)
        
        # Determine number of parked cars
        num_parked = self.np_random.integers(min_parked, max_parked + 1)
        
        if num_parked == 0:
            return parked_cars
        
        # Get available spots (excluding target)
        available_spots = [i for i in range(spots) if i != target_spot_idx]
        
        # Randomly select spots for parked cars
        selected_spots = self.np_random.choice(
            available_spots, 
            size=min(num_parked, len(available_spots)), 
            replace=False
        )
        
        # Create parked vehicles
        for spot_idx in selected_spots:
            # Only place car with certain probability
            if self.np_random.random() < self.config["parked_car_probability"]:
                spot_info = self._get_parking_spot_info(spot_idx)
                
                # Create parked vehicle at the center of the spot
                parked_car = Vehicle(
                    road=self.road,
                    position=[spot_info['x_center'], spot_info['y_center']],
                    heading=0.0,  # Aligned with parking spot
                    speed=0.0     # Stationary
                )
                
                # Set visual properties for parked cars
                parked_car.color = VehicleGraphics.BLUE  # Different color from ego vehicle
                
                # Make sure the car stays stationary
                parked_car.MAX_SPEED = 0.0
                parked_car.MIN_SPEED = 0.0
                
                parked_cars.append(parked_car)
                self.road.vehicles.append(parked_car)
        
        return parked_cars

    def _create_vehicles(self) -> None:
        """Create the controlled vehicle, parked cars, and set up the parking goal."""
        # Get all parking spot lanes
        all_lanes = list(self.road.network.lanes_dict().keys())
        parking_lanes = [lane_id for lane_id in all_lanes 
                        if lane_id[0].startswith("parking_")]
        
        # Select target parking spot first
        target_spot_idx = self.np_random.choice(len(parking_lanes))
        
        # Create parked vehicles (excluding target spot)
        self.parked_vehicles = self._create_parked_vehicles(target_spot_idx)
        
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

        # Set up parking goal in the selected target spot
        for vehicle in self.controlled_vehicles:
            target_lane_id = parking_lanes[target_spot_idx]
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
                obstacle.LENGTH, obstacle.WIDTH = (road_length + 4, 1)
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
        Includes step penalty, parked car proximity penalty, reverse encouragement,
        and higher rewards for backing into the parking goal.

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        # Base proximity reward (without harsh penalties for overshooting)
        goal_distance = np.abs(achieved_goal - desired_goal)
        base_reward = -np.power(
            np.dot(goal_distance, np.array(self.config["reward_weights"])),
            p,
        )
        
        # Small step penalty to encourage efficiency
        step_penalty = -0.01
        
        # Add penalty for getting too close to parked cars
        proximity_penalty = 0.0
        if self.parked_vehicles and len(self.controlled_vehicles) > 0:
            ego_vehicle = self.controlled_vehicles[0]
            ego_pos = np.array([ego_vehicle.position[0], ego_vehicle.position[1]])
            
            for parked_car in self.parked_vehicles:
                parked_pos = np.array([parked_car.position[0], parked_car.position[1]])
                distance = np.linalg.norm(ego_pos - parked_pos)
                
                # Apply penalty if too close (within 3 meters)
                safe_distance = 3.0
                if distance < safe_distance:
                    proximity_penalty -= 0.1 * (safe_distance - distance) / safe_distance
        
        # Encourage reversing when vehicle has overshot the goal
        reverse_bonus = 0.0
        parking_approach_bonus = 0.0
        
        if len(self.controlled_vehicles) > 0:
            ego_vehicle = self.controlled_vehicles[0]
            current_x = achieved_goal[0]
            current_y = achieved_goal[1]
            goal_x = desired_goal[0]
            goal_y = desired_goal[1]
            vehicle_vx = achieved_goal[2]  # x-velocity
            vehicle_vy = achieved_goal[3]  # y-velocity
            
            # Check if vehicle has overshot the goal in x-direction
            overshot_distance = current_x - goal_x
            
            # If significantly overshot and moving backwards (negative velocity), give bonus
            if abs(overshot_distance) > 2.0:  # More than 2 meters past goal
                if (overshot_distance > 0 and vehicle_vx < -0.1) or (overshot_distance < 0 and vehicle_vx > 0.1):
                    # Moving in the right direction to correct overshoot
                    reverse_bonus = 0.05 * min(abs(overshot_distance) / 5.0, 1.0)
            
            # Higher rewards for backing into the parking spot
            # Check if vehicle is approaching the parking goal area
            distance_to_goal = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
            
            # Define parking approach zone (within 5 meters of goal)
            if distance_to_goal < 5.0:
                # Check if moving towards the parking spot (getting closer to goal in y-direction)
                # Parking spots are below the main lane, so negative y velocity means approaching
                if vehicle_vy < -0.05:  # Moving towards parking area (negative y)
                    approach_intensity = min(abs(vehicle_vy) / 1.0, 1.0)  # Scale by speed
                    proximity_factor = (5.0 - distance_to_goal) / 5.0  # Closer = higher reward
                    
                    # MUCH higher reward for backing into parking spot
                    if vehicle_vx < -0.05:  # Reversing (negative x velocity)
                        parking_approach_bonus = 0.15 * approach_intensity * proximity_factor
                    else:  # Forward approach gets lower reward
                        parking_approach_bonus = 0.05 * approach_intensity * proximity_factor
                
                # Small penalty for moving away from parking area when close
                elif vehicle_vy > 0.05 and distance_to_goal < 3.0:
                    parking_approach_bonus = -0.02 * (3.0 - distance_to_goal) / 3.0
        
        # Success bonus if properly parked
        success_bonus = 0.0
        if self._is_success(achieved_goal, desired_goal):
            success_bonus = self.config["success_goal_reward"]
        
        return base_reward + step_penalty + proximity_penalty + reverse_bonus + parking_approach_bonus + success_bonus

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
        """
        Check if the parking is successful based on position within parking spot boundaries,
        velocity, and orientation tolerance.
        
        :param achieved_goal: Current state [x, y, vx, vy, cos_h, sin_h]
        :param desired_goal: Target state [x, y, vx, vy, cos_h, sin_h]
        :return: True if the vehicle is within the parking spot and properly oriented
        """
        # Get vehicle position
        vehicle_x = achieved_goal[0]
        vehicle_y = achieved_goal[1]
        vehicle_vx = achieved_goal[2]
        vehicle_vy = achieved_goal[3]
        
        # Check velocity criteria (should be nearly stopped for successful parking)
        velocity_threshold = 0.15  # m/s
        velocity_success = (abs(vehicle_vx) < velocity_threshold and 
                          abs(vehicle_vy) < velocity_threshold)
        
        # Only consider it a success if velocity is low AND in correct position
        # This allows the vehicle to move through the parking spot without terminating
        if not velocity_success:
            return False
        
        # Find which parking spot the vehicle should be in based on desired goal
        goal_x = desired_goal[0]
        goal_y = desired_goal[1]
        
        # Find the correct parking spot based on goal position
        spots = self.config["parking_spots"]
        spot_length = self.config["parking_spot_length"]
        spot_width = self.config["parking_spot_width"]
        road_length = self.config["road_length"]
        lane_width = 4.0
        
        # Calculate parking spot boundaries
        total_spots_length = spots * spot_length
        spacing = (road_length - total_spots_length) / (spots + 1)
        
        # Find which parking spot contains the goal
        target_spot_idx = None
        for i in range(spots):
            x_start = -road_length/2 + spacing + i * (spot_length + spacing/spots)
            x_end = x_start + spot_length
            y_center = -lane_width/2 - spot_width/2
            
            # Check if goal is in this parking spot
            if x_start <= goal_x <= x_end and abs(goal_y - y_center) < spot_width/2:
                target_spot_idx = i
                break
        
        if target_spot_idx is None:
            return False
        
        # Now check if the vehicle is within the correct parking spot boundaries
        target_x_start = -road_length/2 + spacing + target_spot_idx * (spot_length + spacing/spots)
        target_x_end = target_x_start + spot_length
        target_y_center = -lane_width/2 - spot_width/2
        target_y_min = target_y_center - spot_width/2
        target_y_max = target_y_center + spot_width/2
        
        # Check if vehicle is within parking spot boundaries (with small tolerance for vehicle size)
        vehicle_tolerance = 0.7  # meters tolerance for vehicle boundaries
        position_success = (
            (target_x_start + vehicle_tolerance) <= vehicle_x <= (target_x_end - vehicle_tolerance) and
            (target_y_min + vehicle_tolerance) <= vehicle_y <= (target_y_max - vehicle_tolerance)
        )
        
        if not position_success:
            return False
        
        # Extract orientation information from achieved and desired goals
        # cos_h and sin_h are at indices 4 and 5
        achieved_cos_h = achieved_goal[4]
        achieved_sin_h = achieved_goal[5]
        desired_cos_h = desired_goal[4]
        desired_sin_h = desired_goal[5]
        
        # Convert cos/sin back to heading angles
        achieved_heading = np.arctan2(achieved_sin_h, achieved_cos_h)
        desired_heading = np.arctan2(desired_sin_h, desired_cos_h)
        
        # Calculate the angular difference, handling wrap-around
        angle_diff = achieved_heading - desired_heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-π, π]
        
        # Check if orientation is within tolerance
        orientation_success = abs(angle_diff) <= self.config["orientation_tolerance"]
        
        # SUCCESS requires: correct position + low velocity + correct orientation
        # This ensures the vehicle has actually "parked" rather than just passed through
        return position_success and velocity_success and orientation_success

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        # Only terminate on success (goal reached) or crash, allow exploration otherwise
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParallelParkingEnvEasy(ParallelParkingEnv):
    """Easier version with more time, larger parking spots, and fewer parked cars."""
    def __init__(self):
        super().__init__({
            "duration": 600,                # Increased from 300
            "parking_spot_length": 10,
            "parking_spot_width": 4,
            "success_goal_reward": 0.15,
            "simulation_frequency": 4,      # Even slower for easy mode
            "policy_frequency": 1,
            "orientation_tolerance": 0.2,   # Keep same orientation tolerance
            "max_parked_cars": 1,          # Fewer parked cars for easy mode
            "min_parked_cars": 0,
            "parked_car_probability": 0.5, # Lower probability of parked cars
        })


class ParallelParkingEnvHard(ParallelParkingEnv):
    """Harder version with tighter spots, less time, and more parked cars."""
    def __init__(self):
        super().__init__({
            "duration": 250,                # Adjusted for slower speed
            "parking_spot_length": 6,
            "parking_spot_width": 2.5,
            "success_goal_reward": 0.08,
            "simulation_frequency": 6,      # Slightly faster for hard mode
            "policy_frequency": 3,
            "orientation_tolerance": 0.2,   # Keep same orientation tolerance
            "max_parked_cars": 2,          # Maximum parked cars for hard mode
            "min_parked_cars": 1,          # At least 1 parked car
            "parked_car_probability": 0.9, # Higher probability of parked cars
        })