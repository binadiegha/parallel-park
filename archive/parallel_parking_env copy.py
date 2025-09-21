from __future__ import annotations
from abc import abstractmethod

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle

from gymnasium import Env

class GoalEnv(Env):
    
    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """ This section holds the achieved goal and the desired goal in order to do a comparison as to how close we are to the goal."""
        
        raise NotImplementedError

class ParallelParkingEnv(AbstractEnv, GoalEnv):
    """
    DQN-friendly Parallel Parking environment.

    - One straight traffic lane along +x with a curb on the right (negative y).
    - A rectangular parallel-parking slot cut along the curb.
    - Two parked cars define the slot; agent must reverse in, align with lane, and stop.
    - Flat Kinematics observation + DiscreteMetaAction (for SB3 DQN).

    Success condition (checked in _is_success_pose):
      * Ego inside slot rectangle,
      * heading aligned to lane within tolerance,
      * nearly stopped (speed <= threshold).
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # === OBS / ACTION (DQN requires Discrete actions and flat obs) ===
                "observation": {
                    "type": "Kinematics",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-50, 150],
                        "y": [-10, 10],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "absolute": False,
                    "normalize": True,
                    "see_behind": True,
                },
                "action": {"type": "DiscreteMetaAction"},
                
                # === REWARD SHAPING ===
                "collision_reward": -15.0,
                "success_reward": 50.0,        # Big reward for success
                "step_penalty": -0.005,        # Small step penalty
                "pos_scale": 3.0,              # RBF width (meters) for distance shaping
                "head_weight": 1.0,            # heading alignment weight
                "pos_weight": 2.0,             # position shaping weight
                
                # === SUCCESS TOLERANCES ===
                "success_heading_deg": 15.0,   # ±15° tolerance (was too strict at 10°)
                "success_speed": 0.3,          # m/s (slightly more lenient)
                "success_position_tolerance": 1.0,  # meters from slot center
                
                # === SIMULATION ===
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 700,               # More time to complete task
                
                # === RENDERING ===
                "screen_width": 900,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],
                "scaling": 10,
                
                # === WORLD / VEHICLES ===
                "controlled_vehicles": 1,
                "vehicles_count": 2,           # parked cars outside the slot
                "add_walls": True,
                
                # --- Parallel-slot geometry & spawn ---
                "lane_length": 120.0,
                "lane_y": 0.0,                 # Center lane at y=0
                "lane_width": 8.0,             # Standard lane width
                "slot_center_x": 30.0,         # Slot position along x
                "slot_length": 12.0,           # Reasonable slot length
                "slot_depth": 2.5,             # Depth from curb
                "start_x": 10.0,               # Start position before slot
                "start_y": 0.0,                # Start in lane
                "start_heading": 0.0,          # Facing forward
                "start_speed": 0.0,
                "normalize_reward": True,
        
            }
        )
        return config

    def define_spaces(self) -> None:
        """Let AbstractEnv set spaces from config."""
        super().define_spaces()

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        
        # Reset success tracking
        # self._success_achieved = False

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if self.controlled_vehicles:
            v = self.controlled_vehicles[0]
            info["is_success"] = self._is_success_pose(v)
            
            # Add debugging info
            slot = self._slot
            slot_cx = 0.5 * (slot["xmin"] + slot["xmax"])
            slot_cy = 0.5 * (slot["ymin"] + slot["ymax"])
            dist_to_slot = np.hypot(v.position[0] - slot_cx, v.position[1] - slot_cy)
            
            info.update({
                "distance_to_slot": dist_to_slot,
                "vehicle_speed": abs(getattr(v, "speed", 0.0)),
                "vehicle_heading": v.heading,
                "in_slot_bounds": self._is_in_slot_bounds(v),
                "heading_aligned": self._is_heading_aligned(v),
                "vehicle_stopped": abs(getattr(v, "speed", 0.0)) <= self.config["success_speed"],
            })
        return info

    def _create_road(self) -> None:
        """Create road with lane and parking slot."""
        cfg = self.config
        net = RoadNetwork()
        
        # Main driving lane
        lane = StraightLane(
            start=[-cfg["lane_length"] / 2, cfg["lane_y"]],
            end=[cfg["lane_length"] / 2, cfg["lane_y"]],
            width=cfg["lane_width"],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        )
        net.add_lane("main", "main", lane)

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Define parking slot geometry
        curb_y = cfg["lane_y"] - cfg["lane_width"] / 2.0
        self._slot = {
            "xmin": cfg["slot_center_x"] - cfg["slot_length"] / 2.0,
            "xmax": cfg["slot_center_x"] + cfg["slot_length"] / 2.0,
            "ymin": curb_y - cfg["slot_depth"],
            "ymax": curb_y,
            "center_x": cfg["slot_center_x"],
            "center_y": curb_y - cfg["slot_depth"] / 2.0,
        }

        # Add boundary walls
        if cfg["add_walls"]:
            self._add_boundary_walls()

    def _add_boundary_walls(self):
        """Add walls to keep the agent in bounds."""
        width, height = 140.0, 40.0
        
        # Horizontal walls (top and bottom)
        for y in [-height / 2, height / 2]:
            obstacle = Obstacle(self.road, [0.0, y])
            obstacle.LENGTH, obstacle.WIDTH = (width, 2.0)
            obstacle.diagonal = np.hypot(obstacle.LENGTH, obstacle.WIDTH)
            self.road.objects.append(obstacle)
        
        # Vertical walls (left and right)
        for x in [-width / 2, width / 2]:
            obstacle = Obstacle(self.road, [x, 0.0], heading=np.pi / 2)
            obstacle.LENGTH, obstacle.WIDTH = (height, 2.0)
            obstacle.diagonal = np.hypot(obstacle.LENGTH, obstacle.WIDTH)
            self.road.objects.append(obstacle)

    def _create_vehicles(self) -> None:
        """Create ego vehicle, goal landmark, and parked cars."""
        cfg = self.config
        slot = self._slot

        # Create ego vehicle
        self.controlled_vehicles = []
        ego = self.action_type.vehicle_class(
            self.road,
            [cfg["start_x"], cfg["start_y"]],
            cfg["start_heading"],
            cfg["start_speed"],
        )
        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # Goal landmark at slot center
        ego.goal = Landmark(
            self.road, 
            [slot["center_x"], slot["center_y"]], 
            heading=0.0  # Parallel to lane
        )
        self.road.objects.append(ego.goal)

        # Create parked cars that define the slot
        self._create_parked_cars()

    def _create_parked_cars(self):
        """Create parked cars around the slot."""
        slot = self._slot
        cfg = self.config
        
        # Cars that define the slot boundaries
        car_length = 5.0
        car_width = 2.0
        
        # Front car (after the slot)
        front_car_x = slot["xmax"] + car_length / 2 + 0.5  # Small gap
        front_car = Vehicle(
            road=self.road,
            position=[front_car_x, slot["center_y"]],
            heading=0.0,
            speed=0.0,
        )
        front_car.LENGTH = car_length
        front_car.WIDTH = car_width
        front_car.color = (0, 0, 255)  # Blue
        self.road.vehicles.append(front_car)

        # Rear car (before the slot)
        rear_car_x = slot["xmin"] - car_length / 2 - 0.5   # Small gap
        rear_car = Vehicle(
            road=self.road,
            position=[rear_car_x, slot["center_y"]],
            heading=0.0,
            speed=0.0,
        )
        rear_car.LENGTH = car_length
        rear_car.WIDTH = car_width
        rear_car.color = (0, 0, 255)  # Blue
        self.road.vehicles.append(rear_car)

        # Additional parked cars along the street (not blocking the slot)
        for i in range(cfg["vehicles_count"]):
            # Random position away from the slot area
            attempts = 0
            while attempts < 10:
                x = self.np_random.uniform(-cfg["lane_length"]/3, cfg["lane_length"]/3)
                
                # Ensure not too close to slot or other critical areas
                if (slot["xmin"] - 15.0 <= x <= slot["xmax"] + 15.0 or
                    abs(x - cfg["start_x"]) < 8.0):
                    attempts += 1
                    continue
                
                car = Vehicle(
                    road=self.road,
                    position=[x, slot["center_y"]],
                    heading=0.0,
                    speed=0.0,
                )
                car.LENGTH = car_length
                car.WIDTH = car_width
                car.color = (128, 128, 128)  # Gray
                self.road.vehicles.append(car)
                break
                
            attempts += 1

    def _reward(self, action) -> float:
        """Compute reward with distance and heading shaping."""
        if not self.controlled_vehicles:
            return 0.0
            
        cfg = self.config
        v = self.controlled_vehicles[0]
        slot = self._slot #end goal

        # Base reward components
        reward = 0.0
        
        # Distance to slot center (Gaussian reward)
        dist = np.hypot(v.position[0] - slot["center_x"], v.position[1] - slot["center_y"])
        distance_reward = cfg["pos_weight"] * np.exp(-0.5 * (dist / cfg["pos_scale"]) ** 2)
        reward += distance_reward

        # Heading alignment (prefer heading = 0 for parallel parking)
        heading_error = abs(np.sin(v.heading))  # sin(0) = 0, sin(π/2) = 1
        heading_reward = cfg["head_weight"] * (1.0 - heading_error)
        reward += heading_reward



        # Step penalty (encourage efficiency)
        reward += cfg["step_penalty"]

        # Collision penalty
        if v.crashed:
            reward += cfg["collision_reward"]

        # Success bonus
        if self._is_success_pose(v):
            reward += cfg["success_reward"]
            if not self._success_achieved:
                print(f"SUCCESS ACHIEVED! Position: {v.position}, Heading: {np.degrees(v.heading):.1f}°")
                self._success_achieved = True

        return reward

    def _is_in_slot_bounds(self, v: Vehicle) -> bool:
        """Check if vehicle is within slot boundaries."""
        slot = self._slot
        x, y = v.position
        return (slot["xmin"] <= x <= slot["xmax"]) and (slot["ymin"] <= y <= slot["ymax"])

    def _is_heading_aligned(self, v: Vehicle) -> bool:
        """Check if vehicle heading is aligned with the lane."""
        cfg = self.config
        # Normalize heading to [0, 2π]
        heading = (v.heading + 2 * np.pi) % (2 * np.pi)
        
        # Check alignment with lane direction (0 or π radians)
        error1 = abs(heading)                    # aligned with +x
        error2 = abs(heading - np.pi)            # aligned with -x (reverse)
        error3 = abs(heading - 2 * np.pi)       # aligned with +x (wrapped)
        
        min_error = min(error1, error2, error3)
        return min_error <= np.deg2rad(cfg["success_heading_deg"])

    def _is_success_pose(self, v: Vehicle) -> bool:
        """Check if vehicle is successfully parked."""
        # Must be in slot bounds
        if not self._is_in_slot_bounds(v):
            return False
        
        # Must be aligned with lane
        if not self._is_heading_aligned(v):
            return False
        
        # Must be nearly stopped
        speed = abs(getattr(v, "speed", 0.0))
        if speed > self.config["success_speed"]:
            return False
            
        return True

    def _is_terminated(self) -> bool:
        """Episode terminates on crash or success."""
        if not self.controlled_vehicles:
            return True
            
        v = self.controlled_vehicles[0]
        return bool(v.crashed or self._is_success_pose(v))

    def _is_truncated(self) -> bool:
        """Episode truncates when time limit is reached."""
        return self.time >= self.config["duration"]