import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.10.1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    from highway_env.envs.common.abstract import MultiAgentWrapper

    
    # parking_env.py
    register(
        id="parking-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnv",
    )

    register(
        id="parking-ActionRepeat-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvActionRepeat",
    )

    register(
        id="parking-parked-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvParkedVehicles",
    )

    # parallel-parking
    register(
        id="parallel-parking-v0",
        entry_point="highway_env.envs.parallel_parking_env:ParallelParkingEnv"
    )

     # parallel-parking
    register(
        id="custom-parking-env-v0",
        entry_point="highway_env.envs.custom_parking_env:CustomParkingEnv"
    )
    
    
    register(
        id="parallel-parking-hard-v0",
        entry_point="highway_env.envs.parallel_parking_env:ParallelParkingEnvHard"
    )
    
    register(
        id="parallel-parking-easy-v0",
        entry_point="highway_env.envs.parallel_parking_env:ParallelParkingEnvEasy"
    )


_register_highway_envs()
