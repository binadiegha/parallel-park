#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def main():
    # Set up paths
    project_root = Path(__file__).parent.absolute()
    highway_env_path = project_root / "highway_env"
    dreamerv3_path = project_root / "dreamerv3"
    
    # Add paths to Python path
    sys.path.insert(0, str(highway_env_path))
    sys.path.insert(0, str(dreamerv3_path))
    
    # Set environment variables
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    
    # Import highway_env to register environments
    import highway_env
    from highway_env.envs.parallel_parking_env import ParallelParkingEnv
    
    # Import DreamerV3
    import dreamerv3
    
    # Configuration
    config = dreamerv3.Config()
    
    # Load the parallel parking configuration
    config = config.update({
        'task': 'parallel_parking',
        'logdir': './logs/parallel_parking',
        'run.steps': 3000000,  # 3M steps
        'run.envs': 8,
        'run.train_ratio': 512,
        'batch_size': 16,
        'batch_length': 64,
        'agent.policy_dist_cont': 'bounded_normal',
        'agent.policy.minstd': 0.05,
        'agent.policy.maxstd': 0.5,
        'agent.opt.lr': 6e-5,
    })
    
    # Start training
    dreamerv3.main(config)

if __name__ == "__main__":
    main()