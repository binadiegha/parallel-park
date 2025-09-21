#!/bin/bash
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python scripts/train_dreamerv3.py "$@"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia JAX_PLATFORMS=cuda python ./dreamerv3/dreamerv3/main.py --configs parallel_parking --logdir ./logs/parallel_parking "$@"

