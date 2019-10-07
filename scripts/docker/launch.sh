#!/bin/bash

NV_GPU='1' nvidia-docker run -it --rm \
  --runtime=nvidia \
  --shm-size=1g \
  -p 8888:8888 \
  --net=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD/data:/data \
  -v $PWD:/workspace/recommendation \
  -it \
  ncf_ghc jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

