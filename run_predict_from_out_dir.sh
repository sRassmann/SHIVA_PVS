#!/bin/bash
INPUT=$1

# Optional pattern argument, pred_flair.nii.gz is default
PATTERN=${2:-'pred_flair.nii.gz'}

# Running Docker command with the file list
docker run --rm --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 -it \
  --ulimit stack=67108864 --name "$USER"_GPU_0 --gpus 0 -v $(pwd):/workspace/shiva_wmh \
  -v $(realpath ../flairsyn/output):/workspace/flairsyn/output $USER/shiva_tf python \
  predict_from_out_dir.py -i $INPUT -f $PATTERN -s
