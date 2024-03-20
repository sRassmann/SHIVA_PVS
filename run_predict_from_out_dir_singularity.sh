#!/bin/bash
INPUT=$1

# Optional pattern argument, pred_flair.nii.gz is default
PATTERN=${2:-'pred_flair.nii.gz'}

singularity exec --nv -B $(pwd):/workspace/shiva_wmh -B $(realpath ../flairsyn/output):/workspace/flairsyn/output /home/$USER/singularity/shiva_tf.sif python \
  predict_from_out_dir.py -i $INPUT -f $PATTERN -s
