#!/bin/bash

python merge_files.py \
  --data_directory /DIR/TO/DATA/FOR/EACH/LANGUAGE \
  --save_directory /DIR/FOR/SAVE/MERGE/FILE \
  --experiment_name Glot500 \
  --lg_sampling_factor 0.3 \
  --scale 30 \

