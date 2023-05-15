#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODEL=${1:-cis-lmu/glot500-base}
GPU=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

MAX_LENGTH=512
LC=""
BATCH_SIZE=128
DIM=768
NLAYER=12
LAYER=7

DATA_DIR="/PATH/TO/DATA/"
OUTPUT_DIR="/PATH/TO/OUTPUT/"

python evaluate_retrieval_bible.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --embed_size $DIM \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_LENGTH \
    --num_layers $NLAYER \
    --dist cosine $LC \
    --specific_layer $LAYER

