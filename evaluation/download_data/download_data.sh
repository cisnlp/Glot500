#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR

# Helper function to download the UD-POS data.
# In order to ensure backwards compatibility with the XTREME evaluation,
# languages in XTREME use the UD version used in the original paper; for the new
# languages in XTREME-R, we use a more recent UD version.
function download_treebank {
    base_dir=$2
    out_dir=$3
    url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4923/ud-treebanks-v2.11.tgz
    ud_version="2.11"
    echo "$url"
    curl -s --remote-name-all "$url"

    tar -xzf $base_dir/ud-treebanks-v$ud_version.tgz

    for x in $base_dir/ud-treebanks-v$ud_version/*/*.conllu; do
        file="$(basename $x)"
        IFS='_' read -r -a array <<< "$file"
        lang=${array[0]}
        lang_dir=$out_dir/$lang/
        mkdir -p $lang_dir
        y=$lang_dir/${file/conllu/conll}
        if [ ! -f "$y" ]; then
            echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
            python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms
        else
            echo "${y} exists"
        fi
    done
}

# Download UD-POS dataset.
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll/
    mkdir -p $out_dir
    cd $base_dir

    download_treebank all $base_dir $out_dir

    cd $REPO
    python $REPO/utils_preprocess.py --data_dir $out_dir/ --output_dir $DIR/udpos/ --task  udpos
    echo "Successfully downloaded data at $DIR/udpos" >> $DIR/download.log
}

function download_panx {
    echo "Download panx NER dataset"
    base_dir=$DIR/panx_dataset/
    python $REPO/utils_preprocess.py \
        --data_dir $base_dir \
        --output_dir $DIR/panx \
        --task panx
    echo "Successfully downloaded data at $DIR/panx" >> $DIR/download.log
}

function download_tatoeba {
    base_dir=$DIR/tatoeba/
    wget https://github.com/facebookresearch/LASER/archive/main.zip
    unzip -qq -o main.zip -d $base_dir/
    mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
    echo "Successfully downloaded data at $DIR/tatoeba" >> $DIR/download.log
}

download_tatoeba
download_panx
download_udpos

