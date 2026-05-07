#!/bin/bash
exp_config_path=${1:-"benchmark_configs/exp_0.json"}

use_openai=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export AUTO_SPLIT=${AUTO_SPLIT:-0}
export SUBSET_LEN=${SUBSET_LEN:-1000}
export USE_COT=1
export TOKENIZERS_PARALLELISM=false
export DIST_TIMEOUT=99999999999
export UNSLOTH_DISABLE_FAST_GENERATION="1"

exp_config_stem=$(basename "$exp_config_path" .json)
workdir="${WORKDIR_OVERRIDE:-benchmark_results/n_samples_${SUBSET_LEN}/${exp_config_stem}/}"
mkdir -p "$workdir"
cp "$exp_config_path" "$workdir"

export PYTHONPATH=$PWD:$PYTHONPATH
number_of_gpus=$(($(grep -o "," <<<"$CUDA_VISIBLE_DEVICES" | wc -l) + 1))
echo "Number of GPUs: $number_of_gpus"

if [ "$number_of_gpus" -gt 1 ]; then
    if [ "$AUTO_SPLIT" = 1 ]; then
        number_of_gpus=1
    fi
    torchrun --nproc-per-node=$number_of_gpus --master-port 29555 run.py --work-dir "$workdir" --verbose --config "$exp_config_path"
else
    python run.py --work-dir "$workdir" --verbose --config "$exp_config_path"
fi
