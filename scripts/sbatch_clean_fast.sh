#!/bin/bash
#SBATCH --job-name=smolvlm2-clean-fast
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=<ANON_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
LOG_DIR="${ROOT_DIR}/logs"
ENV_ROOT="<ANON_ROOT>/micromamba-root/envs/eccts310"
PYTHON_BIN="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"
CONFIG_PATH="${CONFIG_PATH:-${1:-}}"

if [ -z "${CONFIG_PATH}" ]; then
    echo "CONFIG_PATH is required" >&2
    exit 1
fi
export CONFIG_PATH

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load gcc || true
    module load python3/3.10.12 || true
    module load cuda/12.5 || true
fi

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHON_BIN
export TORCHRUN_BIN
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
export DIST_TIMEOUT=99999999999
export UNSLOTH_DISABLE_FAST_GENERATION=1
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""

CONFIG_STEM="$(basename "${CONFIG_PATH}" .json)"
export CACHE_PATH="${CACHE_PATH:-${ROOT_DIR}/.runtime_cache/${CONFIG_STEM}}"
mkdir -p "${CACHE_PATH}" "${CACHE_PATH}/tmp"
export SHARED_HF_CACHE="${SHARED_HF_CACHE:-<ANON_ROOT>/hf_cache}"
mkdir -p "${SHARED_HF_CACHE}"
export HF_HOME="${HF_HOME:-$SHARED_HF_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SHARED_HF_CACHE}"
export DATASETS_CACHE="${DATASETS_CACHE:-$SHARED_HF_CACHE}"
export TORCH_HOME="${TORCH_HOME:-$SHARED_HF_CACHE}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_PATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$CACHE_PATH}"
export TMPDIR="${TMPDIR:-$CACHE_PATH/tmp}"
export LMUData="${LMUData:-$CACHE_PATH/LMUData}"
export NO_ALBUMENTATIONS_UPDATE=1
export NLTK_DATA="${NLTK_DATA:-<ANON_ROOT>/nltk_data}"

export SSL_CERT_FILE=""
export REQUESTS_CA_BUNDLE=""
SITE_DIR="${CACHE_PATH}/pysite_${SLURM_JOB_ID:-$$}"
mkdir -p "${SITE_DIR}"
echo "import ssl; ssl._create_default_https_context = ssl._create_unverified_context" > "${SITE_DIR}/sitecustomize.py"
export PYTHONPATH="${SITE_DIR}:${PYTHONPATH}"

# Enable 2-GPU data parallel execution in scripts/benchmark.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export AUTO_SPLIT=0

echo "Running ${CONFIG_PATH} on $(hostname)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON=${PYTHON_BIN}"
echo "Cache=${CACHE_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Starting $(date)"

bash scripts/benchmark.sh "${CONFIG_PATH}"

echo "Finished $(date): ${CONFIG_PATH}"
