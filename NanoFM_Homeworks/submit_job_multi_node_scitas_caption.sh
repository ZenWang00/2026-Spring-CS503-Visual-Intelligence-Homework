#!/bin/bash
#SBATCH --job-name=multi_node_caption
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=multi_node_caption.out
#SBATCH --error=multi_node_caption.err

# === Accept arguments ===
# Usage:
#   sbatch submit_job_multi_node_scitas_caption.sh <config.yaml>
# If omitted, it falls back to the default nano4M config path.
CONFIG_FILE="${1:-cfgs/nano4M/multiclevr_d6-6w512.yaml}"

# === Initialization ===
set -euo pipefail
set -x
cat "$0"

# IMPORTANT: Slurm executes a copied script from /var/spool/slurmd/...
# so BASH_SOURCE[0] points to the spool location. Use SLURM_SUBMIT_DIR
# to locate the repository reliably.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="${REPO_ROOT}/NanoFM_Homeworks"

# Make config path absolute so it works regardless of cwd.
if [[ "${CONFIG_FILE}" = /* ]]; then
  CONFIG_ABS="${CONFIG_FILE}"
else
  # Allow passing either:
  # - cfgs/... (relative to NanoFM_Homeworks/)
  # - NanoFM_Homeworks/cfgs/... (relative to repo root)
  # - any other path relative to repo root
  if [[ -f "${REPO_ROOT}/${CONFIG_FILE}" ]]; then
    CONFIG_ABS="${REPO_ROOT}/${CONFIG_FILE}"
  elif [[ -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]]; then
    CONFIG_ABS="${SCRIPT_DIR}/${CONFIG_FILE}"
  else
    # Fall back to repo-root relative (gives a clearer error path in logs).
    CONFIG_ABS="${REPO_ROOT}/${CONFIG_FILE}"
  fi
fi

export MASTER_PORT=25678
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_DEBUG=INFO
eval "$(conda shell.bash hook)"
conda activate nanofm

# === Run main script ===
srun --kill-on-bad-exit=1 bash -c "
  cd \"${SCRIPT_DIR}\"

  TORCHRUN_ARGS=\"--node-rank=\${SLURM_PROCID} \
     --master-addr=\${MASTER_ADDR} \
     --master-port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc-per-node=2\"

  echo \${SLURM_PROCID}
  echo \${TORCHRUN_ARGS}
  echo \${SLURMD_NODENAME}

  OMP_NUM_THREADS=1 torchrun \${TORCHRUN_ARGS} run_training.py \
    --config \"${CONFIG_ABS}\"
"
