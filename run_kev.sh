#!/bin/bash
# =============================================================================
# kev.sh — Kev scoring + experiments.
#   Kev-9B:   sbatch kev.sh
#   Kev-0.8B: MODEL=kev-0.8b sbatch --job-name=kev_08b kev.sh
#   Debug:    EXTRA_ARGS="--debug_n 50" sbatch --time=00:30:00 kev.sh
# =============================================================================

#SBATCH --job-name=kev_9b
#SBATCH --partition=private-teodoro-gpu
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=Results/%x_%j.log

MODEL=${MODEL:-kev-9b}        # must match main.py --models choices
EXTRA_ARGS=${EXTRA_ARGS:-}

export PYTHONUNBUFFERED=1     # print progress to the log immediately

export KEV_TEMPERATURE=1.0
export KEV_DTYPE=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. Warm up the imports (cold network filesystem -> Kev's 30 s device check times out otherwise)
cd ~/kev
time .venv/bin/python -c "import torch, transformers, fla, kev.serve; print('warm-up ok, cuda:', torch.cuda.is_available())"

# 2. Start the Kev server in the background
.venv/bin/python -m kev.serve --run jaredpalmer/${MODEL}@v1.0 --device cuda --port 8009 \
    > ~/MIE2027/Results/server_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

# 3. Wait for the server; stop the job (and show why) if it crashed
until curl -s localhost:8009/v1/models > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Kev server failed to start. Server log:"
        cat ~/MIE2027/Results/server_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
done
echo "Kev server ready"

# 4. Scoring + experiments
cd ~/MIE2027
.venv/bin/python main.py --models ${MODEL} ${EXTRA_ARGS}

kill $SERVER_PID