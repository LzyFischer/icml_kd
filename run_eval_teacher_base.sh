#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs_eval

run_one () {
  local GPU="$1"
  local DATASET="$2"
  local MODEL="$3"

  local MODEL_TAG
  MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
  local LOG_FILE="logs_eval/${MODEL_TAG}_${DATASET}_gpu${GPU}.log"

  echo "[GPU ${GPU}] START  model=${MODEL} dataset=${DATASET} -> ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="$GPU" \
    python eval_vllm.py --datasets "$DATASET" --model_path "$MODEL" \
    >"$LOG_FILE" 2>&1
  echo "[GPU ${GPU}] DONE   model=${MODEL} dataset=${DATASET}"
}

worker_gpu0 () {
  run_one 0 date          unsloth/gemma-3-270m-it
  run_one 0 strategy_qa   unsloth/gemma-3-270m-it

  run_one 0 strategy_qa   unsloth/Qwen2.5-0.5B-Instruct

  run_one 0 date          unsloth/gemma-3-1b-it
  run_one 0 math          unsloth/gemma-3-1b-it

  run_one 0 strategy_qa   unsloth/Qwen2.5-3B-Instruct
}

worker_gpu1 () {
  run_one 1 math          unsloth/gemma-3-270m-it
  run_one 1 anli          unsloth/gemma-3-270m-it

  run_one 1 arc_challenge unsloth/Qwen2.5-0.5B-Instruct

  run_one 1 commonsense_qa unsloth/gemma-3-1b-it
  run_one 1 anli           unsloth/gemma-3-1b-it

  run_one 1 arc_challenge  unsloth/Qwen2.5-3B-Instruct
}

worker_gpu2 () {
  run_one 2 arc_challenge unsloth/gemma-3-270m-it

  run_one 2 date          unsloth/Qwen2.5-0.5B-Instruct
  run_one 2 math          unsloth/Qwen2.5-0.5B-Instruct

  run_one 2 strategy_qa   unsloth/gemma-3-1b-it

  run_one 2 date          unsloth/Qwen2.5-3B-Instruct
  run_one 2 math          unsloth/Qwen2.5-3B-Instruct
}

worker_gpu3 () {
  run_one 3 commonsense_qa unsloth/gemma-3-270m-it

  run_one 3 commonsense_qa unsloth/Qwen2.5-0.5B-Instruct
  run_one 3 anli           unsloth/Qwen2.5-0.5B-Instruct

  run_one 3 arc_challenge  unsloth/gemma-3-1b-it

  run_one 3 commonsense_qa unsloth/Qwen2.5-3B-Instruct
  run_one 3 anli           unsloth/Qwen2.5-3B-Instruct
}

# 4 workers in parallel, but each worker is serial on its GPU
worker_gpu0 &
worker_gpu1 &
worker_gpu2 &
worker_gpu3 &

wait
echo "All queued jobs finished."