#!/usr/bin/env bash
set -euo pipefail

# ===== knobs =====
GEN_TEMPERATURE=1.0
GEN_MAX_NEW_TOKENS=1024
NUM_GENERATIONS=5
NUM_PASSES=1
MAX_DATA_SAMPLES=""   # e.g. 500 for debug, empty = full train

# train knobs (match your argparse)
NUM_EPOCHS=4
BATCH_SIZE=8
LOSS_TYPE="forward"          # sft | forward | reverse | generalized
STUDENT_MODE="supervised"  # supervised | on_policy | teacher_generated
TRAIN_TEMPERATURE=1.0        # for distill loss temperature
TEACHER_GEN_TEMPERATURE=0.8
MAX_LENGTH=1024
MIXED_PRECISION="bf16"
USE_UNSLOTH="True"
LOAD_IN_4BIT="True"
LR="2e-5"
EVAL_STEPS=999999            # set very large to avoid mid-training eval (optional)
SAVE_STEPS=10                 # only save at end

# eval_vllm knobs
EVAL_TP_SIZE=1
EVAL_MAX_TOKENS=2048
EVAL_TEMPERATURE=0
EVAL_LIMIT=""                # e.g. 500, empty = full test

# dirs
CAND_DIR="candidates"
RANK_DIR="kl_partition_results"
RESULTS_DIR="results"
DATA_ROOT="data"

DATASETS=("date" "anli")

# model families: family|teacher|student_base
MODEL_PAIRS=(
  "qwen25|unsloth/Qwen2.5-3B-Instruct|unsloth/Qwen2.5-0.5B-Instruct"
  "gemma3|unsloth/gemma-3-1b-it|unsloth/gemma-3-270m-it"
)

maybe_arg() {
  # usage: maybe_arg "--flag" "value"
  local flag="$1"
  local val="$2"
  if [[ -n "${val}" ]]; then
    echo "${flag} ${val}"
  else
    echo ""
  fi
}

for dataset in "${DATASETS[@]}"; do
  TRAIN_FILE="${DATA_ROOT}/${dataset}/train.jsonl"

  for triple in "${MODEL_PAIRS[@]}"; do
    IFS='|' read -r family teacher student_base <<< "${triple}"

    echo "===================================================="
    echo "DATASET=${dataset}  FAMILY=${family}"
    echo "Teacher=${teacher}"
    echo "StudentBase=${student_base}"
    echo "===================================================="

    prefix="${dataset}/${family}"
    cand_file="${CAND_DIR}/${prefix}/candidates.jsonl"
    rank_out="${RANK_DIR}/${prefix}"
    mkdir -p "$(dirname "${cand_file}")" "${rank_out}"

    echo ""
    echo "== 1) vLLM generate candidates (dataset_pass) =="
    python generate_candidates.py \
      --teacher_model "${teacher}" \
      --input_file "${TRAIN_FILE}" \
      --output_file "${cand_file}" \
      --num_generations "${NUM_GENERATIONS}" \
      --num_passes "${NUM_PASSES}" \
      --dataset_pass \
      --temperature "${GEN_TEMPERATURE}" \
      --max_new_tokens "${GEN_MAX_NEW_TOKENS}" \
      --use_vllm \
      $(maybe_arg "--max_data_samples" "${MAX_DATA_SAMPLES}")

    echo ""
    echo "== 2) Rank candidates by KL -> rank_0..rank_4 =="
    python rank_candidates_by_kl.py \
      --teacher_model "${teacher}" \
      --student_model "${student_base}" \
      --candidate_file "${cand_file}" \
      --output_dir "${rank_out}"

    echo ""
    echo "== 3) Train 5 students (rank_0..rank_4) =="
    for i in 0 1 2 3 4; do
      train_file="${rank_out}/rank_${i}.jsonl"
      run_name="${dataset}_${family}_rank${i}"
      # IMPORTANT:
      # - pass --dataset_name ${dataset} to control ckpts/<dataset>/...
      # - no --output_dir (your train_student.py doesn't have it)
      python train_student.py \
        --teacher_model "${teacher}" \
        --student_model "${student_base}" \
        --dataset_name "${dataset}" \
        --train_file "${train_file}" \
        --val_file "${train_file}" \
        --prompt_column "instruction" \
        --response_column "response" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --loss_type "${LOSS_TYPE}" \
        --student_mode "${STUDENT_MODE}" \
        --max_length "${MAX_LENGTH}" \
        --max_new_tokens "${GEN_MAX_NEW_TOKENS}" \
        --mixed_precision "${MIXED_PRECISION}" \
        --use_unsloth "${USE_UNSLOTH}" \
        --load_in_4bit "${LOAD_IN_4BIT}" \
        --run_name "${run_name}" \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}"

      # train_student.py will save to:
      # ckpts/<dataset>/student_model_<run_name>
    done

    echo ""
    echo "== 4) eval_vllm for 5 students on BOTH datasets (date + anli) =="
    # 如果你只想评当前 dataset，把 "--datasets date anli" 改成 "--datasets ${dataset}"
    for i in 0 1 2 3 4; do
      run_name="${dataset}_${family}_rank${i}"
      model_path="ckpts/${dataset}/student_model_${run_name}"

      python eval_vllm.py \
        --model_path "${model_path}" \
        --base_model "${student_base}" \
        --datasets date anli \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}" \
        --tp_size "${EVAL_TP_SIZE}" \
        --max_tokens "${EVAL_MAX_TOKENS}" \
        --temperature "${EVAL_TEMPERATURE}" \
        $(maybe_arg "--limit" "${EVAL_LIMIT}")
    done

    echo "✅ Done: dataset=${dataset} family=${family}"
    echo ""
  done
done

echo "🎉 All finished."