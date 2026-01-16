#!/bin/bash

# Default Configuration
ITERATIONS=3
BASE_MODEL="unsloth/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL="unsloth/Qwen2.5-3B-Instruct"
ORIGINAL_DATA="./data/math/train.jsonl"
VAL_DATA="./data/math/test.jsonl" # 假设结构是 .../dataset_name/test.jsonl
WORK_DIR="./iterative_workspace"
START_STUDENT=""

# Help Function
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i <int>    Number of iterations (default: $ITERATIONS)"
    echo "  -b <str>    Base Student Model (default: $BASE_MODEL)"
    echo "  -t <str>    Teacher Model (default: $TEACHER_MODEL)"
    # echo "  -d <str>    Dataset Name"  <-- 移除手动输入，改为自动推导
    echo "  -v <str>    Validation File (default: $VAL_DATA)"
    echo "  -s <str>    Start Student Checkpoint (optional)"
    echo "  -h          Show this help message"
    exit 1
}

# Parse Arguments (这里加入了 -v 来允许覆盖 VAL_DATA)
while getopts "i:b:t:v:s:h" opt; do
    case ${opt} in
        i) ITERATIONS=$OPTARG ;;
        b) BASE_MODEL=$OPTARG ;;
        t) TEACHER_MODEL=$OPTARG ;;
        v) VAL_DATA=$OPTARG ;;
        s) START_STUDENT=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ==========================================
# 关键修改：从 VAL_DATA 提取父文件夹名称
# 例如: "./data/math/test.jsonl" -> "math"
# ==========================================
DATASET_SLUG=$(basename $(dirname "$VAL_DATA"))
echo "Detected Dataset Name from Val Data: $DATASET_SLUG"

# Initialize Path
if [ -z "$START_STUDENT" ]; then
    CURRENT_STUDENT_PATH=$BASE_MODEL
    echo "Starting from Base Model: $CURRENT_STUDENT_PATH"
else
    CURRENT_STUDENT_PATH=$START_STUDENT
    echo "Starting from Checkpoint: $CURRENT_STUDENT_PATH"
fi

mkdir -p $WORK_DIR

# --- Loop Start ---
for ((i=1; i<=ITERATIONS; i++)); do
    echo "========================================================"
    echo "Starting Iteration $i / $ITERATIONS"
    echo "Current Student: $CURRENT_STUDENT_PATH"
    echo "Dataset: $DATASET_SLUG"
    echo "========================================================"

    # 1. GENERATE PHASE (vLLM)
    GENERATED_DATA="${WORK_DIR}/iter_${i}_data.jsonl"
    
    echo "[Phase 1] Generating data using vLLM..."
    python generate_dataset_vllm.py \
        --base_model "$BASE_MODEL" \
        --model_path "$CURRENT_STUDENT_PATH" \
        --input_file "$ORIGINAL_DATA" \
        --output_file "$GENERATED_DATA" \
        --max_tokens 1024 \
        --temperature 1.0

    # 2. TRAIN PHASE (Unsloth)
    BASE_NAME=$(basename "$BASE_MODEL")
    RUN_NAME="iter_${i}_${BASE_NAME}"
    
    echo "[Phase 2] Training student on generated data..."
    python train_student.py \
        --teacher_model "$TEACHER_MODEL" \
        --student_model "$CURRENT_STUDENT_PATH" \
        --dataset_name "$DATASET_SLUG" \
        --train_file "$GENERATED_DATA" \
        --val_file "$VAL_DATA" \
        --run_name "$RUN_NAME" \
        --student_mode "supervised" \
        --loss_type "generalized" \
        --beta 0.5 \
        --lr 2e-5 \
        --batch_size 4 \
        --num_epochs 1 \
        --save_steps 0 \
        --max_length 1024 \
        --use_wandb True

    # 3. UPDATE POINTER
    # 关键修改：Next Checkpoint 路径使用提取出的 DATASET_SLUG
    NEXT_CHECKPOINT="ckpts/${DATASET_SLUG}/student_model_${RUN_NAME}"
    
    if [ -d "$NEXT_CHECKPOINT" ]; then
        echo "Found new checkpoint: $NEXT_CHECKPOINT"
        CURRENT_STUDENT_PATH="$NEXT_CHECKPOINT"
    else
        echo "CRITICAL ERROR: Checkpoint not found at $NEXT_CHECKPOINT"
        # Fallback search
        FALLBACK_CHECKPOINT=$(find "ckpts/${DATASET_SLUG}" -type d -name "student_model_${RUN_NAME}*" | sort | tail -n 1)
        if [ -d "$FALLBACK_CHECKPOINT" ]; then
             echo "Fallback: Found checkpoint at $FALLBACK_CHECKPOINT"
             CURRENT_STUDENT_PATH="$FALLBACK_CHECKPOINT"
        else
             exit 1
        fi
    fi

done

echo "Pipeline finished. Final model at: $CURRENT_STUDENT_PATH"