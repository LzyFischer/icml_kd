#!/bin/bash

# Default Configuration
ITERATIONS=3
BASE_MODEL="unsloth/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL="unsloth/Qwen2.5-3B-Instruct"
ORIGINAL_DATA="./data/math/train.jsonl" # 默认值，会被 -f 覆盖
VAL_DATA="./data/math/test.jsonl"
START_STUDENT=""
RUN_SUFFIX="default"

# Help Function
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i <int>    Number of iterations (default: $ITERATIONS)"
    echo "  -b <str>    Base Student Model"
    echo "  -t <str>    Teacher Model"
    echo "  -f <str>    Original Train File (required for generation source)"
    echo "  -v <str>    Validation File"
    echo "  -s <str>    Start Student Checkpoint (optional)"
    echo "  -n <str>    Run Name Suffix (unique identifier)"
    echo "  -h          Show this help message"
    exit 1
}

# Parse Arguments
while getopts "i:b:t:f:v:s:n:h" opt; do
    case ${opt} in
        i) ITERATIONS=$OPTARG ;;
        b) BASE_MODEL=$OPTARG ;;
        t) TEACHER_MODEL=$OPTARG ;;
        f) ORIGINAL_DATA=$OPTARG ;; # 新增：显式接收训练文件
        v) VAL_DATA=$OPTARG ;;
        s) START_STUDENT=$OPTARG ;;
        n) RUN_SUFFIX=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# 推导数据集名称 (用于 Checkpoint 文件夹命名)
# 假设结构: .../dataset_name/test.jsonl -> dataset_name
DATASET_SLUG=$(basename $(dirname "$VAL_DATA"))

# 定义独立的工作目录
WORK_DIR="./iterative_workspace/${DATASET_SLUG}_${RUN_SUFFIX}"
mkdir -p $WORK_DIR

# Initialize Path
if [ -z "$START_STUDENT" ]; then
    CURRENT_STUDENT_PATH=$BASE_MODEL
else
    CURRENT_STUDENT_PATH=$START_STUDENT
fi

echo ">>> Pipeline Start: $RUN_SUFFIX on $DATASET_SLUG"
echo ">>> Train File: $ORIGINAL_DATA"
echo ">>> Val File:   $VAL_DATA"

# --- Loop Start ---
for ((i=1; i<=ITERATIONS; i++)); do
    
    # 1. GENERATE PHASE (vLLM)
    GENERATED_DATA="${WORK_DIR}/iter_${i}_data.jsonl"
    
    python generate_dataset_vllm.py \
        --base_model "$BASE_MODEL" \
        --model_path "$CURRENT_STUDENT_PATH" \
        --input_file "$ORIGINAL_DATA" \
        --output_file "$GENERATED_DATA" \
        --max_tokens 1024 \
        --temperature 1.0 > "${WORK_DIR}/iter_${i}_gen.log" 2>&1

    # 2. TRAIN PHASE (Unsloth)
    BASE_NAME=$(basename "$BASE_MODEL")
    UNIQUE_RUN_NAME="iter_${i}_${RUN_SUFFIX}"
    
    # 注意：这里去掉了 --dataset_name，完全依赖 --val_file 来决定保存路径逻辑
    # 且显式传入 --train_file 和 --val_file
    python train_student.py \
        --teacher_model "$TEACHER_MODEL" \
        --student_model "$CURRENT_STUDENT_PATH" \
        --train_file "$GENERATED_DATA" \
        --val_file "$VAL_DATA" \
        --run_name "$UNIQUE_RUN_NAME" \
        --student_mode "supervised" \
        --loss_type "generalized" \
        --beta 0.5 \
        --lr 2e-5 \
        --batch_size 4 \
        --num_epochs 1 \
        --save_steps 0 \
        --max_length 1024 \
        --use_wandb False > "${WORK_DIR}/iter_${i}_train.log" 2>&1

    # 3. UPDATE POINTER
    NEXT_CHECKPOINT="ckpts/${DATASET_SLUG}/student_model_${UNIQUE_RUN_NAME}"
    
    if [ -d "$NEXT_CHECKPOINT" ]; then
        CURRENT_STUDENT_PATH="$NEXT_CHECKPOINT"
    else
        # Fallback search
        FALLBACK_CHECKPOINT=$(find "ckpts/${DATASET_SLUG}" -type d -name "student_model_${UNIQUE_RUN_NAME}*" | sort | tail -n 1)
        if [ -d "$FALLBACK_CHECKPOINT" ]; then
             CURRENT_STUDENT_PATH="$FALLBACK_CHECKPOINT"
        else
             echo "CRITICAL ERROR: Checkpoint not found."
             exit 1
        fi
    fi

done

echo "$CURRENT_STUDENT_PATH"