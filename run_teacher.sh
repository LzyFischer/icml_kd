#!/bin/bash

# 定义所有实验
experiments=(
    # 基础实验
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher"
    # "python train_redi.py --train_file ./data/commonsense_qa/train.jsonl --max_steps 100 --num_epochs 1 --run_name teacher"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma"
    
    # # Ablation: no_curriculum
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_curriculum --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_curriculum"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_curriculum"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_curriculum --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_curriculum"
    
    # # Ablation: no_length
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_length --w_length 0.0 --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_length --w_length 0.0"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_length --w_length 0.0 --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_length --w_length 0.0"
    
    # SFT基线
    "python train_sft.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_sft"
    "python train_sft.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_sft"
    "python train_sft.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_sft"
    "python train_sft.py --train_file ./data/math/train.jsonl --num_epochs 1 --max_train_samples 500 --max_length 2048 --run_name teacher_sft"
    "python train_sft.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_sft"
    "python train_sft.py --max_steps 800 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_sft"

    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --max_train_samples 500 --max_length 2048 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_sft"
    # "python train_sft.py --max_steps 800 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_sft"
    

    # Ablation: no_answer
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_answer --w_answer_pred 0.0 --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_answer --w_answer_pred 0.0"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_answer --w_answer_pred 0.0 --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
)

# GPU数量
NUM_GPUS=4

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 任务队列索引
task_idx=0
total_tasks=${#experiments[@]}

# 存储每个GPU当前运行的进程ID
declare -a gpu_pids

# 初始化GPU状态
for i in $(seq 0 $((NUM_GPUS-1))); do
    gpu_pids[$i]=""
done

echo "开始调度 $total_tasks 个任务到 $NUM_GPUS 个GPU上"
echo "日志保存在 $LOG_DIR 目录下"
echo "======================================"

# 函数：在指定GPU上运行任务
run_task() {
    local gpu_id=$1
    local task_id=$2
    local cmd=$3
    
    local log_file="$LOG_DIR/task_${task_id}_gpu_${gpu_id}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id 开始任务 $task_id: $cmd"
    
    # 在后台运行任务，设置CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > $log_file 2>&1 &
    
    # 保存进程ID
    gpu_pids[$gpu_id]=$!
}

# 函数：检查GPU是否空闲
is_gpu_free() {
    local gpu_id=$1
    local pid=${gpu_pids[$gpu_id]}
    
    if [ -z "$pid" ]; then
        return 0  # 空闲
    fi
    
    # 检查进程是否还在运行
    if ps -p $pid > /dev/null 2>&1; then
        return 1  # 忙碌
    else
        return 0  # 空闲
    fi
}

# 主循环：调度任务
while [ $task_idx -lt $total_tasks ]; do
    # 检查每个GPU
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        # 如果还有任务待执行且当前GPU空闲
        if [ $task_idx -lt $total_tasks ] && is_gpu_free $gpu_id; then
            run_task $gpu_id $task_idx "${experiments[$task_idx]}"
            task_idx=$((task_idx+1))
        fi
    done
    
    # 等待一段时间再检查
    sleep 5
done

echo "======================================"
echo "所有任务已提交，等待完成..."

# 等待所有任务完成
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    pid=${gpu_pids[$gpu_id]}
    if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
        echo "等待 GPU $gpu_id 上的任务完成 (PID: $pid)..."
        wait $pid
    fi
done

echo "======================================"
echo "所有任务已完成！"
echo "查看日志: ls -lh $LOG_DIR"