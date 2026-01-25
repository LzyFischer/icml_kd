#!/bin/bash

python train_student.py --teacher_model unsloth/Qwen2.5-3B-Instruct --student_model unsloth/Qwen2.5-0.5B-Instruct --val_file ./kl_partition_results/date/rank_0.jsonl --train_file ./kl_partition_results/date/rank_0.jsonl --num_epochs 1 --run_name student_rank_0 
python train_student.py --teacher_model unsloth/Qwen2.5-3B-Instruct --student_model unsloth/Qwen2.5-0.5B-Instruct --val_file ./kl_partition_results/date/rank_1.jsonl --train_file ./kl_partition_results/date/rank_1.jsonl --num_epochs 1 --run_name student_rank_1 
python train_student.py --teacher_model unsloth/Qwen2.5-3B-Instruct --student_model unsloth/Qwen2.5-0.5B-Instruct --val_file ./kl_partition_results/date/rank_2.jsonl --train_file ./kl_partition_results/date/rank_2.jsonl --num_epochs 1 --run_name student_rank_2 
python train_student.py --teacher_model unsloth/Qwen2.5-3B-Instruct --student_model unsloth/Qwen2.5-0.5B-Instruct --val_file ./kl_partition_results/date/rank_3.jsonl --train_file ./kl_partition_results/date/rank_3.jsonl --num_epochs 1 --run_name student_rank_3 
python train_student.py --teacher_model unsloth/Qwen2.5-3B-Instruct --student_model unsloth/Qwen2.5-0.5B-Instruct --val_file ./kl_partition_results/date/rank_4.jsonl --train_file ./kl_partition_results/date/rank_4.jsonl --num_epochs 1 --run_name student_rank_4 

# Modify --num_epochs or other hyperparameters as needed.
