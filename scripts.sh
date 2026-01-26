CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets anli --model_path /scratch/vjd5zr/project/icml_kd/ckpts/anli/student_model__supervised_t_unsloth_gemma-3-1b-it_s_unsloth_gemma-3-270m-it_step_40 --base_model unsloth/gemma-3-270m-it

CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets date --model_path unsloth/Qwen2.5-3B-Instruct
/home/zihan/icml_kd/ckpts/arc_challenge/student_model__supervised_t_unsloth_Qwen2.5-3B-Instruct_s_unsloth_Qwen2.5-0.5B-Instruct_step_250 --base_model unsloth/Qwen2.5-0.5B-Instruct

unsloth/Qwen2.5-0.5B-Instruct
/home/zihan/icml_kd/ckpts/gsm8k/student_model__supervised_t_unsloth_Qwen2.5-3B-Instruct_s_unsloth_Qwen2.5-0.5B-Instruct_step_500 --base_model unsloth/Qwen2.5-0.5B-Instruct

# unsloth/gemma-3-1b-it


export WANDB_API_KEY=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

CUDA_VISIBLE_DEVICES=3 python train_redi.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1


CUDA_VISIBLE_DEVICES=2 python train_student.py --config config.yaml --teacher_model /home/vjd5zr/project/icml_kd/ckpts/grpo_curriculum/checkpoint-318


CUDA_VISIBLE_DEVICES=2 python train_sft.py --train_file "./data/gsm8k/train.jsonl" --model_name "unsloth/Qwen2.5-3B-Instruct" --teacher_model_name "unsloth/Qwen2.5-0.5B-Instruct" --kd_alpha 0.5 --kd_temperature 2.0 --batch_size 2 --use_wandb



CUDA_VISIBLE_DEVICES=0 python train_redi.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1 --run_name "teacher"
CUDA_VISIBLE_DEVICES=1 python train_redi.py --train_file "./data/anli/train.jsonl" --num_epochs 5 --run_name "teacher"
CUDA_VISIBLE_DEVICES=2 python train_redi.py --train_file "./data/date/train.jsonl" --num_epochs 5 --run_name "teacher"
CUDA_VISIBLE_DEVICES=3 python train_redi.py --train_file "./data/math/train.jsonl" --num_epochs 1 --run_name "teacher"
CUDA_VISIBLE_DEVICES=4 python train_redi.py --train_file "./data/arc_challenge/train.jsonl" --num_epochs 2 --run_name "student" 
CUDA_VISIBLE_DEVICES=5 python train_redi.py --train_file "./data/commonsense_qa/train.jsonl" --num_epochs 1 --run_name "student"

# Ablation experiments
CUDA_VISIBLE_DEVICES=0 python train_redi.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1 --run_name "teacher_no_curriculum" 
CUDA_VISIBLE_DEVICES=1 python train_redi.py --train_file "./data/anli/train.jsonl" --num_epochs 5 --run_name "teacher_no_curriculum"
CUDA_VISIBLE_DEVICES=2 python train_redi.py --train_file "./data/date/train.jsonl" --num_epochs 5 --run_name "teacher_no_curriculum"
CUDA_VISIBLE_DEVICES=3 python train_redi.py --train_file "./data/math/train.jsonl" --num_epochs 1 --run_name "teacher_no_curriculum"
CUDA_VISIBLE_DEVICES=4 python train_redi.py --train_file "./data/arc_challenge/train.jsonl" --num_epochs 2 --run_name "student_no_curriculum"
CUDA_VISIBLE_DEVICES=5 python train_redi.py --train_file "./data/commonsense_qa/train.jsonl" --num_epochs 1 --run_name "student_no_curriculum"

CUDA_VISIBLE_DEVICES=0 python train_redi.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1 --run_name "teacher_no_length" --w_length 0.0
CUDA_VISIBLE_DEVICES=1 python train_redi.py --train_file "./data/anli/train.jsonl" --num_epochs 5 --run_name "teacher_no_length" --w_length 0.0
CUDA_VISIBLE_DEVICES=2 python train_redi.py --train_file "./data/date/train.jsonl" --num_epochs 5 --run_name "teacher_no_length" --w_length 0.0
CUDA_VISIBLE_DEVICES=3 python train_redi.py --train_file "./data/math/train.jsonl" --num_epochs 1 --run_name "teacher_no_length" --w_length 0.0
CUDA_VISIBLE_DEVICES=4 python train_redi.py --train_file "./data/arc_challenge/train.jsonl" --num_epochs 2 --run_name "student_no_length" --w_length 0.0
CUDA_VISIBLE_DEVICES=5 python train_redi.py --train_file "./data/commonsense_qa/train.jsonl" --num_epochs 1 --run_name "student_no_length" --w_length 0.0

CUDA_VISIBLE_DEVICES=0 python train_redi.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1 --run_name "teacher_no_answer" --w_answer_pred 0.0
CUDA_VISIBLE_DEVICES=1 python train_redi.py --train_file "./data/anli/train.jsonl" --num_epochs 5 --run_name "teacher_no_answer" --w_answer_pred 0.0
CUDA_VISIBLE_DEVICES=2 python train_redi.py --train_file "./data/date/train.jsonl" --num_epochs 5 --run_name "teacher_no_answer" --w_answer_pred 0.0
CUDA_VISIBLE_DEVICES=3 python train_redi.py --train_file "./data/math/train.jsonl" --num_epochs 1 --run_name "teacher_no_answer" --w_answer_pred 0.0
CUDA_VISIBLE_DEVICES=4 python train_redi.py --train_file "./data/arc_challenge/train.jsonl" --num_epochs 2 --run_name "student_no_answer" --w_answer_pred 0.0
CUDA_VISIBLE_DEVICES=5 python train_redi.py --train_file "./data/commonsense_qa/train.jsonl" --num_epochs 1 --run_name "student_no_answer" --w_answer_pred 0.0

CUDA_VISIBLE_DEVICES=0 python train_sft.py --train_file "./data/strategy_qa/train.jsonl" --num_epochs 1
CUDA_VISIBLE_DEVICES=1 python train_sft.py --train_file "./data/anli/train.jsonl" --num_epochs 5
CUDA_VISIBLE_DEVICES=2 python train_sft.py --train_file "./data/date/train.jsonl" --num_epochs 5
CUDA_VISIBLE_DEVICES=3 python train_sft.py --train_file "./data/math/train.jsonl" --num_epochs 1
CUDA_VISIBLE_DEVICES=4 python train_sft.py --train_file "./data/arc_challenge/train.jsonl" --num_epochs 2
CUDA_VISIBLE_DEVICES=5 python train_sft.py --train_file "./data/commonsense_qa/train.jsonl" --num_epochs 1