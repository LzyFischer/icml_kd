# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets date --model_path unsloth/Qwen2.5-3B-Instruct
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --temperature 1 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/teacher_sft/step_20_lora --base_model unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --temperature 1 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/teacher_sft/step_40_lora --base_model unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --temperature 1 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/teacher_sft/step_60_lora --base_model unsloth/Qwen2.5-3B-Instruct 
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --temperature 1 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/teacher_sft/step_60_lora --base_model unsloth/Qwen2.5-3B-Instruct &

# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets date --model_path unsloth/gemma-3-270m-it &
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets math --model_path unsloth/gemma-3-270m-it &
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets arc_challenge --model_path unsloth/gemma-3-270m-it &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets commonsense_qa --model_path unsloth/gemma-3-270m-it &
# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets strategy_qa --model_path unsloth/gemma-3-270m-it
CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets anli --model_path unsloth/gemma-3-270m-it --temperature 1 &

# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets date --model_path unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets commonsense_qa --model_path unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets strategy_qa --model_path unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets arc_challenge --model_path unsloth/Qwen2.5-0.5B-Instruct
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets math --model_path unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets anli --model_path unsloth/Qwen2.5-0.5B-Instruct


# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets date --model_path unsloth/gemma-3-1b-it &
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets commonsense_qa --model_path unsloth/gemma-3-1b-it &
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets strategy_qa --model_path unsloth/gemma-3-1b-it &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets arc_challenge --model_path unsloth/gemma-3-1b-it
# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets math --model_path unsloth/gemma-3-1b-it &
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets anli --model_path unsloth/gemma-3-1b-it
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets date --model_path unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets commonsense_qa --model_path unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets strategy_qa --model_path unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --datasets arc_challenge --model_path unsloth/Qwen2.5-3B-Instruct
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --datasets math --model_path unsloth/Qwen2.5-3B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --datasets anli --model_path unsloth/Qwen2.5-3B-Instruct







# CUDA_VISIBLE_DEVICES=1 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model /home/zihan/icml_kd/ckpts/date/teacher_sft/step_20_lora --num_epochs 2 --batch_size 4 &
# CUDA_VISIBLE_DEVICES=2 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model /home/zihan/icml_kd/ckpts/date/teacher_sft/step_40_lora --num_epochs 2 --batch_size 4 &
# CUDA_VISIBLE_DEVICES=3 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model /home/zihan/icml_kd/ckpts/date/teacher_sft/step_60_lora --num_epochs 2 --batch_size 4
# CUDA_VISIBLE_DEVICES=1 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model /home/zihan/icml_kd/ckpts/date/teacher_sft/step_60_lora --num_epochs 2 --batch_size 4 &
# CUDA_VISIBLE_DEVICES=2 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model /home/zihan/icml_kd/ckpts/date/teacher_sft/step_80_lora --num_epochs 2 --batch_size 4 &
# CUDA_VISIBLE_DEVICES=3 python train_student.py --train_file ./data/date/train.jsonl --val_file ./data/date/test.jsonl --teacher_model unsloth/Qwen2.5-3B-Instruct --num_epochs 2 --batch_size 4

# CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --temperature 0.5 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/student_model__supervised_t__home_zihan_icml_kd_ckpts_date_teacher_sft_step_20_lora_s_unsloth_Qwen2.5-0.5B-Instruct_step_80 --base_model unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --temperature 0.5 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/student_model__supervised_t__home_zihan_icml_kd_ckpts_date_teacher_sft_step_40_lora_s_unsloth_Qwen2.5-0.5B-Instruct_step_80 --base_model unsloth/Qwen2.5-0.5B-Instruct &
# CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --temperature 0.5 --datasets date --model_path /home/zihan/icml_kd/ckpts/date/student_model__supervised_t__home_zihan_icml_kd_ckpts_date_teacher_sft_step_60_lora_s_unsloth_Qwen2.5-0.5B-Instruct_step_80 --base_model unsloth/Qwen2.5-0.5B-Instruct
# CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --temperature 0.5 --datasets date --model_path student_model__supervised_t_unsloth_Qwen2.5-3B-Instruct_s_unsloth_Qwen2.5-0.5B-Instruct_step_80 --base_model unsloth/Qwen2.5-0.5B-Instruct &