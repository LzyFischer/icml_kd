CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --datasets date --model_path /scratch/vjd5zr/project/icml_kd/ckpts/date/teacher/checkpoint-300 --base_model unsloth/Qwen2.5-3B-Instruct

/scratch/vjd5zr/project/icml_kd/ckpts/math/student_model__student_t_unsloth_gemma-3-1b-it_s_unsloth_gemma-3-270m-it_step_40

# unsloth/gemma-3-1b-it


export WANDB_API_KEY=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

CUDA_VISIBLE_DEVICES=1 python train_redi.py 
--teacher_model /home/vjd5zr/project/icml_kd/ckpts/checkpoint-160


python train_student.py --config config.yaml --teacher_model /home/vjd5zr/project/icml_kd/ckpts/grpo_curriculum/checkpoint-318
