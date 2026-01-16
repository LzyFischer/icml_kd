from unsloth import FastModel
import unsloth
import argparse
import os
import random
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
import tqdm
import warnings
import pdb

# Import from separated files
from load_data import load_data_source, collate_fn_builder, Example
from utils import (
    str2bool, ensure_dtype_match, generalized_jsd_loss, compute_grpo_loss,
    generate_on_policy, compute_verifiable_reward, compute_difficulty_reward, 
    evaluate_student, HAS_UNSLOTH
)



warnings.filterwarnings("ignore", category=UserWarning, module="unsloth.kernels.utils")
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

# -----------------------------------------------------------------------------
# Two-Phase Training Logic
# -----------------------------------------------------------------------------

def teacher_phase(
    teacher: PreTrainedModel,
    teacher_ref: PreTrainedModel,
    student: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: dict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> dict:
    """
    Teacher phase: Apply GRPO to teacher model.
    """
    teacher.train()
    student.eval()
    
    prompts = batch["prompts"]
    ground_truths = batch["responses"]
    
    # Generate multiple trajectories per prompt
    all_trajectories = generate_on_policy(
        accelerator.unwrap_model(teacher),
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.teacher_temperature,
        num_samples=args.num_teacher_samples,
        use_unsloth=args.use_unsloth,
    )
    
    # Flatten trajectories for batch processing
    flat_prompts = []
    flat_responses = []
    verifiable_rewards = []
    
    for prompt, trajs, gt in zip(prompts, all_trajectories, ground_truths):
        for traj in trajs:
            flat_prompts.append(prompt)
            flat_responses.append(traj)
            verifiable_rewards.append(compute_verifiable_reward(traj, gt))
    
    # Create batch from trajectories
    traj_examples = [Example(prompt=p, response=r) for p, r in zip(flat_prompts, flat_responses)]
    collate = collate_fn_builder(tokenizer, args.max_length)
    traj_batch = collate(traj_examples)
    
    # Move to device
    traj_input_ids = traj_batch["input_ids"].to(teacher.device)
    traj_attention_mask = traj_batch["attention_mask"].to(teacher.device)
    traj_labels = traj_batch["labels"].to(teacher.device)
    
    # Get logits from all models
    teacher_outputs = teacher(input_ids=traj_input_ids, attention_mask=traj_attention_mask)
    with torch.no_grad():
        ref_outputs = teacher_ref(input_ids=traj_input_ids, attention_mask=traj_attention_mask)
        student_outputs = student(input_ids=traj_input_ids, attention_mask=traj_attention_mask)
    
    # Compute log probabilities for generated tokens only
    mask = (traj_labels != -100).float()
    
    teacher_logprobs = F.log_softmax(teacher_outputs.logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
    student_logprobs = F.log_softmax(student_outputs.logits, dim=-1)
    
    # Gather log probs for actual tokens
    teacher_lp = torch.gather(teacher_logprobs, -1, traj_input_ids.unsqueeze(-1)).squeeze(-1)
    ref_lp = torch.gather(ref_logprobs, -1, traj_input_ids.unsqueeze(-1)).squeeze(-1)
    student_lp = torch.gather(student_logprobs, -1, traj_input_ids.unsqueeze(-1)).squeeze(-1)
    
    # Compute difficulty reward
    difficulty_rewards = compute_difficulty_reward(
        teacher_outputs.logits[:, :-1, :],
        student_outputs.logits[:, :-1, :],
        mask[:, 1:],
        beta=args.beta,
        temperature=args.temperature,
    )
    
    # Combine rewards
    verifiable_rewards_tensor = torch.tensor(verifiable_rewards, device=teacher.device, dtype=torch.float32)
    combined_rewards = verifiable_rewards_tensor + args.difficulty_weight * difficulty_rewards
    
    # Compute GRPO loss
    loss, metrics = compute_grpo_loss(
        teacher_lp,
        ref_lp,
        student_lp,
        combined_rewards,
        mask,
        kl_coef=args.teacher_kl_coef,
        student_kl_coef=args.teacher_student_kl_coef,
    )
    
    # Update teacher
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    
    metrics["teacher/loss"] = loss.item()
    metrics["teacher/verifiable_reward"] = verifiable_rewards_tensor.mean().item()
    metrics["teacher/difficulty_reward"] = difficulty_rewards.mean().item()
    
    return metrics

def student_phase(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: dict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    phase_progress: float,
) -> dict:
    """
    Student phase: Apply mixed distillation with three modes:
    1. Supervised (existing dataset answers)
    2. Student on-policy (student generates)
    3. Teacher-generated (teacher generates for student to learn)
    """
    student.train()
    teacher.eval()
    
    # Three-way mixing with scheduled probabilities
    # phase_progress = 1
    on_policy_prob = args.lambda_on_policy * phase_progress
    teacher_gen_prob = args.lambda_teacher_gen * phase_progress
    
    # Normalize so supervised gets the remainder
    total_dynamic = on_policy_prob + teacher_gen_prob
    if total_dynamic > 1.0:
        on_policy_prob /= total_dynamic
        teacher_gen_prob /= total_dynamic
        total_dynamic = 1.0
    
    supervised_prob = 1.0 - total_dynamic
    
    # Sample which mode to use
    rand_val = random.random()
    if rand_val < on_policy_prob:
        # Mode 1: Student on-policy distillation
        prompts = batch["prompts"]
        
        with torch.no_grad():
            FastModel.for_inference(student)
            student_responses = generate_on_policy(
                accelerator.unwrap_model(student),
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                num_samples=1,
                use_unsloth=args.use_unsloth,
            )
            student_responses = [r[0] for r in student_responses]
            FastModel.for_training(student)
        
        new_examples = [Example(prompt=p, response=r) for p, r in zip(prompts, student_responses)]
        collate = collate_fn_builder(tokenizer, args.max_length)
        new_batch = collate(new_examples)
        
        input_ids = new_batch["input_ids"].to(student.device)
        attention_mask = new_batch["attention_mask"].to(student.device)
        labels = new_batch["labels"].to(student.device)
        
        mode = "student_on_policy"
        
    elif rand_val < on_policy_prob + teacher_gen_prob:
        # Mode 2: Teacher-generated distillation (NEW!)
        prompts = batch["prompts"]
        prompt_ids = batch["input_ids"][:, :batch["attention_mask"].sum(dim=1).min()]  # 裁剪到最短prompt
        prompt_mask = batch["attention_mask"][:, :prompt_ids.size(1)]

        with torch.no_grad():
            FastModel.for_inference(accelerator.unwrap_model(teacher))
            teacher_responses = generate_on_policy(
                accelerator.unwrap_model(teacher),
                tokenizer,
                prompts,
                prompt_ids,
                prompt_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.teacher_gen_temperature,  # Can use different temp
                num_samples=1,
                use_unsloth=args.use_unsloth,
            )
            teacher_responses = [r[0] for r in teacher_responses]
            FastModel.for_training(accelerator.unwrap_model(teacher))
        
        new_examples = [Example(prompt=p, response=r) for p, r in zip(prompts, teacher_responses)]
        collate = collate_fn_builder(tokenizer, args.max_length)
        new_batch = collate(new_examples)
        
        input_ids = new_batch["input_ids"].to(student.device)
        attention_mask = new_batch["attention_mask"].to(student.device)
        labels = new_batch["labels"].to(student.device)
        
        mode = "teacher_generated"
        
    else:
        # Mode 3: Supervised distillation (existing dataset answers)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mode = "supervised"
    
    # Compute logits
    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad():
        teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
    
    mask = (labels != -100).float()
    loss = generalized_jsd_loss(
        student_outputs.logits[:, :-1, :],
        teacher_outputs.logits[:, :-1, :],
        mask=mask[:, 1:],
        beta=args.beta,
        temperature=args.temperature,
    )
    
    # Update student
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    
    metrics = {
        f"student/loss_{mode}": loss.item(),
        "student/on_policy_prob": on_policy_prob,
        "student/teacher_gen_prob": teacher_gen_prob,
        "student/supervised_prob": supervised_prob,
    }
    
    return metrics

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_unsloth and not HAS_UNSLOTH:
        raise ImportError("Unsloth requested but not installed.")

    if args.run_name is None:
        s_name = args.student_model.replace("/", "_")
        t_name = args.teacher_model.replace("/", "_")
        args.run_name = f"{args.prefix}_t_{t_name}_s_{s_name}"

    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None, mixed_precision=args.mixed_precision)
    if args.use_wandb:
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": {"name": args.run_name, "id": None}})

    if accelerator.is_main_process:
        print(f"Teacher: {args.teacher_model} | Student: {args.student_model}")
        print(f"Two-Phase Training: Teacher steps={args.teacher_steps}, Student steps={args.student_steps}")

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    if args.use_unsloth:
        student, tokenizer = FastModel.from_pretrained(
            model_name=args.student_model,
            max_seq_length=args.max_length,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dt
        )
        
        student = FastModel.get_peft_model(
            student,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            finetune_vision_layers=False,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

        teacher, _ = FastModel.from_pretrained(
            model_name=args.teacher_model,
            max_seq_length=args.max_length,
            load_in_4bit=args.load_in_4bit, 
            dtype=args.dt
        )
        
        teacher = FastModel.get_peft_model(
            teacher,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            finetune_vision_layers=False,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        
        # Reference teacher (frozen copy)
        teacher_ref, _ = FastModel.from_pretrained(
            model_name=args.teacher_model,
            max_seq_length=args.max_length,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dt
        )
        FastModel.for_inference(teacher_ref)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True, padding_side="left")
        
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        teacher_ref = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        teacher_ref.eval()
        teacher_ref.requires_grad_(False)
        
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        student.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if accelerator.is_main_process:
        print(f"Tokenizer Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_source = args.train_file if args.train_file else args.dataset_name
    val_source = args.val_file if args.val_file else args.dataset_name
    val_split = "train" if args.val_file else "test"

    train_examples = load_data_source(train_source, split="train", prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_train_samples)
    val_examples = load_data_source(val_source, split=val_split, prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_val_samples)

    collate = collate_fn_builder(tokenizer, args.max_length)
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    student, teacher, teacher_ref, teacher_optimizer, student_optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        student, teacher, teacher_ref, teacher_optimizer, student_optimizer, train_dataloader, val_dataloader
    )

    global_step = 0
    total_iterations = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm.tqdm(range(total_iterations), disable=not accelerator.is_main_process)

    args.save_dataet_name = args.dataset_name.replace("/", "_") if not args.train_file else os.path.basename(os.path.dirname(args.train_file))
    os.makedirs(f"ckpts/{args.save_dataet_name}", exist_ok=True)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Calculate progress for scheduling
            phase_progress = global_step / total_iterations
            
            # Determine current phase
            iteration_in_cycle = global_step % (args.teacher_steps + args.student_steps)
            if iteration_in_cycle < args.teacher_steps:
                # ===== TEACHER PHASE =====
                metrics = teacher_phase(
                    teacher=teacher,
                    teacher_ref=teacher_ref,
                    student=student,
                    tokenizer=tokenizer,
                    batch=batch,
                    args=args,
                    optimizer=teacher_optimizer,
                    accelerator=accelerator,
                )
                phase_name = "teacher"
            else:
                # ===== STUDENT PHASE =====
                metrics = student_phase(
                    student=student,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    batch=batch,
                    args=args,
                    optimizer=student_optimizer,
                    accelerator=accelerator,
                    phase_progress=phase_progress,
                )
                phase_name = "student"
            
            global_step += 1
            progress_bar.update(1)
            
            if accelerator.is_main_process:
                metrics["phase"] = 1 if phase_name == "teacher" else 0
                metrics["global_step"] = global_step
                accelerator.log(metrics, step=global_step)
            
            # Checkpointing
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    print(f"Saving checkpoint at step {global_step}...")
                    
                    # Save Student
                    student_ckpt_dir = f"ckpts/{args.save_dataet_name}/student_model_{args.run_name}_step_{global_step}"
                    os.makedirs(student_ckpt_dir, exist_ok=True)
                    accelerator.unwrap_model(student).save_pretrained(student_ckpt_dir)
                    tokenizer.save_pretrained(student_ckpt_dir)
                    
                    # Save Teacher
                    # teacher_ckpt_dir = f"ckpts/{args.save_dataet_name}/teacher_model_{args.run_name}_step_{global_step}"
                    # os.makedirs(teacher_ckpt_dir, exist_ok=True)
                    # accelerator.unwrap_model(teacher).save_pretrained(teacher_ckpt_dir)
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                val_ce_loss = evaluate_student(student, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    print(f" Step {global_step} ({phase_name}) | Val CE Loss: {val_ce_loss:.4f}")
                    accelerator.log({"val/ce_loss": val_ce_loss}, step=global_step)

    if accelerator.is_main_process:
        output_dir = f"ckpts/{args.save_dataet_name}/student_model_{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save student
        accelerator.unwrap_model(student).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save teacher
        # teacher_dir = f"ckpts/{args.save_dataet_name}/teacher_model_{args.run_name}"
        # os.makedirs(teacher_dir, exist_ok=True)
        # accelerator.unwrap_model(teacher).save_pretrained(teacher_dir)
        
        if args.use_wandb: 
            accelerator.end_training()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--student_model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--response_column", type=str, default="response")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    
    # Two-phase training args
    parser.add_argument("--teacher_steps", type=int, default=0, help="Number of teacher phase steps per cycle")
    parser.add_argument("--student_steps", type=int, default=10, help="Number of student phase steps per cycle")
    
    # Teacher phase args
    parser.add_argument("--teacher_lr", type=float, default=5e-5, help="Learning rate for teacher")
    parser.add_argument("--num_teacher_samples", type=int, default=4, help="Number of trajectories per prompt")
    parser.add_argument("--teacher_temperature", type=float, default=1.0, help="Temperature for teacher generation")
    parser.add_argument("--teacher_kl_coef", type=float, default=0.1, help="KL coefficient for teacher-ref")
    parser.add_argument("--teacher_student_kl_coef", type=float, default=0.05, help="KL coefficient for teacher-student")
    parser.add_argument("--difficulty_weight", type=float, default=0.5, help="Weight for difficulty reward")
    
    # Student phase args
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for student")
    parser.add_argument("--lambda_on_policy", type=float, default=0.3, help="Max probability for student on-policy at end")
    parser.add_argument("--lambda_teacher_gen", type=float, default=0.3, help="Max probability for teacher-generated at end")
    parser.add_argument("--teacher_gen_temperature", type=float, default=0.8, help="Temperature for teacher generation in distillation")
    
    # Common args
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="two-phase-distillation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X steps. 0 means save only at the end.")
    parser.add_argument("--use_unsloth", type=str2bool, default=True)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key): 
                    setattr(args, key, value)

    # Type conversions
    for attr in ['lr', 'teacher_lr', 'beta', 'temperature', 'lambda_on_policy', 
                'lambda_teacher_gen', 'teacher_gen_temperature',  # Add these
                'teacher_temperature', 'teacher_kl_coef', 'teacher_student_kl_coef', 'difficulty_weight']:
        if hasattr(args, attr) and isinstance(getattr(args, attr), str):
            setattr(args, attr, float(getattr(args, attr)))
    
    for attr in ['use_wandb', 'use_unsloth', 'load_in_4bit']:
        if hasattr(args, attr) and isinstance(getattr(args, attr), str):
            setattr(args, attr, str2bool(getattr(args, attr)))

    if args.mixed_precision == "bf16":
        args.dt = torch.bfloat16
    elif args.mixed_precision == "fp16":
        args.dt = torch.float16
    else:
        args.dt = torch.float32

    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)