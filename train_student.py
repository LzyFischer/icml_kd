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
    str2bool, ensure_dtype_match, generalized_jsd_loss,
    generate_on_policy, evaluate_student, HAS_UNSLOTH,
    forward_kl_div_loss, reverse_kl_div_loss, compute_sft_loss
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

warnings.filterwarnings("ignore", category=UserWarning, module="unsloth.kernels.utils")
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

# -----------------------------------------------------------------------------
# Student Phase Training
# -----------------------------------------------------------------------------

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
    Student phase: Apply distillation based on the explicitly selected mode.
    """
    student.train()
    teacher.eval()
    
    mode = args.student_mode
    prompts = batch["prompts"]

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    # Compute Distillation Loss
    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
    if args.loss_type == "sft":
        # Pure SFT logic using the new utility function
        loss = compute_sft_loss(student_outputs.logits, labels)
    else:
        # Distillation modes require teacher logits
        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        
        # Prepare inputs for distillation functions in utils.py
        mask = (labels != -100).float()[:, 1:]
        s_logits_distill = student_outputs.logits[:, :-1, :]
        t_logits_distill = teacher_outputs.logits[:, :-1, :]

        if args.loss_type == "forward":
            loss = forward_kl_div_loss(s_logits_distill, t_logits_distill, mask=mask, temperature=args.temperature)
        elif args.loss_type == "reverse":
            # generalized_jsd_loss in utils.py implements Reverse KL
            loss = reverse_kl_div_loss(s_logits_distill, t_logits_distill, mask=mask, temperature=args.temperature)
        elif args.loss_type == "generalized":
            # generalized_jsd_loss_ in utils.py implements full JSD
            loss = generalized_jsd_loss(s_logits_distill, t_logits_distill, mask=mask, beta=args.beta, temperature=args.temperature)

    # Update student
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    
    return {f"student/loss_{mode}": loss.item(), "student/active_mode": mode, "student/loss_type": args.loss_type}

def train_student(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_unsloth and not HAS_UNSLOTH:
        raise ImportError("Unsloth requested but not installed.")

    if args.run_name is None:
        s_name = args.student_model.replace("/", "_")
        t_name = args.teacher_model.replace("/", "_")
        args.run_name = f"{args.prefix}_{args.student_mode}_t_{t_name}_s_{s_name}"

    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None, mixed_precision=args.mixed_precision)
    if args.use_wandb:
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": {"name": args.run_name, "id": None}})

    if accelerator.is_main_process:
        print(f"Student Training Phase")
        print(f"Teacher: {args.teacher_model} | Student: {args.student_model}")

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    if args.use_unsloth:
        is_lora_checkpoint = os.path.exists(os.path.join(args.student_model, "adapter_config.json"))
        
        student, tokenizer = FastModel.from_pretrained(
            model_name=args.student_model,
            max_seq_length=args.max_length,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dt,
        )
        
        if not is_lora_checkpoint:
            # Iter 1: Base model loaded, initialize new LoRA adapters
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
        else:
            # Iter 2+: LoRA checkpoint loaded (Unsloth auto-loaded adapters)
            if accelerator.is_main_process:
                print(f"Resuming training from LoRA checkpoint: {args.student_model}")
            
            # Critical: Unsloth defaults to inference mode when loading adapters.
            # We must explicitly enable training gradients for LoRA layers.
            FastModel.for_training(student)

        # Load teacher (can be a checkpoint from teacher training)
        teacher, _ = FastModel.from_pretrained(
            model_name=args.teacher_model,
            max_seq_length=args.max_length,
            load_in_4bit=args.load_in_4bit, 
            dtype=args.dt,
        )
        FastModel.for_inference(teacher)
        
        student.config.use_cache = True 
        teacher.config.use_cache = True

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True, padding_side="left")
        
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        teacher.eval()
        teacher.requires_grad_(False)
        
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
    # Optimizer
    # -------------------------------------------------------------------------
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_source = args.train_file if args.train_file else args.dataset_name
    val_source = args.val_file if args.val_file else args.dataset_name
    val_split = "train" if args.val_file else "test"

    if args.student_mode == "teacher_generated":
        train_source = os.path.join(os.path.dirname(train_source), args.teacher_model.replace("/", "_"), os.path.basename(train_source))
        val_source = os.path.join(os.path.dirname(val_source), args.teacher_model.replace("/", "_"), os.path.basename(val_source))

    train_examples = load_data_source(train_source, split="train", prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_train_samples)
    val_examples = load_data_source(val_source, split=val_split, prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_val_samples)

    collate = collate_fn_builder(tokenizer, args.max_length)
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    student, teacher, student_optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        student, teacher, student_optimizer, train_dataloader, val_dataloader
    )

    global_step = 0
    total_iterations = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm.tqdm(range(total_iterations), disable=not accelerator.is_main_process)

    args.save_dataset_name = args.dataset_name.replace("/", "_") if not args.val_file or "rank" in args.val_file else os.path.basename(os.path.dirname(args.val_file))
    os.makedirs(f"ckpts/{args.save_dataset_name}", exist_ok=True)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Calculate progress for scheduling
            phase_progress = global_step / total_iterations
            
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
            
            global_step += 1
            progress_bar.update(1)
            
            if accelerator.is_main_process:
                metrics["global_step"] = global_step
                accelerator.log(metrics, step=global_step)
            
            # Checkpointing
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    print(f"Saving checkpoint at step {global_step}...")
                    
                    student_ckpt_dir = f"ckpts/{args.save_dataset_name}/student_model_{args.run_name}_step_{global_step}"
                    os.makedirs(student_ckpt_dir, exist_ok=True)
                    accelerator.unwrap_model(student).save_pretrained(student_ckpt_dir)
                    tokenizer.save_pretrained(student_ckpt_dir)
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                val_ce_loss = evaluate_student(student, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    print(f" Step {global_step} | Val CE Loss: {val_ce_loss:.4f}")
                    accelerator.log({"val/ce_loss": val_ce_loss}, step=global_step)

    if accelerator.is_main_process:
        output_dir = f"ckpts/{args.save_dataset_name}/student_model_{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        accelerator.unwrap_model(student).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
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
    
    # Student phase args
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for student")
    parser.add_argument("--loss_type", type=str, default="forward", choices=["sft", "forward", "reverse", "generalized"])
    parser.add_argument("--student_mode", type=str, default="supervised", choices=["supervised", "on_policy", "teacher_generated"])
    parser.add_argument("--teacher_gen_temperature", type=float, default=0.8, help="Temperature for teacher generation in distillation")
    
    # Common args
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="student-phase-distillation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X steps. 0 means save only at the end.")
    parser.add_argument("--use_unsloth", type=str2bool, default=True)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key): 
                    setattr(args, key, value)

    # Type conversions
    for attr in ['lr', 'beta', 'temperature', 'lambda_on_policy', 
                'lambda_teacher_gen', 'teacher_gen_temperature']:
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
    train_student(args)