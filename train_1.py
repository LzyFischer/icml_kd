import argparse
import math
import os
import random
import yaml
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from accelerate import Accelerator
from datasets import load_dataset
import tqdm
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------

def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the generalized Jensen–Shannon divergence."""
    # Temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Convert logits to log‑probabilities
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    # Compute mixture distribution M
    log_beta = math.log(beta)
    log_one_minus_beta = math.log(1.0 - beta)
    stacked = torch.stack(
        [log_beta + teacher_log_probs, log_one_minus_beta + student_log_probs],
        dim=0,
    )
    mixture_log_probs = torch.logsumexp(stacked, dim=0)

    # KL divergences
    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs,
        student_log_probs,
        reduction="none",
        log_target=True,
    )
    jsd_per_token = beta * kl_teacher + (1.0 - beta) * kl_student
    jsd_per_position = jsd_per_token.sum(dim=-1)

    if mask is not None:
        jsd_per_position = jsd_per_position * mask
        total_loss = jsd_per_position.sum()
        num_positions = mask.sum().clamp(min=1)
    else:
        total_loss = jsd_per_position.sum()
        num_positions = jsd_per_position.numel()

    return total_loss / num_positions

# -----------------------------------------------------------------------------
# Data Handling (Updated for Local Files)
# -----------------------------------------------------------------------------

@dataclass
class Example:
    prompt: str
    response: str

def load_data_source(
    path_or_name: str,
    split: str = "train",
    prompt_col: str = "instruction",
    resp_col: str = "response",
    limit: Optional[int] = None
) -> List[Example]:
    """
    Load dataset from HF Hub OR local JSON file.
    Handles the {"instances": [...]} structure if local.
    """
    print(f"Loading data from: {path_or_name}...")
    
    # Check if it is a local file
    if os.path.isfile(path_or_name):
        # Local JSON loading
        # The 'field' argument is used if the JSON has a root key like "instances"
        ds = load_dataset("json", data_files={split: path_or_name}, field="instances", split=split)
    else:
        # Hugging Face Hub loading
        ds = load_dataset(path_or_name, split=split)

    if limit:
        ds = ds.select(range(min(len(ds), limit)))

    examples = []
    for row in ds:
        if prompt_col in row and resp_col in row:
            examples.append(Example(prompt=row[prompt_col], response=row[resp_col]))
    
    print(f"Loaded {len(examples)} examples.")
    return examples

def chat_template_pair(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a prompt/response pair into input IDs, attention mask and labels."""
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        return_dict=True,
        truncate=True,
        max_length=max_length,
    )

    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        return_tensors="pt",
        return_dict=True,
        truncate=True,
        max_length=max_length,
    )
    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]

    labels = input_ids.clone()
    user_length = user_only["input_ids"].size(-1)
    
    if user_length < labels.size(0):
        labels[: user_length] = -100
    else:
        labels[:] = -100 

    return input_ids, attention_mask, labels

def collate_fn_builder(
    tokenizer: PreTrainedTokenizer, max_length: int
) -> Callable[[List[Example]], Dict[str, torch.Tensor]]:
    def collate(batch: List[Example]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for ex in batch:
            ids, attn, lbls = chat_template_pair(
                tokenizer, ex.prompt, ex.response, max_length
            )
            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(lbls)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate

def generate_on_policy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
) -> List[str]:
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in prompts],
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_attention_mask=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=top_k > 0,
        top_k=top_k if top_k > 0 else None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    responses: List[str] = []
    for text, prompt in zip(decoded, prompts):
        # Heuristic: Remove the user prompt from the generated text
        # (This varies slightly by chat template, this is a generic fallback)
        if prompt in text:
            reply = text.split(prompt)[-1].strip()
        else:
            reply = text
        responses.append(reply)
    return responses

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_student(
    model: PreTrainedModel,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        # Gather loss across GPUs if needed
        total_loss += accelerator.gather(loss).mean().item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Configurable Run Name for WandB
    if args.run_name is None:
        # Auto-generate a descriptive name if not provided
        s_name = args.student_model.split("/")[-1]
        t_name = args.teacher_model.split("/")[-1]
        args.run_name = f"distill_{s_name}_from_{t_name}_lr{args.lr}_beta{args.beta}"

    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    if args.use_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name, "id": None}} # id=None ensures new run
        )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print(f"Teacher: {args.teacher_model}")
        print(f"Student: {args.student_model}")

    # Load Models
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    student.resize_token_embeddings(len(tokenizer))
    
    teacher.eval()
    teacher.requires_grad_(False)

    if args.use_lora:
        if accelerator.is_main_process:
            print(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}, targets={args.lora_target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none", # 通常设为 none，也可以设为 all
        )
        student = get_peft_model(student, peft_config)
        
        # 打印可训练参数量，确认 LoRA 生效
        if accelerator.is_main_process:
            student.print_trainable_parameters()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # 2. Load Datasets (Support for Local Files)
    # If train_file provided, use it. Else use dataset_name.
    train_source = args.train_file if args.train_file else args.dataset_name
    val_source = args.val_file if args.val_file else args.dataset_name

    train_examples = load_data_source(
        train_source, 
        split=args.train_split,
        prompt_col=args.prompt_column, 
        resp_col=args.response_column,
        limit=args.max_train_samples
    )
    
    val_examples = load_data_source(
        val_source, 
        split=args.val_split,
        prompt_col=args.prompt_column, 
        resp_col=args.response_column,
        limit=args.max_val_samples
    )

    collate = collate_fn_builder(tokenizer, args.max_length)
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    student, teacher, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        student, teacher, optimizer, train_dataloader, val_dataloader
    )

    global_step = 0
    total_steps = args.num_epochs * len(train_dataloader)
    
    if accelerator.is_main_process:
        print(f"Starting training: {args.num_epochs} epochs, {total_steps} steps.")

    progress_bar = tqdm.tqdm(range(total_steps), disable=not accelerator.is_main_process)

    for epoch in range(args.num_epochs):
        student.train()
        for batch in train_dataloader:
            with accelerator.accumulate(student):
                loss_dict = {}
                
                # --- On-Policy Branch ---
                if random.random() < args.lambda_on_policy:
                    # Simple prompt sampling:
                    indices = random.sample(range(len(train_examples)), args.batch_size)
                    prompts = [train_examples[i].prompt for i in indices]

                    with torch.no_grad():
                        student_responses = generate_on_policy(
                            accelerator.unwrap_model(student),
                            tokenizer,
                            prompts,
                            max_new_tokens=args.max_new_tokens,
                            temperature=1.0,
                        )
                    
                    new_examples = [Example(prompt=p, response=r) for p, r in zip(prompts, student_responses)]
                    new_batch = collate(new_examples)
                    
                    input_ids = new_batch["input_ids"].to(student.device)
                    attention_mask = new_batch["attention_mask"].to(student.device)
                    labels = new_batch["labels"].to(student.device)

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
                    loss_dict["loss/on_policy"] = loss.item()
                    loss_dict["loss/supervised"] = 0.0

                # --- Supervised Branch ---
                else:
                    student_outputs = student(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"]
                    )
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"]
                        )

                    mask = (batch["labels"] != -100).float()
                    loss = generalized_jsd_loss(
                        student_outputs.logits[:, :-1, :],
                        teacher_outputs.logits[:, :-1, :],
                        mask=mask[:, 1:],
                        beta=args.beta,
                        temperature=args.temperature,
                    )
                    loss_dict["loss/supervised"] = loss.item()
                    loss_dict["loss/on_policy"] = 0.0

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                # 3. Enhanced Logging
                if accelerator.is_main_process:
                    loss_dict["loss/total"] = loss.item()
                    loss_dict["lr"] = optimizer.param_groups[0]["lr"]
                    loss_dict["epoch"] = epoch + (global_step / len(train_dataloader))
                    accelerator.log(loss_dict, step=global_step)

                # Periodic Evaluation
                if global_step % args.eval_steps == 0:
                    val_ce_loss = evaluate_student(student, val_dataloader, accelerator)
                    if accelerator.is_main_process:
                        print(f" Step {global_step} | Val CE Loss: {val_ce_loss:.4f}")
                        accelerator.log({"val/ce_loss": val_ce_loss}, step=global_step)

    # Save
    # Save
    if accelerator.is_main_process:
        output_dir = f"distilled_model_{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取未被 DDP/Accelerate 包装的原始模型
        unwrapped_model = accelerator.unwrap_model(student)
        
        # 保存模型
        # PEFT 的 save_pretrained 会自动只保存 adapter_model.bin
        unwrapped_model.save_pretrained(output_dir)
        
        # 保存 tokenizer
        tokenizer.save_pretrained(output_dir)
        
        print(f"LoRA adapters saved to {output_dir}")

        if args.use_wandb:
            accelerator.end_training()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    # Models
    parser.add_argument("--teacher_model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--student_model", type=str, default="google/gemma-2b-it")
    
    # Data - Updated for Local Support
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca", help="HF Dataset Name (Fallback)")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="train")
    parser.add_argument("--train_file", type=str, default=None, help="Path to local training JSON")
    parser.add_argument("--val_file", type=str, default=None, help="Path to local validation JSON")
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--response_column", type=str, default="response", help="Default matches gsm_100 json")
    
    # Hyperparams
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--lambda_on_policy", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="distillation-project")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)

    parser.add_argument("--use_lora", action="store_true", help="是否启用 LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                        default=["q_proj", "v_proj"], 
                        help="应用 LoRA 的模块，例如 q_proj v_proj k_proj o_proj")

    args = parser.parse_args()

    # YAML Override Logic
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}...")
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    print(f"  Override: {key} = {value}")

    # 1. Force scientific notation strings to float
    # This addresses the issue where '1e-5' from YAML is treated as a string
    if isinstance(args.lr, str):
        args.lr = float(args.lr)
    if isinstance(args.beta, str):
        args.beta = float(args.beta)
    if isinstance(args.temperature, str):
        args.temperature = float(args.temperature)
    if isinstance(args.lambda_on_policy, str):
        args.lambda_on_policy = float(args.lambda_on_policy)

    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)