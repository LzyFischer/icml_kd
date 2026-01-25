import argparse
import os
import random
import warnings
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from contextlib import nullcontext

# 1. Import wandb
import wandb
import json

# Unsloth imports
from unsloth import FastLanguageModel

# Evaluation utilities (dataset registry and helpers)
from eval_vllm import (
    DATASET_REGISTRY,
    extract_reference,
)

warnings.filterwarnings("ignore")


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_full_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Memory‑efficient full KL divergence on a per‑token basis.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    kl_values = torch.zeros(batch_size, seq_len, device=device)

    # Process in manageable chunks to limit memory consumption
    chunk_size = 512  # adjust this based on available memory
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        s_chunk = student_logits[:, start_idx:end_idx, :]
        t_chunk = teacher_logits[:, start_idx:end_idx, :]
        # Softmax with temperature
        s_probs = F.softmax(s_chunk / temperature, dim=-1)
        t_probs = F.softmax(t_chunk / temperature, dim=-1)
        # KL divergence: KL(t || s)
        kl_chunk = (t_probs * (t_probs.log() - s_probs.log())).sum(dim=-1)
        kl_values[:, start_idx:end_idx] = kl_chunk
        # Release intermediate tensors to conserve memory
        del s_chunk, t_chunk, s_probs, t_probs, kl_chunk

    return kl_values


def build_chat_dataset(
    raw_ds: Dataset,
    dataset_config: Dict,
    eval_type: str,
) -> Dataset:
    """
    Transform a raw dataset into a list of chat conversations.
    """
    chats: List[List[Dict[str, str]]] = []
    formatter = dataset_config.get("formatter")
    for row in raw_ds:
        try:
            # The formatter returns a tuple (prompt_text, answer_text) but we
            # ignore the answer_text here and rely on extract_reference for a
            # ground‑truth answer.
            prompt_text, _ = formatter(row)
            reference_text = extract_reference(row, eval_type)
            # Construct a two‑turn chat: user then assistant
            chat = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": reference_text},
            ]
            chats.append(chat)
        except Exception:
            # Skip problematic rows
            continue
    return Dataset.from_dict({"chat": chats})


def collate_batch(batch, tokenizer, max_length: int = 1024):
    """
    Collate a list of chat samples into a batch of tensors for training.
    """
    formatted_texts: List[str] = []
    for example in batch:
        messages = example["chat"]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(formatted)
    # Tokenize the formatted texts with padding and truncation
    encodings = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    # Create labels: copy of input_ids with padding positions masked out
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file that overrides command‑line arguments.")
    parser.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct", help="Identifier or path for the teacher model.")
    parser.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Identifier or path for the frozen student model.")
    parser.add_argument("--train_file", type=str, default="./data/date/train.jsonl", help="Path to the training dataset file (JSON or JSONL).")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional name of the dataset used to look up formatters in the registry.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Truncate the training dataset to this many samples.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_epochs.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="Weight applied to the KL divergence term in the loss.")
    parser.add_argument("--kl_temperature", type=float, default=1.0, help="Temperature used when computing KL divergence.")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of the LoRA adapters.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for the LoRA adapters.")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Whether to load models in 4‑bit precision using bitsandbytes.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to log metrics to Weights & Biases.")
    parser.add_argument("--run_name", type=str, default="sft_with_kl", help="Name of the run used for output directory naming.")
    parser.add_argument("--save_steps", type=int, default=20, help="Interval (in steps) at which to save checkpoints. 0 disables intermediate saving.")
    args = parser.parse_args()
    # Load config overrides from YAML if provided
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 2. Initialize WandB
    # -------------------------------------------------------------------------
    if args.use_wandb:
        wandb.init(
            project="kl-divergence-training",  # You can change this project name
            name=args.run_name,
            config=vars(args),
        )

    # -------------------------------------------------------------------------
    # Dataset loading and preprocessing
    # -------------------------------------------------------------------------
    dataset_name = args.dataset_name
    if not dataset_name and args.train_file:
        try:
            dataset_name = os.path.basename(os.path.dirname(args.train_file))
        except Exception:
            dataset_name = "default"
    
    dataset_config = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY.get("default"))
    if dataset_name not in DATASET_REGISTRY:
        for k in DATASET_REGISTRY:
            if dataset_name and dataset_name in k:
                dataset_config = DATASET_REGISTRY[k]
                break
    eval_type = dataset_config.get("type", "text")
    
    print(f"Loading training data from: {args.train_file}")
    try:
        if args.train_file.endswith(".jsonl"):
            ds = load_dataset("json", data_files={"train": args.train_file}, split="train")
        else:
            ds = load_dataset("json", data_files={"train": args.train_file}, field="instances", split="train")
    except Exception as e:
        # ATTEMPT 2: Fallback to Python JSON loading (Schema-agnostic)
        print(f"Warning: load_dataset failed ({e}). Falling back to standard Python json/jsonl load.")
        data = []
        with open(args.train_file, "r", encoding="utf-8") as f:
            if args.train_file.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                try:
                    full_data = json.load(f)
                    if isinstance(full_data, list):
                        data = full_data
                    elif isinstance(full_data, dict) and "instances" in full_data:
                        data = full_data["instances"]
                    else:
                        # Try flat dict
                        data = [full_data]
                except Exception:
                    data = []
        ds = data  # ds is now a simple list of dicts
    if args.max_train_samples:
        ds = ds[: min(len(ds), args.max_train_samples)]
    
    train_ds = build_chat_dataset(ds, dataset_config, eval_type)
    print(f"Prepared {len(train_ds)} chat examples for training.")

    # -------------------------------------------------------------------------
    # Model and tokenizer loading
    # -------------------------------------------------------------------------
    print(f"Loading teacher model: {args.teacher_model}")
    is_lora_checkpoint = os.path.exists(os.path.join(args.teacher_model, "adapter_config.json"))
    teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=0.25,
    )
    if not is_lora_checkpoint:
        teacher_model = FastLanguageModel.get_peft_model(
            teacher_model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        FastLanguageModel.for_training(teacher_model)
    
    print(f"Loading student model: {args.student_model}")
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        dtype=None,
        gpu_memory_utilization=0.05,
    )
    FastLanguageModel.for_inference(student_model)

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, teacher_tokenizer, args.max_length),
    )

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    teacher_model.train()
    trainable_parameters = [p for p in teacher_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)

    total_steps = len(train_dataloader) * args.num_epochs
    step_count = 0
    print("Starting supervised fine‑tuning with KL divergence...")

    is_gemma = "gemma" in student_model.config._name_or_path
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if is_gemma else nullcontext()
    with autocast_ctx:
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass teacher
                outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = outputs.logits
                
                # Cross Entropy Loss
                ce_loss = F.cross_entropy(
                    teacher_logits.view(-1, teacher_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                # Forward pass student (no grad)
                with torch.no_grad():
                    student_outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    student_logits = student_outputs.logits
                
                # KL Divergence Loss
                kl_per_token = compute_full_kl_efficient(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=args.kl_temperature,
                )
                kl_mask = (labels != -100).to(kl_per_token.dtype)
                kl_loss = (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp(min=1)
                
                # Combined Loss
                loss = ce_loss + args.kl_weight * kl_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step_count += 1

                if args.max_steps > 0 and step_count >= args.max_steps:
                    print(f"Reached max_steps of {args.max_steps}. Ending training.")
                    break
                
                # 3. Log metrics to WandB
                if args.use_wandb:
                    wandb.log({
                        "train/total_loss": loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/epoch": epoch + (batch_idx / len(train_dataloader)),
                        "train/step": step_count,
                    })

                if args.save_steps > 0 and step_count % args.save_steps == 0:
                    save_dir = os.path.join("ckpts", dataset_name, args.run_name)
                    os.makedirs(save_dir, exist_ok=True)
                    ckpt_path = os.path.join(save_dir, f"step_{step_count}_lora")
                    teacher_model.save_lora(ckpt_path)
                    print(f"Saved checkpoint at step {step_count} to {ckpt_path}")
            
            print(f"Completed epoch {epoch + 1}")

    # -------------------------------------------------------------------------
    # Save final
    # -------------------------------------------------------------------------
    final_dir = os.path.join("ckpts", dataset_name, args.run_name)
    os.makedirs(final_dir, exist_ok=True)
    teacher_model.save_lora(os.path.join(final_dir, "final_lora"))
    print(f"Training complete! LoRA adapters saved to {final_dir}/final_lora")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()