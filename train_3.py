# Try importing Unsloth
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

import argparse
import math
import os
import random
import yaml
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
import pdb



import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="unsloth.kernels.utils")
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def ensure_dtype_match(model, target_dtype=torch.bfloat16):
    """Ensure all model parameters have the same dtype."""
    for name, param in model.named_parameters():
        if param.dtype != target_dtype and param.requires_grad:
            param.data = param.data.to(target_dtype)
    return model

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the generalized Jensen–Shannon divergence."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Handle vocabulary size mismatch by taking the minimum
    min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :min_vocab_size]
    teacher_logits = teacher_logits[..., :min_vocab_size]

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    log_beta = math.log(beta)
    log_one_minus_beta = math.log(1.0 - beta)
    stacked = torch.stack(
        [log_beta + teacher_log_probs, log_one_minus_beta + student_log_probs],
        dim=0,
    )
    mixture_log_probs = torch.logsumexp(stacked, dim=0)

    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs, student_log_probs, reduction="none", log_target=True
    )
    jsd_per_token = beta * kl_teacher + (1.0 - beta) * kl_student
    jsd_per_position = jsd_per_token.sum(dim=-1)

    if mask is not None:
        jsd_per_position = jsd_per_position * mask

    if reduction == "none":
        return jsd_per_position
    elif reduction == "mean":
        if mask is not None:
            return jsd_per_position.sum() / mask.sum().clamp(min=1)
        else:
            return jsd_per_position.mean()
    elif reduction == "sum":
        return jsd_per_position.sum()

def compute_grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    kl_coef: float = 0.1,
    student_kl_coef: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO loss with KL penalties from both reference teacher and student.
    
    Args:
        logprobs: Current teacher log probabilities [batch, seq_len]
        ref_logprobs: Reference teacher log probabilities [batch, seq_len]
        student_logprobs: Student log probabilities [batch, seq_len]
        rewards: Per-sequence rewards [batch]
        mask: Valid token mask [batch, seq_len]
        kl_coef: KL coefficient for teacher-ref divergence
        student_kl_coef: KL coefficient for teacher-student divergence
    """
    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Compute log ratios
    log_ratio = logprobs - ref_logprobs
    kl_ref = log_ratio
    
    # KL with student
    kl_student = logprobs - student_logprobs
    
    # Masked sum for per-sequence values
    kl_ref_per_seq = (kl_ref * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    kl_student_per_seq = (kl_student * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    log_ratio_per_seq = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    # GRPO objective: maximize rewards while minimizing KL divergence
    loss = -(rewards * log_ratio_per_seq - kl_coef * kl_ref_per_seq - student_kl_coef * kl_student_per_seq).mean()
    
    metrics = {
        "teacher/reward_mean": rewards.mean().item(),
        "teacher/kl_ref": kl_ref_per_seq.mean().item(),
        "teacher/kl_student": kl_student_per_seq.mean().item(),
    }
    
    return loss, metrics

# -----------------------------------------------------------------------------
# Data Handling
# -----------------------------------------------------------------------------

@dataclass
class Example:
    prompt: str
    response: str

def format_arc_challenge(row) -> Tuple[str, str]:
    """Formats ARC-Challenge dictionary choices into a text prompt."""
    q = row.get("question", "")
    choices = row.get("choices", {})
    
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])])
    else:
        formatted_choices = str(choices)
        
    prompt = f"Question: {q}\nChoices:\n{formatted_choices}\nAnswer:"
    return prompt, row.get("response", "")

def format_anli(row) -> Tuple[str, str]:
    """Formats ANLI premise and hypothesis."""
    prompt = (
        f"Premise: {row.get('premise', '')}\n"
        f"Hypothesis: {row.get('hypothesis', '')}\n\n"
        f"Does the premise entail the hypothesis?"
    )
    return prompt, row.get("response", "")

def format_math(row) -> Tuple[str, str]:
    """Formats MATH dataset."""
    q = row.get("question", "")
    return f"Question: {q}\nAnswer:", row.get("response", "")

def format_strategy_qa(row) -> Tuple[str, str]:
    """Formats StrategyQA."""
    q = row.get("question", "")
    return f"Question: {q}\nAnswer:", row.get("response", "")

def load_data_source(path_or_name: str, split: str = "train", prompt_col: str = "instruction", resp_col: str = "response", limit: Optional[int] = None) -> List[Example]:
    print(f"Loading data from: {path_or_name}...")
    
    if os.path.isfile(path_or_name):
        if path_or_name.endswith(".jsonl"):
            ds = load_dataset("json", data_files={split: path_or_name}, split=split)
        else:
            try:
                ds = load_dataset("json", data_files={split: path_or_name}, field="instances", split=split)
            except Exception:
                print("Could not load with field='instances', trying flat JSON...")
                ds = load_dataset("json", data_files={split: path_or_name}, split=split)
    else:
        ds = load_dataset(path_or_name, split=split)

    if limit:
        ds = ds.select(range(min(len(ds), limit)))

    lower_name = path_or_name.lower()
    
    if "arc" in lower_name:
        formatter = format_arc_challenge
        print("-> Detected ARC Challenge format.")
    elif "anli" in lower_name:
        formatter = format_anli
        print("-> Detected ANLI format.")
    elif "math" in lower_name:
        formatter = format_math
        print("-> Detected MATH format.")
    elif "strategy" in lower_name or "qa" in lower_name:
        formatter = format_strategy_qa
        print("-> Detected StrategyQA format.")
    else:
        def generic_formatter(row):
            return row.get(prompt_col, ""), row.get(resp_col, "")
        formatter = generic_formatter
        print(f"-> Using generic format: {prompt_col} -> {resp_col}")

    examples = []
    for row in ds:
        try:
            p, r = formatter(row)
            if p and r:
                examples.append(Example(prompt=p, response=r))
        except Exception:
            continue
    
    print(f"Loaded {len(examples)} examples.")
    return examples

def chat_template_pair(tokenizer: PreTrainedTokenizer, prompt: str, response: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    user_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", return_dict=True)
    full = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], return_tensors="pt", return_dict=True)
    
    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]

    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    labels = input_ids.clone()
    user_length = user_only["input_ids"].size(-1)
    
    mask_len = min(user_length, labels.size(0))
    labels[:mask_len] = -100

    return input_ids, attention_mask, labels

def collate_fn_builder(tokenizer: PreTrainedTokenizer, max_length: int) -> Callable[[List[Example]], Dict[str, Union[torch.Tensor, List[str]]]]:
    def collate(batch: List[Example]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        input_ids_list, attention_mask_list, labels_list = [], [], []
        raw_prompts = [] 
        raw_responses = []
        
        for ex in batch:
            ids, attn, lbls = chat_template_pair(tokenizer, ex.prompt, ex.response, max_length)
            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(lbls)
            raw_prompts.append(ex.prompt)
            raw_responses.append(ex.response)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels,
            "prompts": raw_prompts,
            "responses": raw_responses,
        }
    return collate

def generate_on_policy(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str], *, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0, num_samples: int = 1, use_unsloth: bool = False) -> List[List[str]]:
    """Generate multiple samples per prompt."""
    tokenizer.padding_side = "left"
    
    # Expand prompts for multiple samples
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * num_samples)
    
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in expanded_prompts],
        return_tensors="pt", 
        return_dict=True, 
        padding=True,
        return_attention_mask=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    if use_unsloth and HAS_UNSLOTH:
        FastModel.for_inference(model) 
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=True, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        FastModel.for_training(model) 
    else:
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=True, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Group responses by original prompt
    all_responses = []
    for i, prompt in enumerate(prompts):
        prompt_responses = []
        for j in range(num_samples):
            idx = i * num_samples + j
            text = decoded[idx]
            if prompt in text:
                reply = text.split(prompt)[-1].strip()
            else:
                reply = text
            prompt_responses.append(reply)
        all_responses.append(prompt_responses)
    
    return all_responses

def compute_verifiable_reward(generated: str, ground_truth: str) -> float:
    """Simple binary reward: 1 if match, 0 otherwise."""
    gen_normalized = generated.strip().lower()
    gt_normalized = ground_truth.strip().lower()
    return 1.0 if gen_normalized == gt_normalized else 0.0

@torch.no_grad()
def compute_difficulty_reward(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute difficulty reward based on JSD between teacher and student.
    Higher JSD = more difficult = lower reward.
    Returns normalized rewards (higher is easier).
    """
    jsd_per_token = generalized_jsd_loss(
        student_logits, teacher_logits, mask=mask,
        beta=beta, temperature=temperature, reduction="none"
    )
    
    # Average over sequence
    jsd_per_seq = jsd_per_token.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    # Convert to reward: easier (low JSD) = high reward
    difficulty_reward = -jsd_per_seq
    
    # Normalize across batch
    difficulty_reward = (difficulty_reward - difficulty_reward.mean()) / (difficulty_reward.std() + 1e-8)
    
    return difficulty_reward

@torch.no_grad()
def evaluate_student(model: PreTrainedModel, dataloader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        total_loss += accelerator.gather(outputs.loss).mean().item()
        num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

# -----------------------------------------------------------------------------
# Two-Phase Training Loop
# -----------------------------------------------------------------------------

def teacher_phase(
    teacher: PreTrainedModel,
    teacher_ref: PreTrainedModel,
    student: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Dict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """
    Teacher phase: Apply GRPO to teacher model.
    1. Generate trajectories from teacher
    2. Compute verifiable reward
    3. Compute difficulty reward using JSD with student
    4. Apply GRPO with KL penalties
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
    batch: Dict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    phase_progress: float,
) -> Dict[str, float]:
    """
    Student phase: Apply mixed distillation.
    1. With scheduled probability, use on-policy or supervised distillation
    2. Use JSD loss for distillation
    """
    student.train()
    teacher.eval()
    
    # Scheduled mixing: increase on-policy over time
    on_policy_prob = args.lambda_on_policy
    
    if random.random() < on_policy_prob:
        # On-policy distillation
        prompts = batch["prompts"]
        
        with torch.no_grad():
            student_responses = generate_on_policy(
                accelerator.unwrap_model(student),
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                num_samples=1,
                use_unsloth=args.use_unsloth
            )
            student_responses = [r[0] for r in student_responses]  # Take first sample
        
        new_examples = [Example(prompt=p, response=r) for p, r in zip(prompts, student_responses)]
        collate = collate_fn_builder(tokenizer, args.max_length)
        new_batch = collate(new_examples)
        
        input_ids = new_batch["input_ids"].to(student.device)
        attention_mask = new_batch["attention_mask"].to(student.device)
        labels = new_batch["labels"].to(student.device)
        
        mode = "on_policy"
    else:
        # Supervised distillation
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
    }
    
    return metrics

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_unsloth and not HAS_UNSLOTH:
        raise ImportError("Unsloth requested but not installed.")

    if args.run_name is None:
        prefix = "unsloth" if args.use_unsloth else "hf"
        s_name = args.student_model.split("/")[-1]
        args.run_name = f"{prefix}_twophase_{s_name}_lr{args.lr}"

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
            full_finetuning=False,
            load_in_8bit=False,
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
            full_finetuning=False,
            load_in_8bit=False,
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
            full_finetuning=False,
            load_in_8bit=False,
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
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                val_ce_loss = evaluate_student(student, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    print(f" Step {global_step} ({phase_name}) | Val CE Loss: {val_ce_loss:.4f}")
                    accelerator.log({"val/ce_loss": val_ce_loss}, step=global_step)

    if accelerator.is_main_process:
        output_dir = f"distilled_model_{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save student
        accelerator.unwrap_model(student).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save teacher
        teacher_dir = f"teacher_model_{args.run_name}"
        os.makedirs(teacher_dir, exist_ok=True)
        accelerator.unwrap_model(teacher).save_pretrained(teacher_dir)
        
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
    parser.add_argument("--lambda_on_policy", type=float, default=0.5, help="Max probability for on-policy at end")
    
    # Common args
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="two-phase-distillation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--use_unsloth", type=str2bool, default=True)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key): 
                    setattr(args, key, value)

    # Type conversions
    for attr in ['lr', 'teacher_lr', 'beta', 'temperature', 'lambda_on_policy', 
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