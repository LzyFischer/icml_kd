import argparse
import math
import torch
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pdb
import time

# Try importing Unsloth for checks inside utils
try:
    from unsloth import FastModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# -----------------------------------------------------------------------------
# General Utils
# -----------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def ensure_dtype_match(model, target_dtype=torch.bfloat16):
    """Ensure all model parameters have the same dtype."""
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
    return model

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard Cross-Entropy loss for SFT."""
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens for CrossEntropyLoss
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )

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


def forward_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    beta: float = 0.5,  # unused, kept for API compatibility
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute KL(teacher || student) for distillation."""
    # Temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    
    # Handle vocabulary size mismatch
    min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :min_vocab_size]
    teacher_logits = teacher_logits[..., :min_vocab_size]
    
    # Compute log probabilities
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    
    # Forward KL: KL(teacher || student)
    # = sum_i p_teacher(i) * (log p_teacher(i) - log p_student(i))
    kl_per_token = torch.nn.functional.kl_div(
        student_log_probs,      
        teacher_log_probs,      
        reduction="none",
        log_target=True,        # Indicate target is in log-space
    )  # [..., V]
    
    kl_per_position = kl_per_token.sum(dim=-1)  # [...]
    
    if mask is not None:
        kl_per_position = kl_per_position * mask
    
    # Temperature scaling (standard in distillation)
    kl_per_position = kl_per_position * (temperature ** 2)
    
    if reduction == "none":
        return kl_per_position
    elif reduction == "mean":
        if mask is not None:
            return kl_per_position.sum() / mask.sum().clamp(min=1)
        else:
            return kl_per_position.mean()
    elif reduction == "sum":
        return kl_per_position.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def reverse_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    beta: float = 0.5,  # unused, kept for API compatibility
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute KL(student || teacher) for distillation."""
    # Temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    
    # Handle vocabulary size mismatch
    min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :min_vocab_size]
    teacher_logits = teacher_logits[..., :min_vocab_size]
    
    # Compute log probabilities
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    
    # Reverse KL: KL(student || teacher)
    # = sum_i p_student(i) * (log p_student(i) - log p_teacher(i))
    kl_per_token = torch.nn.functional.kl_div(
        teacher_log_probs,      # Input: log-probs of approximating distribution
        student_log_probs,      # Target: log-probs of target distribution
        reduction="none",
        log_target=True,        # Indicate target is in log-space
    )  # [..., V]
    
    kl_per_position = kl_per_token.sum(dim=-1)  # [...]
    
    if mask is not None:
        kl_per_position = kl_per_position * mask
    
    # Temperature scaling (standard in distillation)
    kl_per_position = kl_per_position * (temperature ** 2)
    
    if reduction == "none":
        return kl_per_position
    elif reduction == "mean":
        if mask is not None:
            return kl_per_position.sum() / mask.sum().clamp(min=1)
        else:
            return kl_per_position.mean()
    elif reduction == "sum":
        return kl_per_position.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    
KL_REGISTRY = {
    "generalized_jsd": generalized_jsd_loss,
    "forward": forward_kl_div_loss,
    "reverse": reverse_kl_div_loss,
}

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
# Generation and Rewards
# -----------------------------------------------------------------------------

def generate_on_policy(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str], max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0, num_samples: int = 1, use_unsloth: bool = False) -> List[List[str]]:
    """Generate multiple samples per prompt."""
    tokenizer.padding_side = "left"
    
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
    
    # Handling Unsloth inference context
    do_sample = temperature > 0.0 and top_k > 0
    if use_unsloth and HAS_UNSLOTH:
        FastModel.for_inference(model)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=do_sample, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        FastModel.for_training(model)
    else:
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=do_sample, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
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