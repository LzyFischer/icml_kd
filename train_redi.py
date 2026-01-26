from unsloth import FastLanguageModel
import os
import torch
import torch.nn.functional as F
import argparse
import yaml
import warnings
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union
import math
import wandb
import json
from contextlib import nullcontext
import pdb

# Import your custom data loader
from load_data import load_data_source
from utils import generalized_jsd_loss, KL_REGISTRY

# -----------------------------------------------------------------------------
# IMPORT FROM EVAL_VLLM (As requested)
# -----------------------------------------------------------------------------
# We import the registry and helper functions to handle different data types (Math, MCQ, etc.)
from eval_vllm import (
    DATASET_REGISTRY,
    extract_prediction,
    extract_reference,  # Added this import
    math_answers_equal,
    normalize_math_answer,
)

# Suppress warnings
warnings.filterwarnings("ignore")
# os.environ["VLLM_USE_V1"] = "1"

# -----------------------------------------------------------------------------
# Curriculum Learning Utilities
# -----------------------------------------------------------------------------


class CurriculumScheduler:
    """Manages curriculum progression across training steps."""

    def __init__(self, total_steps: int, warmup_steps: int = 100):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Increment step counter."""
        self.current_step += 1

    def get_progress(self) -> float:
        """Get training progress [0, 1]."""
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return min(1.0, progress)

    def get_alignment_mode(self) -> str:
        """
        Get current alignment mode for curriculum.
        CE -> top1_kl -> topk_kl -> full_kl
        """
        progress = self.get_progress()
        if progress < 0.3:
            return "top1_kl"  # Cross-entropy loss
        elif progress < 0.9:
            return "topk_kl"  # KL on top-1 predictions
        elif progress < 1.0:
            return "full_kl"  # KL on top-k predictions
        else:
            return "full_kl"  # Full KL divergence

    def get_token_percentage(self) -> float:
        """
        Get percentage of tokens to use in KL calculation.
        Gradually increase from 30% to 100%.
        """
        progress = self.get_progress()
        min_pct = 0.3
        max_pct = 1.0
        return min_pct + (max_pct - min_pct) * progress


# -----------------------------------------------------------------------------
# Enhanced Reward Functions
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# NEW: SWLP (Surprisal-Weighted Length Penalty) Helper Functions
# -----------------------------------------------------------------------------


def get_step_segmentation_masks(
    input_ids: torch.Tensor, period_id: int, newline_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identifies reasoning steps based on delimiters.
    Logic: If delimiters appear consecutively (e.g., ".\n\n"),
    the LAST one is the split point.
    """
    # 1. Base delimiter mask
    is_period = input_ids == period_id
    is_newline = input_ids == newline_id
    is_delimiter = is_period | is_newline  # [B, L]

    # 2. Shift detection to check the next token
    # We want: Current is Delimiter AND Next is NOT Delimiter
    next_is_delimiter = torch.zeros_like(is_delimiter)
    next_is_delimiter[:, :-1] = is_delimiter[:, 1:]  # Shift left

    # 3. Identify Step Ends (The tail of a delimiter chain)
    is_step_end = is_delimiter & (~next_is_delimiter)

    # 4. Generate Step IDs
    # Step ID increases AFTER the step end.
    step_starts = torch.zeros_like(is_step_end)
    step_starts[:, 1:] = is_step_end[:, :-1]  # Shift right to mark start of new step
    step_starts[:, 0] = 1  # Force start at index 0

    step_ids = torch.cumsum(step_starts.long(), dim=-1) - 1  # 0-indexed IDs
    return is_step_end, step_ids


def compute_swlp_reward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer,
    penalty_coefficient: float = 0.01,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Intersection of Unimportance Length Penalty.
    Only penalizes steps where BOTH Student and Teacher find the start 'unsurprising'.
    """

    # Dynamic ID detection for robustness across tokenizers
    period_id = tokenizer.encode(".", add_special_tokens=False)[-1]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    # 1. Shift Logits and Labels (Standard Causal LM Logic)
    # logits[t] predicts input_ids[t+1]
    shift_s_logits = student_logits[..., :-1, :].contiguous()
    shift_t_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # 2. Compute Surprisal (NLL)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    s_nll = loss_fct(
        shift_s_logits.view(-1, shift_s_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.shape)
    t_nll = loss_fct(
        shift_t_logits.view(-1, shift_t_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.shape)

    # 3. Compute Unimportance (Probability proxy)
    # High Prob (Low NLL) -> Unimportance ~ 1.0 (Trivial/Redundant)
    # Low Prob (High NLL) -> Unimportance ~ 0.0 (Important/Surprising)
    s_unimp = torch.exp(-s_nll / temperature)
    t_unimp = torch.exp(-t_nll / temperature)

    # Intersection: Both must agree it's trivial to be penalized
    intersection_unimp = s_unimp * t_unimp  # [B, L-1]

    # 4. Step Segmentation
    is_step_end, step_ids = get_step_segmentation_masks(
        shift_labels, period_id, newline_id
    )

    # 5. Identify Anchor (First Token) for each step
    is_first_token = torch.zeros_like(is_step_end)
    is_first_token[:, 0] = True
    is_first_token[:, 1:] = is_step_end[:, :-1]

    # 6. Broadcast First Token Unimportance to the whole step
    batch_size, seq_len = intersection_unimp.shape
    step_weights = torch.zeros_like(intersection_unimp)

    # Vectorized gathering is tricky with variable step counts per batch,
    # using a robust loop over batch (batch size is usually small in GRPO, e.g., 4-16)
    for b in range(batch_size):
        ids = step_ids[b]  # [L-1]
        first_indices = torch.nonzero(is_first_token[b], as_tuple=True)[0]

        # Get values at anchor points
        first_vals = intersection_unimp[b, first_indices]

        # Determine valid range (ids can't exceed number of found steps)
        num_steps = len(first_vals)
        safe_mask = ids < num_steps

        # Fill step_weights: look up the value for the step ID of the current token
        step_weights[b, safe_mask] = first_vals[ids[safe_mask]]

    # 7. Apply Completion Mask (Don't penalize prompt)
    # completion_mask matches input_ids [B, L]. We need [B, L-1]
    valid_mask = completion_mask[..., 1:].float()

    # 8. Calculate Final Penalty per Sample
    # Sum of weighted length
    sample_penalty = torch.sum(step_weights * valid_mask, dim=-1)

    # Reward is negative penalty
    rewards = -penalty_coefficient * sample_penalty

    # Return raw rewards and the mean unimportance for logging
    return rewards, intersection_unimp.mean()
# -----------------------------------------------------------------------------

def compute_top1_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    teacher_targets = teacher_logits.argmax(dim=-1)
    
    # 增加 clamp 防止 logit 爆炸
    student_logits = torch.clamp(student_logits, min=-100, max=100) 
    
    loss_per_token = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)) / temperature, # 记得除以温度
        teacher_targets.view(-1),
        reduction='none'
    ).view(student_logits.shape[:-1])
    
    # 同样加上 nan_to_num
    loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0)
    
    return loss_per_token


def compute_topk_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Memory-efficient top-k KL using log_softmax for stability."""
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    loss_per_token = torch.zeros(batch_size, seq_len, device=device)
    
    # 加上一个小数值防止除0 (虽然 log_softmax 不需要，但为了保险)
    epsilon = 1e-8 

    for b in range(batch_size):
        # 1. 获取 Teacher 的 Top-K
        topk_vals, topk_indices = teacher_logits[b].topk(k, dim=-1)
        
        # 2. 对齐 Student 的 logits
        student_topk = torch.gather(student_logits[b], -1, topk_indices)
        
        # 3. 使用 log_softmax 提高稳定性 (Key Change)
        # Teacher
        t_log_probs = F.log_softmax(topk_vals / temperature, dim=-1)
        t_probs = torch.exp(t_log_probs)
        
        # Student
        s_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
        
        # 4. 计算 KL: p * (log_p - log_q)
        # 这里的 t_log_probs 和 s_log_probs 已经是 log 后的值了
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        
        loss_per_token[b] = kl
        
    return loss_per_token

def compute_full_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Memory-efficient full KL using log_softmax."""
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    loss_per_token = torch.zeros(batch_size, seq_len, device=device)
    
    chunk_size = 128
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        s_chunk = student_logits[:, start_idx:end_idx, :]
        t_chunk = teacher_logits[:, start_idx:end_idx, :]
        
        # Key Change: Use log_softmax
        t_log_probs = F.log_softmax(t_chunk / temperature, dim=-1)
        s_log_probs = F.log_softmax(s_chunk / temperature, dim=-1)
        t_probs = torch.exp(t_log_probs)
        
        kl_chunk = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        loss_per_token[:, start_idx:end_idx] = kl_chunk
    
    return loss_per_token


def select_topk_tokens_by_kl(
    kl_per_token: torch.Tensor, 
    mask: torch.Tensor, 
    top_percentage: float
) -> torch.Tensor:
    """Memory-efficient top-k token selection."""
    batch_size, seq_len = kl_per_token.shape
    new_mask = torch.zeros_like(mask)
    
    valid_tokens = mask.sum(dim=1)
    k_per_seq = (valid_tokens * top_percentage).long().clamp(min=1)
    
    for i in range(batch_size):
        k = k_per_seq[i].item()
        if k > 0:
            masked_kl = kl_per_token[i].clone()
            masked_kl[mask[i] == 0] = -float('inf')
            topk_indices = masked_kl.topk(k, largest=True).indices
            new_mask[i, topk_indices] = 1
            del masked_kl, topk_indices
    
    return new_mask


# -----------------------------------------------------------------------------
# 替换 compute_alignment_loss 为高效版本
# -----------------------------------------------------------------------------

def compute_alignment_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    beta: float = 0.5,  # 不再使用
    temperature: float = 1.0,
    k: int = 100,  # 默认改为100
    kl_fn=None,  # 不再使用
) -> torch.Tensor:
    """
    Memory-efficient alignment loss.
    替换原有函数，移除未使用的参数以保持接口兼容。
    """
    if mode == "ce" or mode == "top1_kl":
        loss_per_token = compute_top1_kl_efficient(
            student_logits, teacher_logits, temperature=temperature
        )
    elif mode == "topk_kl":
        loss_per_token = compute_topk_kl_efficient(
            student_logits, teacher_logits, k=k, temperature=temperature
        )
    else:  # full_kl
        loss_per_token = compute_full_kl_efficient(
            student_logits, teacher_logits, temperature=temperature
        )
    
    loss_per_token = loss_per_token * mask
    if torch.isnan(loss_per_token).any() or torch.isinf(loss_per_token).any():
        # print("Warning: NaN or Inf detected in alignment loss, masking them out.") # 调试用
        loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0, posinf=0.0, neginf=0.0)
    return loss_per_token


# -----------------------------------------------------------------------------
# Main Reward Computation with Curriculum
# -----------------------------------------------------------------------------


class CurriculumRewardFunction:
    """
    Comprehensive reward function with curriculum learning and SWLP logic.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        curriculum_scheduler: CurriculumScheduler,
        eval_type: str = "math",
        device: str = "cuda",
        # Reward weights
        w_verified: float = 1.0,
        w_alignment: float = 0.5,
        w_length: float = 0.2,  # This now controls the SWLP weight
        w_answer_pred: float = 0.3,
        # SWLP Hyperparameters (New)
        swlp_beta: float = 0.02,  # The lambda penalty coefficient
        swlp_temperature: float = 0.5,  # The sensitivity to surprisal
        # Other
        n_answer_tokens: int = 20,
        kl_type: str = "generalized_jsd",
        max_length: int = 2048,
    ):
        self.kl_fn = KL_REGISTRY[kl_type]
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.scheduler = curriculum_scheduler
        self.eval_type = eval_type
        self.device = device
        self.max_length = max_length

        # Weights
        self.w_verified = w_verified
        self.w_alignment = w_alignment
        self.w_length = w_length
        self.w_answer_pred = w_answer_pred

        # SWLP Params
        self.swlp_beta = swlp_beta
        self.swlp_temperature = swlp_temperature

        self.n_answer_tokens = n_answer_tokens

        self.__name__ = "curriculum_reward"
        self.last_metrics = {}

    def _stash_metrics(self, metrics: Dict[str, float]):
        self.last_metrics = {k: float(v) for k, v in metrics.items()}

    def _clear_memory(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(
        self,
        prompts: List[List[Dict]],
        completions: List[List[Dict]],
        reference: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Memory-optimized reward computation with micro-batching.
        替换 CurriculumRewardFunction 类中的 __call__ 方法。
        """
        
        # 1. Initial cleanup
        self._clear_memory()
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        if reference is None:
            reference = kwargs.get('answer', kwargs.get('answers', []))
            if not reference:
                reference = [""] * len(prompts)
        
        batch_size = len(prompts)
        
        # Extract text
        prompt_texts = []
        completion_texts = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list): 
                p_content = p[0].get('content', '') if isinstance(p[0], dict) else str(p[0])
            else: 
                p_content = str(p)
            if isinstance(c, list): 
                c_content = c[0].get('content', '') if isinstance(c[0], dict) else str(c[0])
            else: 
                c_content = str(c)
            prompt_texts.append(p_content)
            completion_texts.append(c_content)
        
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        
        # Get curriculum settings
        alignment_mode = self.scheduler.get_alignment_mode()
        token_percentage = self.scheduler.get_token_percentage()

        # =====================================================================
        # Pre-compute verified rewards (CPU only, cheap)
        # =====================================================================
        verified_rewards = []
        for completion, ref in zip(completion_texts, reference):
            prediction = extract_prediction(completion, self.eval_type)
            is_correct = False
            if self.eval_type == "math":
                is_correct = math_answers_equal(prediction, ref)
            else:
                is_correct = str(prediction).lower().strip() == str(ref).lower().strip()
            verified_rewards.append(1.0 if is_correct else 0.0)
        verified_rewards = torch.tensor(verified_rewards, device=self.device)

        # =====================================================================
        # Micro-batching strategy
        # =====================================================================
        micro_batch_size = 4  # 可以根据显存调整为1
        all_alignment_rewards = []
        all_swlp_rewards = []
        all_answer_pred_rewards = []
        mean_redundancies = []
        
        is_gemma = "gemma" in self.student_model.config._name_or_path
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if is_gemma else nullcontext()
        
        with torch.no_grad(), autocast_ctx:
            # Pre-compute reference embeddings (shared across micro-batches)
            ref_inputs = self.tokenizer(
                reference, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=64
            ).to(self.device)
            
            ref_outputs = self.student_model(
                input_ids=ref_inputs.input_ids, 
                output_hidden_states=True
            )
            ref_embeddings = ref_outputs.hidden_states[-1].mean(dim=1).clone()
            del ref_outputs, ref_inputs
            self._clear_memory()
            
            # Process in micro-batches
            for i in range(0, batch_size, micro_batch_size):
                end_idx = min(i + micro_batch_size, batch_size)
                micro_texts = full_texts[i:end_idx]
                micro_prompts = prompt_texts[i:end_idx]
                
                # Tokenize micro-batch
                inputs = self.tokenizer(
                    micro_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                
                # Generate completion mask
                completion_mask = inputs.attention_mask.clone().float()
                for j, p_text in enumerate(micro_prompts):
                    prompt_tokens = self.tokenizer(
                        p_text, 
                        add_special_tokens=True, 
                        truncation=False
                    ).input_ids
                    if isinstance(prompt_tokens[0], list): 
                        prompt_tokens = prompt_tokens[0]
                    p_len = len(prompt_tokens)
                    mask_len = min(p_len, completion_mask.shape[1])
                    completion_mask[j, :mask_len] = 0.0
                
                # =====================================================================
                # Sequential model calls with immediate cleanup
                # =====================================================================
                
                # Student forward
                student_outputs = self.student_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                student_logits = student_outputs.logits.clone()
                student_hidden = student_outputs.hidden_states[-1].clone()
                del student_outputs
                torch.cuda.empty_cache()
                
                # Teacher forward
                teacher_outputs = self.teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False
                )
                teacher_logits = teacher_outputs.logits.clone()
                del teacher_outputs
                torch.cuda.empty_cache()
                
                # =====================================================================
                # Alignment Reward (使用高效版本)
                # =====================================================================
                alignment_loss = compute_alignment_loss(
                    student_logits, 
                    teacher_logits, 
                    completion_mask,
                    mode=alignment_mode, 
                    temperature=1.0,
                    k=100  # 从1000减少到100
                )
                
                token_mask = select_topk_tokens_by_kl(
                    alignment_loss, 
                    completion_mask, 
                    top_percentage=token_percentage
                )
                
                masked_alignment = (alignment_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
                alignment_rewards_micro = -masked_alignment
                all_alignment_rewards.append(alignment_rewards_micro.cpu())
                
                del alignment_loss, token_mask, masked_alignment
                torch.cuda.empty_cache()
                
                # =====================================================================
                # SWLP Length Reward
                # =====================================================================
                swlp_rewards_micro, mean_redundancy = compute_swlp_reward(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    input_ids=inputs.input_ids,
                    completion_mask=completion_mask,
                    tokenizer=self.tokenizer,
                    penalty_coefficient=self.swlp_beta,
                    temperature=self.swlp_temperature
                )
                all_swlp_rewards.append(swlp_rewards_micro.cpu())
                mean_redundancies.append(mean_redundancy.cpu())
                
                # Delete logits immediately after use
                del teacher_logits, student_logits
                torch.cuda.empty_cache()
                
                # =====================================================================
                # Answer Prediction Reward
                # =====================================================================
                answer_pred_rewards_micro = torch.zeros(end_idx - i, device=self.device)
                for j in range(end_idx - i):
                    full_len = inputs.attention_mask[j].sum().item()
                    if full_len > self.n_answer_tokens:
                        start_idx = int(full_len - self.n_answer_tokens)
                        end_idx_slice = int(full_len)
                        curr_embed = student_hidden[j, start_idx:end_idx_slice, :].mean(dim=0)
                        
                        similarity = F.cosine_similarity(
                            curr_embed.unsqueeze(0), 
                            ref_embeddings[i + j].unsqueeze(0), 
                            dim=-1
                        )
                        answer_pred_rewards_micro[j] = similarity.item()
                    else:
                        answer_pred_rewards_micro[j] = 0.5
                
                all_answer_pred_rewards.append(answer_pred_rewards_micro.cpu())
                
                # Cleanup micro-batch
                del student_hidden, inputs, completion_mask
                torch.cuda.empty_cache()
            
            # Final cleanup
            del ref_embeddings
            self._clear_memory()
        
        # =====================================================================
        # Combine results on CPU then move to GPU
        # =====================================================================
        alignment_rewards = torch.cat(all_alignment_rewards).to(self.device)
        swlp_rewards = torch.cat(all_swlp_rewards).to(self.device)
        answer_pred_rewards = torch.cat(all_answer_pred_rewards).to(self.device)
        mean_redundancy = torch.stack(mean_redundancies).mean()
        
        # Normalize and combine
        alignment_norm = alignment_rewards
        length_norm = swlp_rewards
        answer_norm = (answer_pred_rewards + 1) / 2

        complex_reward = (
            (alignment_norm * self.w_alignment)
            + (length_norm * self.w_length)
            + (answer_norm * self.w_answer_pred)
        )

        final_reward = verified_rewards * self.w_verified + complex_reward

        # Metrics
        metrics = {
            "reward/final_mean": final_reward.mean().item(),
            "reward/verified_mean": verified_rewards.mean().item(),
            # "reward/length_mean": length_norm.mean().item(),
            "reward/alignment_mean": alignment_norm.mean().item(),
            "reward/swlp_length_penalty_mean": length_norm.mean().item(),
            "reward/redundancy_score": mean_redundancy.item(),
            "reward/answer_pred_mean": answer_norm.mean().item(),
            "curriculum/progress": self.scheduler.get_progress(),
        }
        self._stash_metrics(metrics)

        return final_reward.cpu().tolist()


# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------


def train_teacher(args):
    # 1. Configuration
    max_seq_length = args.max_length
    lora_rank = args.lora_r

    # -------------------------------------------------------------------------
    # DYNAMIC DATASET DETECTION
    # -------------------------------------------------------------------------
    # If dataset_name not provided, infer from path (e.g., data/gsm8k/train.jsonl -> gsm8k)
    dataset_name = args.dataset_name
    if not dataset_name and args.train_file:
        try:
            # Assuming structure is .../dataset_name/filename
            dataset_name = os.path.basename(os.path.dirname(args.train_file))
        except Exception:
            dataset_name = "default"

    print(f"Detected Dataset Name: {dataset_name}")

    # Look up registry for correct evaluation type
    dataset_config = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY.get("default"))
    # If exact name match fails, try partial match (common pattern in registry)
    if dataset_name not in DATASET_REGISTRY:
        for k in DATASET_REGISTRY:
            if dataset_name in k:
                dataset_config = DATASET_REGISTRY[k]
                break

    eval_type = dataset_config.get("type", "text")
    print(f"Using Evaluation Type: {eval_type}")

    # -------------------------------------------------------------------------
    # Load Data & Models
    # -------------------------------------------------------------------------
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
        ds = ds.select(range(min(len(ds), args.max_train_samples)))
    

    steps_per_epoch = len(ds) // (args.batch_size * args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_steps = args.max_steps
        print(f"Mode: Max Steps detected. Training will stop after {total_steps} steps (Overriding epochs).")
    else:
        total_steps = steps_per_epoch * args.num_epochs
        print(f"Mode: Epoch based. Training will stop after {args.num_epochs} epochs (~{total_steps} steps).")

    curriculum_scheduler = CurriculumScheduler(
        total_steps=total_steps,
        warmup_steps=args.curriculum_warmup_steps,
    )

    print(f"Loading Trainer Model: {args.teacher_model}")
    is_lora_checkpoint = os.path.exists(os.path.join(args.teacher_model, "adapter_config.json"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.25,
    )

    if not is_lora_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        # Iter 2+: LoRA checkpoint loaded (Unsloth auto-loaded adapters)
        print(f"Resuming training from LoRA checkpoint: {args.student_model}")

        # Critical: Unsloth defaults to inference mode when loading adapters.
        # We must explicitly enable training gradients for LoRA layers.
        FastLanguageModel.for_training(model)

    print(f"Loading Student Model for Rewards: {args.student_model}")
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        gpu_memory_utilization=0.05,
    )
    FastLanguageModel.for_inference(student_model)

    # Prepare Dataset
    dataset_dict = {
        "prompt": [],
        "reference": []
    }
    
    # CHANGED: Use formatter for prompt, but extract_reference for truth.
    # Also removed hardcoded system prompt, using simple chat template.
    formatter = dataset_config["formatter"]
    
    for row in ds:
        try:
            # Get prompt string from formatter (ignore formatter's response)
            prompt_text, _ = formatter(row)
            
            # Get strict reference from helper
            reference_text = extract_reference(row, eval_type)
            
            # Use standard user prompt structure (no system prompt)
            dataset_dict["prompt"].append([
                {"role": "user", "content": prompt_text}
            ])
            dataset_dict["reference"].append(reference_text)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

    hf_dataset = Dataset.from_dict(dataset_dict)
    print(f"Processed {len(hf_dataset)} training examples.")

    # Initialize Reward Function with EVAL_TYPE
    reward_function = CurriculumRewardFunction(
        student_model=student_model,
        teacher_model=model,
        tokenizer=student_tokenizer,
        curriculum_scheduler=curriculum_scheduler,
        eval_type=eval_type,  # <--- PASSING THE EVAL TYPE
        device="cuda",
        w_verified=args.w_verified,
        w_alignment=args.w_alignment,
        w_length=args.w_length,
        w_answer_pred=args.w_answer_pred,
        n_answer_tokens=args.n_answer_tokens,
        kl_type=args.kl_type,
        swlp_beta=args.swlp_beta,
        swlp_temperature=args.swlp_temperature,
        max_length=args.max_new_tokens,
    )

    training_args = GRPOConfig(
        output_dir=f"ckpts/{dataset_name}/{args.run_name}",
        run_name=args.run_name,
        learning_rate=args.teacher_lr,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_teacher_samples,
        max_prompt_length=max_seq_length // 2,
        max_completion_length=args.max_new_tokens,
        max_steps=args.max_steps,
        save_steps=args.save_steps if args.save_steps > 0 else 100,
        max_grad_norm=1.0,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=hf_dataset,
    )

    # Custom Callback
    from transformers import TrainerCallback

    class CurriculumCallback(TrainerCallback):
        def __init__(self, scheduler, reward_fn):
            self.scheduler = scheduler
            self.reward_fn = reward_fn

        def on_train_begin(self, args, state, control, **kwargs):
            """Define the X-axis metric when training starts."""
            if state.is_world_process_zero and wandb.run is not None:
                # Tell WandB: "For any metric matching 'reward/*', use 'train/global_step' as the X-axis"
                wandb.define_metric("reward/*", step_metric="train/global_step")
                wandb.define_metric("curriculum/*", step_metric="train/global_step")
                wandb.define_metric("train/global_step", summary="max")  # Keep track of max step

        def on_step_end(self, args, state, control, **kwargs):
            self.scheduler.step()
            metrics = getattr(self.reward_fn, "last_metrics", {}) or {}

            if metrics:
                print("[Rewards]")
                for k in [
                    "reward/final_mean",
                    "reward/verified_mean",
                    "reward/alignment_mean",
                    "reward/length_mean",
                    "reward/answer_pred_mean",
                ]:
                    if k in metrics:
                        print(f" {k}: {metrics[k]:.4f}")

            if state.is_world_process_zero and metrics and wandb.run is not None:
                # 1. Add global_step to the metrics dict
                metrics["train/global_step"] = state.global_step

                # 2. Log WITHOUT the 'step' argument
                # Let WandB handle the internal step counter naturally
                wandb.log(metrics)

            if state.global_step % 10 == 0:
                print(
                    f"\n[Curriculum] Step {state.global_step} | Progress: {self.scheduler.get_progress():.2%}"
                )

    trainer.add_callback(CurriculumCallback(curriculum_scheduler, reward_function))

    print("Starting GRPO Training with Curriculum Learning...")
    trainer.train()

    model.save_lora(f"ckpts/{dataset_name}/{args.run_name}/final_lora")
    # tokenizer
    tokenizer.save_pretrained(f"ckpts/{dataset_name}/{args.run_name}/final_lora")
    print(f"Training complete! Model saved to ckpts/{args.run_name}/final_lora")


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="./data/date/train.jsonl")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--teacher_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_teacher_samples", type=int, default=4)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=0)
    parser.add_argument("--w_verified", type=float, default=0.4)
    parser.add_argument("--w_alignment", type=float, default=0.6)
    parser.add_argument("--w_length", type=float, default=0.6)
    parser.add_argument("--w_answer_pred", type=float, default=0.4)
    parser.add_argument("--swlp_beta", type=float, default=0.01, help="Penalty coefficient per redundant step")
    parser.add_argument("--swlp_temperature", type=float, default=4, help="Temperature for unimportance prob")
    parser.add_argument("--n_answer_tokens", type=int, default=10)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="teacher")
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument(
        "--kl_type",
        type=str,
        default="reverse",
        choices=["generalized_jsd", "forward", "reverse"],
    )

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


if __name__ == "__main__":
    args = parse_args()
    train_teacher(args)