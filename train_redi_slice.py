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
import json
import wandb

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
    normalize_math_answer
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["VLLM_USE_V1"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return min(1.0, progress)
    
    def get_alignment_mode(self) -> str:
        """
        Get current alignment mode for curriculum.
        CE -> top1_kl -> topk_kl -> full_kl
        """
        progress = self.get_progress()
        if progress < 0.25:
            return "ce"  # Cross-entropy loss
        elif progress < 0.5:
            return "top1_kl"  # KL on top-1 predictions
        elif progress < 0.75:
            return "topk_kl"  # KL on top-k predictions
        else:
            return "full_kl"  # Full KL divergence
    
    def get_token_percentage(self) -> float:
        """
        Get percentage of tokens to use in KL calculation.
        Gradually increase from 10% to 100%.
        """
        progress = self.get_progress()
        min_pct = 1.0
        max_pct = 1.0
        return min_pct + (max_pct - min_pct) * progress

# -----------------------------------------------------------------------------
# Enhanced Reward Functions
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# NEW: SWLP (Surprisal-Weighted Length Penalty) Helper Functions
# -----------------------------------------------------------------------------

def get_step_segmentation_masks(
    input_ids: torch.Tensor, 
    period_id: int, 
    newline_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identifies reasoning steps based on delimiters.
    Logic: If delimiters appear consecutively (e.g., ".\n\n"), 
    the LAST one is the split point.
    """
    # 1. Base delimiter mask
    is_period = (input_ids == period_id)
    is_newline = (input_ids == newline_id)
    is_delimiter = is_period | is_newline  # [B, L]
    
    # 2. Shift detection to check the next token
    # We want: Current is Delimiter AND Next is NOT Delimiter
    next_is_delimiter = torch.zeros_like(is_delimiter)
    next_is_delimiter[:, :-1] = is_delimiter[:, 1:] # Shift left
    
    # 3. Identify Step Ends (The tail of a delimiter chain)
    is_step_end = is_delimiter & (~next_is_delimiter)
    
    # 4. Generate Step IDs
    # Step ID increases AFTER the step end.
    step_starts = torch.zeros_like(is_step_end)
    step_starts[:, 1:] = is_step_end[:, :-1] # Shift right to mark start of new step
    step_starts[:, 0] = 1 # Force start at index 0
    
    step_ids = torch.cumsum(step_starts.long(), dim=-1) - 1 # 0-indexed IDs
    return is_step_end, step_ids

def compute_swlp_reward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer,
    penalty_coefficient: float = 0.01,
    temperature: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Intersection of Unimportance Length Penalty.
    Only penalizes steps where BOTH Student and Teacher find the start 'unsurprising'.
    """
    # Dynamic ID detection for robustness across tokenizers
    # Note: We do this check inside to ensure we have the correct tokenizer reference
    period_id = tokenizer.encode(".", add_special_tokens=False)[-1]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    # 1. Shift Logits and Labels (Standard Causal LM Logic)
    # logits[t] predicts input_ids[t+1]
    shift_s_logits = student_logits[..., :-1, :].contiguous()
    shift_t_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 2. Compute Surprisal (NLL)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    s_nll = loss_fct(shift_s_logits.view(-1, shift_s_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
    t_nll = loss_fct(shift_t_logits.view(-1, shift_t_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
    
    # 3. Compute Unimportance (Probability proxy)
    # High Prob (Low NLL) -> Unimportance ~ 1.0 (Trivial/Redundant)
    # Low Prob (High NLL) -> Unimportance ~ 0.0 (Important/Surprising)
    s_unimp = torch.exp(-s_nll / temperature)
    t_unimp = torch.exp(-t_nll / temperature)
    
    # Intersection: Both must agree it's trivial to be penalized
    intersection_unimp = s_unimp * t_unimp # [B, L-1]
    
    # 4. Step Segmentation
    # We analyze steps on the shifted labels (the actual tokens being predicted)
    is_step_end, step_ids = get_step_segmentation_masks(shift_labels, period_id, newline_id)
    
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
        ids = step_ids[b] # [L-1]
        first_indices = torch.nonzero(is_first_token[b], as_tuple=True)[0]
        
        # Get values at anchor points
        first_vals = intersection_unimp[b, first_indices]
        
        # Determine valid range (ids can't exceed number of found steps)
        # In rare cases of truncation, step_ids might go higher than found first_tokens
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


def compute_topk_kl(
    student_logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    k: int = 5,
    beta: float = 0.5,
    temperature: float = 1.0,
    kl_fn=generalized_jsd_loss
) -> torch.Tensor:
    """
    Compute JSD divergence on top-k predictions only.
    Treats the Top-K tokens as a closed vocabulary for normalization.
    """
    # Get top-k indices from teacher
    topk_vals, topk_indices = teacher_logits.topk(k, dim=-1)
    
    # Create masked distributions (gather logits)
    student_topk = torch.gather(student_logits, -1, topk_indices)
    teacher_topk = topk_vals
    
    # Compute JSD on the reduced vocabulary
    loss_per_position = kl_fn(
        student_topk, 
        teacher_topk, 
        mask=None, 
        beta=beta, 
        temperature=temperature, 
        reduction="none"
    )
    
    return loss_per_position

def compute_top1_kl(
    student_logits: torch.Tensor, 
    teacher_logits: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    kl_fn=generalized_jsd_loss
) -> torch.Tensor:
    """
    Compute JSD based on top-1 predictions against a 'Hard' Teacher target.
    """
    # Identify teacher targets
    teacher_indices = teacher_logits.argmax(dim=-1).unsqueeze(-1)

    # Create a "hard" teacher logit tensor
    hard_teacher_logits = torch.full_like(teacher_logits, -1e4)
    hard_teacher_logits.scatter_(-1, teacher_indices, 1e4)

    # Compute JSD
    loss_per_position = kl_fn(
        student_logits,
        hard_teacher_logits,
        mask=None,
        beta=beta,
        temperature=temperature,
        reduction="none"
    )
    
    return loss_per_position

def compute_alignment_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    beta: float = 0.5,
    temperature: float = 1.0,
    k: int = 1000,
    kl_fn=generalized_jsd_loss
) -> torch.Tensor:
    """
    Compute alignment loss based on curriculum mode.
    """
    mode = "ce"
    if mode == "ce":
        # Standard Cross Entropy (Hard Target KL)
        teacher_preds = teacher_logits.argmax(dim=-1)
        loss_per_token = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_preds.view(-1),
            reduction='none'
        ).view(student_logits.shape[:-1])
    
    elif mode == "top1_kl":
        loss_per_token = compute_top1_kl(
            student_logits, teacher_logits, beta=beta, temperature=temperature, kl_fn=kl_fn
        )
    
    elif mode == "topk_kl":
        loss_per_token = compute_topk_kl(
            student_logits, teacher_logits, k=k, beta=beta, temperature=temperature, kl_fn=kl_fn
        )
    
    else:  # full_kl
        loss_per_token = kl_fn(
            student_logits,
            teacher_logits,
            mask=None,
            beta=beta,
            temperature=temperature,
            reduction="none"
        )
    
    # Apply mask
    loss_per_token = loss_per_token * mask

    return loss_per_token

def select_topk_tokens_by_kl(
    kl_per_token: torch.Tensor,
    mask: torch.Tensor,
    top_percentage: float
) -> torch.Tensor:
    """Select top-k tokens with highest KL for alignment."""
    batch_size, seq_len = kl_per_token.shape
    
    masked_kl = kl_per_token.clone()
    masked_kl[mask == 0] = -float('inf')
    
    valid_tokens = mask.sum(dim=1)
    k_per_seq = (valid_tokens * top_percentage).long().clamp(min=1)
    
    new_mask = torch.zeros_like(mask)
    for i in range(batch_size):
        k = k_per_seq[i].item()
        if k > 0:
            topk_indices = masked_kl[i].topk(k).indices
            new_mask[i, topk_indices] = 1
    
    return new_mask


# -----------------------------------------------------------------------------
# Main Reward Computation with Curriculum
# -----------------------------------------------------------------------------

class CurriculumRewardFunction:
    """
    Memory-Optimized Curriculum Reward Function with Micro-Batching.
    Splits large GRPO batches into mini-batches to prevent OOM with large-vocab models (e.g., Qwen 2.5).
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
        w_length: float = 0.2,
        w_answer_pred: float = 0.3,
        # SWLP Hyperparameters
        swlp_beta: float = 0.02,
        swlp_temperature: float = 0.5,
        # Other
        n_answer_tokens: int = 20,
        kl_type: str = "generalized_jsd",
        # Memory Optimization
        mini_batch_size: int = 4,  # <--- NEW: Controls internal batch size
    ):  
        self.kl_fn = KL_REGISTRY[kl_type]
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.scheduler = curriculum_scheduler
        self.eval_type = eval_type
        self.device = device
        
        # Weights
        self.w_verified = w_verified
        self.w_alignment = w_alignment
        self.w_length = w_length
        self.w_answer_pred = w_answer_pred
        
        # SWLP Params
        self.swlp_beta = swlp_beta
        self.swlp_temperature = swlp_temperature
        
        self.n_answer_tokens = n_answer_tokens
        self.mini_batch_size = mini_batch_size  # <--- Store this
        
        self.__name__ = "curriculum_reward"
        self.last_metrics = {}

    def _clear_memory(self):
        """Force garbage collection and clear CUDA cache."""
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _stash_metrics(self, metrics: Dict[str, float]):
        self.last_metrics = {k: float(v) for k, v in metrics.items()}

    def _process_mini_batch(
        self, 
        prompt_texts: List[str], 
        completion_texts: List[str], 
        reference: List[str],
        alignment_mode: str,
        token_percentage: float
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Processes a small chunk of data to compute rewards without exploding VRAM.
        Returns: (List of rewards, Dictionary of metrics for this chunk)
        """
        batch_size = len(prompt_texts)
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        
        # =====================================================================
        # 1. Verified Reward (CPU/Cheap)
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
            
        verified_rewards_t = torch.tensor(verified_rewards, device=self.device)

        # =====================================================================
        # 2. Forward Passes & Masking (Heavy Lifting)
        # =====================================================================
        try:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048 # Adjust if still OOM
                ).to(self.device)
                
                # Create Completion Mask
                completion_mask = inputs.attention_mask.clone().float()
                for i, p_text in enumerate(prompt_texts):
                    # Robust prompt length calc
                    prompt_tokens = self.tokenizer(p_text, add_special_tokens=True, truncation=False).input_ids
                    if isinstance(prompt_tokens[0], list): prompt_tokens = prompt_tokens[0]
                    p_len = len(prompt_tokens)
                    mask_len = min(p_len, completion_mask.shape[1])
                    completion_mask[i, :mask_len] = 0.0

                # --- A. Answer Prediction (Reference Model) ---
                # Calculate and DELETE immediately to save memory
                ref_inputs = self.tokenizer(reference, return_tensors="pt", padding=True, truncation=True).to(self.device)
                ref_outputs = self.student_model(input_ids=ref_inputs.input_ids, output_hidden_states=True)
                ref_embeddings = ref_outputs.hidden_states[-1].mean(dim=1).clone()
                
                # Cleanup Reference Outputs
                del ref_outputs, ref_inputs
                self._clear_memory()

                # --- B. Student Forward ---
                student_outputs = self.student_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                student_logits = student_outputs.logits
                student_hidden = student_outputs.hidden_states[-1]
                
                # --- C. Teacher Forward (Peak Memory Usage Here) ---
                # Ensure we use autocast for logits if possible to save space
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    teacher_outputs = self.teacher_model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=False
                    )
                    teacher_logits = teacher_outputs.logits

                # =============================================================
                # 3. Compute Rewards
                # =============================================================
                
                # --- Alignment Reward ---
                alignment_loss = compute_alignment_loss(
                    student_logits, teacher_logits, completion_mask,
                    mode=alignment_mode, beta=0.5, temperature=1.0
                )
                token_mask = select_topk_tokens_by_kl(
                    alignment_loss, completion_mask, top_percentage=token_percentage
                )
                masked_alignment = (alignment_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
                alignment_rewards = -masked_alignment
                
                del alignment_loss, token_mask # Free up
                
                # --- SWLP Reward ---
                swlp_rewards, mean_redundancy = compute_swlp_reward(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    input_ids=inputs.input_ids,
                    completion_mask=completion_mask,
                    tokenizer=self.tokenizer,
                    penalty_coefficient=self.swlp_beta,
                    temperature=self.swlp_temperature
                )

                # CRITICAL: Drop Logits NOW
                del teacher_outputs, teacher_logits, student_logits
                self._clear_memory()

                # --- Answer Prediction Reward ---
                answer_pred_rewards = torch.zeros(batch_size, device=self.device)
                for i in range(batch_size):
                    full_len = inputs.attention_mask[i].sum().item()
                    if full_len > self.n_answer_tokens:
                        start = int(full_len - self.n_answer_tokens)
                        end = int(full_len)
                        # Slice student hidden states
                        ans_emb = student_hidden[i, start:end, :].mean(dim=0)
                        sim = F.cosine_similarity(
                            ans_emb.unsqueeze(0), ref_embeddings[i].unsqueeze(0), dim=-1
                        )
                        answer_pred_rewards[i] = sim.item()
                    else:
                        answer_pred_rewards[i] = 0.5
                
                # Cleanup remaining large tensors
                del student_outputs, student_hidden, inputs, ref_embeddings
                self._clear_memory()

        except Exception as e:
            print(f"Error in mini-batch forward: {e}")
            import traceback
            traceback.print_exc()
            self._clear_memory()
            return [0.0] * batch_size, {}

        # =====================================================================
        # 4. Combine & Return
        # =====================================================================
        alignment_norm = alignment_rewards
        length_norm = swlp_rewards 
        answer_norm = (answer_pred_rewards + 1) / 2
        
        complex_reward = (
            (alignment_norm * self.w_alignment) +
            (length_norm * self.w_length) + 
            (answer_norm * self.w_answer_pred)
        )
        
        final_reward = verified_rewards_t * self.w_verified + complex_reward
        
        # Calculate mean metrics for this chunk
        metrics = {
            "reward/final_mean": final_reward.mean().item(),
            "reward/verified_mean": verified_rewards_t.mean().item(),
            "reward/length_mean": length_norm.mean().item(),
            "reward/alignment_mean": alignment_norm.mean().item(),
            "reward/swlp_length_penalty_mean": -length_norm.mean().item(),
            "reward/redundancy_score": mean_redundancy.item(),
            "reward/answer_pred_mean": answer_norm.mean().item(),
        }
        
        return final_reward.cpu().tolist(), metrics

    def __call__(
        self,
        prompts: List[List[Dict]],
        completions: List[List[Dict]],
        reference: List[str] = None,
        **kwargs
    ) -> List[float]:
        # Clean start
        self._clear_memory()
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        # Handle reference extraction
        if reference is None:
            reference = kwargs.get('answer', kwargs.get('answers', []))
            if not reference:
                reference = [""] * len(prompts)
        
        # Extract Text
        prompt_texts = []
        completion_texts = []
        for p, c in zip(prompts, completions):
            # Safe extraction
            if isinstance(p, list): p_content = p[0].get('content', '') if isinstance(p[0], dict) else str(p[0])
            else: p_content = str(p)
            if isinstance(c, list): c_content = c[0].get('content', '') if isinstance(c[0], dict) else str(c[0])
            else: c_content = str(c)
            prompt_texts.append(p_content)
            completion_texts.append(c_content)
        
        # Get global curriculum state
        alignment_mode = self.scheduler.get_alignment_mode()
        token_percentage = self.scheduler.get_token_percentage()
        
        # =====================================================================
        # MICRO-BATCH LOOP
        # =====================================================================
        total_batch_size = len(prompt_texts)
        all_final_rewards = []
        aggregated_metrics = {}
        
        # Process in chunks
        for i in range(0, total_batch_size, self.mini_batch_size):
            # 1. Slice Inputs
            p_sub = prompt_texts[i : i + self.mini_batch_size]
            c_sub = completion_texts[i : i + self.mini_batch_size]
            ref_sub = reference[i : i + self.mini_batch_size]
            
            # 2. Process Chunk
            sub_rewards, sub_metrics = self._process_mini_batch(
                p_sub, c_sub, ref_sub, alignment_mode, token_percentage
            )
            
            # 3. Accumulate Results
            all_final_rewards.extend(sub_rewards)
            
            # 4. Accumulate Metrics (Weighted average based on chunk size)
            chunk_len = len(p_sub)
            for k, v in sub_metrics.items():
                if k not in aggregated_metrics:
                    aggregated_metrics[k] = 0.0
                aggregated_metrics[k] += v * chunk_len
            
            # 5. Clean between chunks
            self._clear_memory()
            
        # =====================================================================
        # Finalize
        # =====================================================================
        
        # Normalize metrics
        if total_batch_size > 0:
            final_metrics = {k: v / total_batch_size for k, v in aggregated_metrics.items()}
            # Add curriculum info (constant across batch)
            final_metrics["curriculum/progress"] = self.scheduler.get_progress()
            self._stash_metrics(final_metrics)
        else:
            self._stash_metrics({})
            
        return all_final_rewards

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
        except:
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
    
    # CHANGED: Load dataset directly to access raw rows for extract_reference
    print(f"Loading data from: {args.train_file or args.dataset_name}...")

    try:
        if path_or_name.endswith(".jsonl"):
            ds = load_dataset("json", data_files={split: path_or_name}, split=split)
        else:
            ds = load_dataset("json", data_files={split: path_or_name}, field="instances", split=split)
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
    total_steps = steps_per_epoch * args.num_epochs
    
    curriculum_scheduler = CurriculumScheduler(
        total_steps=total_steps,
        warmup_steps=args.curriculum_warmup_steps
    )
    
    print(f"Loading Trainer Model: {args.teacher_model}")
    is_lora_checkpoint = os.path.exists(os.path.join(args.teacher_model, "adapter_config.json"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.3,
    )

    if not is_lora_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        gpu_memory_utilization=0.1,
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
        num_train_epochs=args.num_epochs,
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
                wandb.define_metric("train/global_step", summary="max") # Keep track of max step

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
                        "reward/answer_pred_mean"
                        ]:
                        if k in metrics: print(f"  {k}: {metrics[k]:.4f}")
                    
            
            if state.is_world_process_zero and metrics and wandb.run is not None:
                # 1. Add global_step to the metrics dict
                metrics["train/global_step"] = state.global_step
                
                # 2. Log WITHOUT the 'step' argument
                # Let WandB handle the internal step counter naturally
                wandb.log(metrics)

            if state.global_step % 10 == 0:
                print(f"\n[Curriculum] Step {state.global_step} | Progress: {self.scheduler.get_progress():.2%}")

    trainer.add_callback(CurriculumCallback(curriculum_scheduler, reward_function))
    
    print("Starting GRPO Training with Curriculum Learning...")
    trainer.train()
    
    model.save_lora(f"ckpts/{dataset_name}/{args.run_name}/final_lora")
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
    parser.add_argument("--teacher_lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=1248)
    parser.add_argument("--max_new_tokens", type=int, default=1248)
    parser.add_argument("--num_teacher_samples", type=int, default=4)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=0)
    parser.add_argument("--w_verified", type=float, default=0.3)
    parser.add_argument("--w_alignment", type=float, default=0.5)
    parser.add_argument("--w_length", type=float, default=0.1)
    parser.add_argument("--w_answer_pred", type=float, default=0.4)
    parser.add_argument("--swlp_beta", type=float, default=0.01, help="Penalty coefficient per redundant step")
    parser.add_argument("--swlp_temperature", type=float, default=0.5, help="Temperature for unimportance prob")
    parser.add_argument("--n_answer_tokens", type=int, default=10)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="teacher")
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--kl_type", type=str, default="reverse", choices=["generalized_jsd", "forward", "reverse"])
    
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