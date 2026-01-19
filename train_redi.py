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
os.environ["VLLM_USE_V1"] = "0"

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

def compute_student_understanding_time(
    student_hidden_states: torch.Tensor,
    reference_embedding: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.7,
    gamma: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute soft 'first understanding time'."""
    student_norm = F.normalize(student_hidden_states, dim=-1)
    reference_norm = F.normalize(reference_embedding.unsqueeze(1), dim=-1)
    
    similarities = (student_norm * reference_norm).sum(dim=-1)
    p_i = torch.sigmoid(gamma * (similarities - tau))
    p_i = p_i * mask
    
    batch_size, seq_len = p_i.shape
    q_i = torch.zeros_like(p_i)
    
    for i in range(seq_len):
        if i == 0:
            q_i[:, i] = p_i[:, i]
        else:
            prod = torch.ones(batch_size, device=p_i.device)
            for j in range(i):
                prod = prod * (1 - p_i[:, j])
            q_i[:, i] = p_i[:, i] * prod
    
    positions = torch.arange(seq_len, device=p_i.device).float().unsqueeze(0)
    t_soft = (positions * q_i).sum(dim=1)
    
    return t_soft, similarities

def compute_length_regularization_reward(
    t_soft: torch.Tensor,
    actual_length: torch.Tensor,
    b: float = 5.0,
    lambda_penalty: float = 0.1
) -> torch.Tensor:
    """Compute length regularization reward."""
    early_reward = 1.0 / torch.log(2.0 + t_soft)
    excess_length = actual_length - t_soft - b
    length_penalty = lambda_penalty * F.softplus(excess_length)
    
    return early_reward - length_penalty

# -----------------------------------------------------------------------------
# Main Reward Computation with Curriculum
# -----------------------------------------------------------------------------

class CurriculumRewardFunction:
    """
    Comprehensive reward function with curriculum learning and dynamic evaluation types.
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
        # Hyperparameters
        tau: float = 0.7,
        gamma: float = 10.0,
        b: float = 5.0,
        lambda_penalty: float = 0.1,
        n_answer_tokens: int = 20,
        kl_type: str = "generalized_jsd"
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
        
        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.b = b
        self.lambda_penalty = lambda_penalty
        self.n_answer_tokens = n_answer_tokens
        
        self.__name__ = "curriculum_reward"
        self.last_metrics = {}
    
    def _stash_metrics(self, metrics: Dict[str, float]):
        self.last_metrics = {k: float(v) for k, v in metrics.items()}
    
    def __call__(
        self,
        prompts: List[List[Dict]],
        completions: List[List[Dict]],
        reference: List[str] = None,
        **kwargs
    ) -> List[float]:
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        if reference is None:
            reference = kwargs.get('answer', kwargs.get('answers', []))
            if not reference:
                reference = [""] * len(prompts)
        
        batch_size = len(prompts)
        
        # Extract text from chat format
        prompt_texts = []
        completion_texts = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list) and len(p) > 0:
                prompt_texts.append(p[0].get('content', '') if isinstance(p[0], dict) else str(p[0]))
            else:
                prompt_texts.append(str(p))
            
            if isinstance(c, list) and len(c) > 0:
                completion_texts.append(c[0].get('content', '') if isinstance(c[0], dict) else str(c[0]))
            else:
                completion_texts.append(str(c))
        
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        
        # Get curriculum settings
        alignment_mode = self.scheduler.get_alignment_mode()
        token_percentage = self.scheduler.get_token_percentage()
        
        # =====================================================================
        # 1. Verified Reward
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
        # 2. Forward Passes
        # =====================================================================
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                ref_inputs = self.tokenizer(
                    reference,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # --- NEW: GENERATE COMPLETION MASK ---
                # We start with the padding mask (1 for real tokens, 0 for padding)
                completion_mask = inputs.attention_mask.clone().float()
                
                # Iterate through batch to mask out the prompt tokens
                for i, p_text in enumerate(prompt_texts):
                    # We tokenize the prompt to find its length.
                    # Note: We use add_special_tokens=True to match the behavior of 'inputs'
                    # which likely includes a BOS token at the start.
                    prompt_tokens = self.tokenizer(
                        p_text, 
                        return_tensors="pt", 
                        add_special_tokens=True, 
                        truncation=False
                    ).input_ids[0]
                    
                    p_len = len(prompt_tokens)
                    
                    # Zero out the prompt part of the mask
                    # Ensure we don't go out of bounds if prompt is somehow longer than max_len
                    mask_len = min(p_len, completion_mask.shape[1])
                    completion_mask[i, :mask_len] = 0.0
                
                # -------------------------------------

                student_outputs = self.student_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                student_hidden = student_outputs.hidden_states[-1]
                student_logits = student_outputs.logits

                ref_outputs = self.student_model(
                    input_ids=ref_inputs.input_ids,
                    output_hidden_states=True
                )
                ref_embeddings = ref_outputs.hidden_states[-1].mean(dim=1)
                
                teacher_outputs = self.teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False
                )
                teacher_logits = teacher_outputs.logits
                
                # Original mask for sequence lengths (includes prompt, excludes padding)
                mask = inputs.attention_mask.float() 
                seq_lengths = mask.sum(dim=1)
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return [1.0] * batch_size
        
        # 3. Alignment Reward
        # CHANGED: Passed `completion_mask` instead of `mask`
        alignment_loss = compute_alignment_loss(
            student_logits, teacher_logits, completion_mask,
            mode=alignment_mode, beta=0.5, temperature=1.0
        )
        
        # 4. Token Selection
        # CHANGED: Passed `completion_mask` so we only select Top-K tokens from the ANSWER
        token_mask = select_topk_tokens_by_kl(
            alignment_loss, completion_mask, top_percentage=token_percentage
        )
        
        # Compute mean alignment loss over the selected tokens
        masked_alignment = (alignment_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
        alignment_rewards = -masked_alignment
        
        # 5. Length/Understanding Time
        # Note: We likely still want to calculate understanding time based on the full flow,
        # but if you want that restricted to answers too, use completion_mask here as well.
        # Keeping original mask for now to capture prompt-response understanding dynamics.
        t_soft, _ = compute_student_understanding_time(
            student_hidden, ref_embeddings, mask, tau=self.tau, gamma=self.gamma
        )
        length_rewards = compute_length_regularization_reward(
            t_soft, seq_lengths, b=self.b, lambda_penalty=self.lambda_penalty
        )
        
        # 6. Answer Prediction
        answer_pred_rewards = torch.zeros(batch_size, device=self.device)
        try:
            for i in range(batch_size):
                seq_len = int(seq_lengths[i].item())
                if seq_len > self.n_answer_tokens:
                    answer_start = seq_len - self.n_answer_tokens
                    answer_embeddings = student_hidden[i, answer_start:seq_len, :]
                    answer_embedding = answer_embeddings.mean(dim=0)
                    similarity = F.cosine_similarity(
                        answer_embedding.unsqueeze(0), ref_embeddings[i].unsqueeze(0), dim=-1
                    )
                    answer_pred_rewards[i] = similarity.item()
                else:
                    answer_pred_rewards[i] = 0.5
        except:
            answer_pred_rewards = torch.ones(batch_size, device=self.device) * 0.5
        
        # 7. Combine
        alignment_norm = torch.exp(alignment_rewards) * 10.0
        length_norm = torch.exp(length_rewards * 0.1)
        answer_norm = (answer_pred_rewards + 1) / 2

        
        complex_reward = (
            (alignment_norm * self.w_alignment) +
            (length_norm * self.w_length) +
            (answer_norm * self.w_answer_pred)
        )
        
        final_reward = 2.0 * verified_rewards * complex_reward
        final_reward = torch.where(
            verified_rewards > 0.5,
            final_reward,
            complex_reward * 2
        )
        
        metrics = {
            "reward/final_mean": final_reward.mean().item(),
            "reward/verified_mean": verified_rewards.mean().item(),
            "reward/alignment_mean": alignment_norm.mean().item(),
            "reward/length_mean": length_norm.mean().item(),
            "reward/answer_pred_mean": answer_norm.mean().item(),
            "curriculum/progress": self.scheduler.get_progress(),
            "curriculum/token_percentage": token_percentage,
            "curriculum/alignment_mode": ["ce", "top1_kl", "topk_kl", "full_kl"].index(alignment_mode),
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
        raw_ds = load_dataset("json", data_files={"train": args.train_file}, split="train")
    except:
         raw_ds = load_dataset("json", data_files={"train": args.train_file}, split="train", field="instances")
         
    if args.max_train_samples:
        raw_ds = raw_ds.select(range(min(len(raw_ds), args.max_train_samples)))

    steps_per_epoch = len(raw_ds) // (args.batch_size * args.gradient_accumulation_steps)
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
        gpu_memory_utilization=0.7,
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
        gpu_memory_utilization=0.2,
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
    
    for row in raw_ds:
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
        tau=args.tau,
        gamma=args.gamma,
        b=args.length_baseline,
        lambda_penalty=args.lambda_penalty,
        n_answer_tokens=args.n_answer_tokens,
        kl_type=args.kl_type
    )
    
    training_args = GRPOConfig(
        output_dir=f"ckpts/{args.run_name}",
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
    
    model.save_lora(f"ckpts/{args.run_name}/final_lora")
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1100)
    parser.add_argument("--max_new_tokens", type=int, default=1100)
    parser.add_argument("--num_teacher_samples", type=int, default=5)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=0)
    parser.add_argument("--w_verified", type=float, default=0.0)
    parser.add_argument("--w_alignment", type=float, default=1.0)
    parser.add_argument("--w_length", type=float, default=0.0)
    parser.add_argument("--w_answer_pred", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--length_baseline", type=float, default=5.0)
    parser.add_argument("--lambda_penalty", type=float, default=0.1)
    parser.add_argument("--n_answer_tokens", type=int, default=10)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="grpo_curriculum")
    parser.add_argument("--save_steps", type=int, default=100)
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