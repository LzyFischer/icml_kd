from unsloth import FastLanguageModel
import os
import torch
import torch.nn.functional as F
import argparse
import yaml
import warnings
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import math

# Import your custom data loader
from load_data import load_data_source
from utils import generalized_jsd_loss
import pdb

# Suppress warnings
warnings.filterwarnings("ignore")

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
        min_pct = 0.4
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
    temperature: float = 1.0
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
    # generalized_jsd_loss handles softmax/log_softmax and temperature internally
    # and returns summed loss per position (reduction="none")
    loss_per_position = generalized_jsd_loss(
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
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute JSD based on top-1 predictions.
    To make JSD meaningful for Top-1, we treat the Teacher as having 
    a 'Hard' (One-Hot) distribution at the argmax index.
    """
    # Identify teacher targets
    teacher_indices = teacher_logits.argmax(dim=-1).unsqueeze(-1)

    # Create a "hard" teacher logit tensor to simulate One-Hot distribution
    # Set non-targets to very low, target to very high
    hard_teacher_logits = torch.full_like(teacher_logits, -1e4)
    hard_teacher_logits.scatter_(-1, teacher_indices, 1e4)

    # Compute JSD between Student (full smooth distribution) and Teacher (hard distribution)
    loss_per_position = generalized_jsd_loss(
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
    k: int = 10
) -> torch.Tensor:
    """
    Compute alignment loss based on curriculum mode.
    
    Args:
        student_logits: [batch, seq_len, vocab]
        teacher_logits: [batch, seq_len, vocab]
        mask: [batch, seq_len]
        mode: one of ["ce", "top1_kl", "topk_kl", "full_kl"]
    """
    if mode == "ce":
        # Standard Cross Entropy (Hard Target KL)
        teacher_preds = teacher_logits.argmax(dim=-1)
        loss_per_token = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_preds.view(-1),
            reduction='none'
        ).view(student_logits.shape[:-1])
    
    elif mode == "top1_kl":
        # JSD against Hard Teacher Target
        loss_per_token = compute_top1_kl(
            student_logits, 
            teacher_logits, 
            beta=beta, 
            temperature=temperature
        )
    
    elif mode == "topk_kl":
        # JSD on Top-K subset
        loss_per_token = compute_topk_kl(
            student_logits, 
            teacher_logits, 
            k=k, 
            beta=beta, 
            temperature=temperature
        )
    
    else:  # full_kl
        # Standard JSD on full vocabulary
        loss_per_token = generalized_jsd_loss(
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
    """
    Select top-k tokens with highest KL for alignment.
    
    Args:
        kl_per_token: [batch, seq_len]
        mask: [batch, seq_len]
        top_percentage: percentage of tokens to keep (0-1)
    
    Returns:
        new_mask: [batch, seq_len] with only top-k tokens marked
    """
    batch_size, seq_len = kl_per_token.shape
    
    # Mask out padding tokens by setting to -inf
    masked_kl = kl_per_token.clone()
    masked_kl[mask == 0] = -float('inf')
    
    # Calculate number of tokens to keep per sequence
    valid_tokens = mask.sum(dim=1)  # [batch]
    k_per_seq = (valid_tokens * top_percentage).long().clamp(min=1)
    
    # Select top-k tokens per sequence
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
    """
    Compute soft "first understanding time" using similarity with correct answer.
    
    Args:
        student_hidden_states: [batch, seq_len, hidden_dim]
        reference_embedding: [batch, hidden_dim] - embedding of correct answer
        mask: [batch, seq_len]
        tau: threshold for understanding
        gamma: temperature parameter
    
    Returns:
        t_soft: [batch] - soft understanding time
        similarities: [batch, seq_len] - similarity scores
    """
    # Normalize embeddings
    student_norm = F.normalize(student_hidden_states, dim=-1)
    reference_norm = F.normalize(reference_embedding.unsqueeze(1), dim=-1)
    
    # Compute similarities [batch, seq_len]
    similarities = (student_norm * reference_norm).sum(dim=-1)
    
    # Compute soft probabilities: p_i = σ(γ(s_i - τ))
    p_i = torch.sigmoid(gamma * (similarities - tau))
    
    # Apply mask
    p_i = p_i * mask
    
    # Compute q_i = p_i * Π_{j<i}(1 - p_j)
    # This is the probability of "first understanding" at position i
    batch_size, seq_len = p_i.shape
    q_i = torch.zeros_like(p_i)
    
    for i in range(seq_len):
        if i == 0:
            q_i[:, i] = p_i[:, i]
        else:
            # Product of (1 - p_j) for j < i
            prod = torch.ones(batch_size, device=p_i.device)
            for j in range(i):
                prod = prod * (1 - p_i[:, j])
            q_i[:, i] = p_i[:, i] * prod
    
    # Compute soft time: t_soft = Σ_i i * q_i
    positions = torch.arange(seq_len, device=p_i.device).float().unsqueeze(0)  # [1, seq_len]
    t_soft = (positions * q_i).sum(dim=1)  # [batch]
    
    return t_soft, similarities

def compute_length_regularization_reward(
    t_soft: torch.Tensor,
    actual_length: torch.Tensor,
    b: float = 5.0,
    lambda_penalty: float = 0.1
) -> torch.Tensor:
    """
    Compute length regularization reward.
    
    R = 1/log(2 + t_soft) - λ * softplus(L - t_soft - b)
    
    Encourages:
    - Early understanding (small t_soft)
    - Not too verbose (L close to t_soft + b)
    """
    # Early understanding reward
    early_reward = 1.0 / torch.log(2.0 + t_soft)
    
    # Length penalty
    excess_length = actual_length - t_soft - b
    length_penalty = lambda_penalty * F.softplus(excess_length)
    
    total_reward = early_reward - length_penalty
    
    return total_reward

def compute_answer_prediction_reward(
    student_model,
    tokenizer,
    reasoning_text: List[str],
    ground_truth_text: List[str],
    n_answer_tokens: int = 20,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute reward based on student's ability to predict answer given reasoning.
    
    Args:
        student_model: The student model
        tokenizer: Tokenizer
        reasoning_text: List of reasoning texts (teacher's output minus last n tokens)
        ground_truth_text: List of complete correct answers
        n_answer_tokens: Number of tokens to consider as "answer"
    
    Returns:
        rewards: [batch] similarity between generated and correct answer
    """
    batch_size = len(reasoning_text)
    rewards = []
    
    # Get embeddings for ground truth answers
    with torch.no_grad():
        # Tokenize ground truth
        gt_inputs = tokenizer(
            ground_truth_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Get last n tokens of ground truth as "answer section"
        gt_answer_ids = gt_inputs.input_ids[:, -n_answer_tokens:]
        
        # Get student embeddings for ground truth answers
        gt_outputs = student_model(
            input_ids=gt_inputs.input_ids,
            output_hidden_states=True
        )
        gt_embeddings = gt_outputs.hidden_states[-1][:, -n_answer_tokens:, :]  # [batch, n, hidden]
        gt_answer_embedding = gt_embeddings.mean(dim=1)  # [batch, hidden]
        
        # Generate from reasoning
        reasoning_inputs = tokenizer(
            reasoning_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate n tokens
        generated = student_model.generate(
            input_ids=reasoning_inputs.input_ids,
            attention_mask=reasoning_inputs.attention_mask,
            max_new_tokens=n_answer_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Get embeddings for generated answer
        gen_outputs = student_model(
            input_ids=generated,
            output_hidden_states=True
        )
        gen_embeddings = gen_outputs.hidden_states[-1][:, -n_answer_tokens:, :]
        gen_answer_embedding = gen_embeddings.mean(dim=1)  # [batch, hidden]
        
        # Compute similarity
        similarity = F.cosine_similarity(
            gen_answer_embedding,
            gt_answer_embedding,
            dim=-1
        )  # [batch]
    
    return similarity

# -----------------------------------------------------------------------------
# Main Reward Computation with Curriculum
# -----------------------------------------------------------------------------

class CurriculumRewardFunction:
    """
    Comprehensive reward function with curriculum learning.
    """
    
    def __init__(
        self,
        student_model,
        teacher_model,  # Added teacher model
        tokenizer,
        curriculum_scheduler: CurriculumScheduler,
        device: str = "cuda",
        # Reward weights
        w_verified: float = 1.0,
        w_alignment: float = 0.5,
        w_token_selection: float = 0.3,
        w_length: float = 0.2,
        w_answer_pred: float = 0.3,
        # Hyperparameters
        tau: float = 0.7,
        gamma: float = 10.0,
        b: float = 5.0,
        lambda_penalty: float = 0.1,
        n_answer_tokens: int = 20
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.scheduler = curriculum_scheduler
        self.device = device
        
        # Weights (use multiplication to combine)
        self.w_verified = w_verified
        self.w_alignment = w_alignment
        self.w_token_selection = w_token_selection
        self.w_length = w_length
        self.w_answer_pred = w_answer_pred
        
        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.b = b
        self.lambda_penalty = lambda_penalty
        self.n_answer_tokens = n_answer_tokens
        
        # Add __name__ attribute for GRPOTrainer compatibility
        self.__name__ = "curriculum_reward"
    
    def __call__(
        self,
        prompts: List[List[Dict]],
        completions: List[List[Dict]],
        reference: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute comprehensive reward with curriculum learning.
        
        Returns rewards scaled to reasonable range [0, 10].
        """
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        # Handle different input formats
        if reference is None:
            # Try to get from kwargs
            reference = kwargs.get('answer', kwargs.get('answers', []))
            if not reference:
                print("Warning: No reference answers provided, using empty strings")
                reference = [""] * len(prompts)
        
        batch_size = len(prompts)
        
        # Extract text from chat format
        prompt_texts = []
        completion_texts = []
        
        for p, c in zip(prompts, completions):
            # Handle both dict and string formats
            if isinstance(p, list) and len(p) > 0:
                if isinstance(p[0], dict):
                    prompt_texts.append(p[0].get('content', ''))
                else:
                    prompt_texts.append(str(p[0]))
            else:
                prompt_texts.append(str(p))
            
            if isinstance(c, list) and len(c) > 0:
                if isinstance(c[0], dict):
                    completion_texts.append(c[0].get('content', ''))
                else:
                    completion_texts.append(str(c[0]))
            else:
                completion_texts.append(str(c))
        
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        
        # Get current curriculum settings
        alignment_mode = self.scheduler.get_alignment_mode()
        token_percentage = self.scheduler.get_token_percentage()
        
        # =====================================================================
        # 1. Verified Reward (Answer Match)
        # =====================================================================
        verified_rewards = []
        for completion, ref in zip(completion_texts, reference):
            clean_completion = completion.strip().lower()
            clean_ref = ref.strip().lower()
            verified_rewards.append(1.0 if clean_ref in clean_completion else 0.0)
        verified_rewards = torch.tensor(verified_rewards, device=self.device)
        
        # =====================================================================
        # 2. Get Model Outputs (separate forward passes for efficiency)
        # =====================================================================
        try:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Reference answer tokenization for embeddings
                ref_inputs = self.tokenizer(
                    reference,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Student forward pass with hidden states
                student_outputs = self.student_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                student_hidden = student_outputs.hidden_states[-1]  # [batch, seq, hidden]
                student_logits = student_outputs.logits

                # Get reference embeddings
                ref_outputs = self.student_model(
                    input_ids=ref_inputs.input_ids,
                    output_hidden_states=True
                )
                ref_embeddings = ref_outputs.hidden_states[-1].mean(dim=1)  # [batch, hidden]
                
                # Teacher forward pass with logits (REAL IMPLEMENTATION)
                teacher_outputs = self.teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False # We just need logits
                )
                teacher_logits = teacher_outputs.logits
                
                mask = inputs.attention_mask.float()
                seq_lengths = mask.sum(dim=1)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return neutral rewards if forward pass fails
            return [1.0] * batch_size
        
        # =====================================================================
        # 3. Curriculum-based Alignment Reward
        # =====================================================================
        alignment_loss = compute_alignment_loss(
            student_logits,
            teacher_logits,
            mask,
            mode=alignment_mode,
            beta=0.5,
            temperature=1.0
        )  # [batch, seq_len]
        
        # =====================================================================
        # 4. Token Selection Curriculum
        # =====================================================================
        # Select top-k tokens by KL magnitude
        token_mask = select_topk_tokens_by_kl(
            alignment_loss,
            mask,
            top_percentage=token_percentage
        )
        
        # Compute masked alignment reward
        masked_alignment = (alignment_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
        alignment_rewards = -masked_alignment  # Negative loss as reward
        
        # =====================================================================
        # 5. Length Regularization (Student Understanding Time)
        # =====================================================================
        t_soft, similarities = compute_student_understanding_time(
            student_hidden,
            ref_embeddings,
            mask,
            tau=self.tau,
            gamma=self.gamma
        )
        
        length_rewards = compute_length_regularization_reward(
            t_soft,
            seq_lengths,
            b=self.b,
            lambda_penalty=self.lambda_penalty
        )
        
        # =====================================================================
        # 6. Answer Prediction Reward (Simplified for efficiency)
        # =====================================================================
        # Instead of generating, we compute embedding similarity directly
        # Extract last n tokens as "answer section"
        answer_pred_rewards = torch.zeros(batch_size, device=self.device)
        
        try:
            for i in range(batch_size):
                seq_len = int(seq_lengths[i].item())
                if seq_len > self.n_answer_tokens:
                    # Get embeddings of last n tokens
                    answer_start = seq_len - self.n_answer_tokens
                    answer_embeddings = student_hidden[i, answer_start:seq_len, :]
                    answer_embedding = answer_embeddings.mean(dim=0)  # [hidden]
                    
                    # Compare with reference
                    similarity = F.cosine_similarity(
                        answer_embedding.unsqueeze(0),
                        ref_embeddings[i].unsqueeze(0),
                        dim=-1
                    )
                    answer_pred_rewards[i] = similarity.item()
                else:
                    # Too short, use moderate reward
                    answer_pred_rewards[i] = 0.5
        except Exception as e:
            print(f"Error in answer prediction: {e}")
            answer_pred_rewards = torch.ones(batch_size, device=self.device) * 0.5
        
        # =====================================================================
        # 7. Combine Rewards (using multiplication for scale invariance)
        # =====================================================================
        # Normalize each component to [0, 1] range
        alignment_norm = torch.exp(alignment_rewards * 0.1)
        length_norm = torch.exp(length_rewards * 0.1)
        answer_norm = (answer_pred_rewards + 1) / 2  # Cosine similarity to [0,1]
        
        # Multiplicative combination (geometric mean style)
        complex_reward = (
            (alignment_norm * self.w_alignment) +
            (length_norm * self.w_length) +
            (answer_norm * self.w_answer_pred)
        )
        # Final reward: verified * complex (multiplicative gating)
        # Scale to [0, 10] range
        final_reward = 4.0 * verified_rewards * complex_reward
        
        # If not verified, still give partial credit from complex reward
        final_reward = torch.where(
            verified_rewards > 0.5,
            final_reward,
            complex_reward * 1  # Scaled down for incorrect answers
        )
        
        return final_reward.cpu().tolist()

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def train_teacher(args):
    # 1. Configuration
    max_seq_length = args.max_length
    lora_rank = args.lora_r
    
    # Estimate total steps for curriculum
    raw_examples = load_data_source(
        args.train_file if args.train_file else args.dataset_name,
        split="train",
        limit=args.max_train_samples
    )
    num_examples = len(raw_examples)
    steps_per_epoch = num_examples // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    
    # Initialize curriculum scheduler
    curriculum_scheduler = CurriculumScheduler(
        total_steps=total_steps,
        warmup_steps=args.curriculum_warmup_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Curriculum warmup steps: {args.curriculum_warmup_steps}")
    
    # 2. Load Teacher Model (The model being trained)
    print(f"Loading Trainer Model: {args.teacher_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6, # Adjusted for multiple models
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    
    # 3. Load Student Model for Reward Calculation
    print(f"Loading Student Model for Rewards: {args.student_model}")
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        gpu_memory_utilization=0.2, # Allocate smaller memory chunk
    )
    FastLanguageModel.for_inference(student_model)
    
    # 5. Prepare Dataset
    dataset_dict = {
        "prompt": [],
        "reference": []  # Changed from "answer" to "reference"
    }
    
    system_prompt = "You are a helpful assistant. Please reason step by step."
    
    for ex in raw_examples:
        dataset_dict["prompt"].append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.prompt}
        ])
        dataset_dict["reference"].append(ex.response)
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # 6. Initialize Reward Function
    reward_function = CurriculumRewardFunction(
        student_model=student_model,
        teacher_model=model, # Pass the real teacher model instance
        tokenizer=student_tokenizer,
        curriculum_scheduler=curriculum_scheduler,
        device="cuda",
        w_verified=args.w_verified,
        w_alignment=args.w_alignment,
        w_token_selection=args.w_token_selection,
        w_length=args.w_length,
        w_answer_pred=args.w_answer_pred,
        tau=args.tau,
        gamma=args.gamma,
        b=args.length_baseline,
        lambda_penalty=args.lambda_penalty,
        n_answer_tokens=args.n_answer_tokens
    )
    
    # 7. Training Configuration
    training_args = GRPOConfig(
        output_dir=f"ckpts/{args.run_name}",
        run_name=args.run_name,
        learning_rate=args.teacher_lr,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
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
    
    # 8. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=hf_dataset,
    )
    
    # Custom callback to update curriculum and log metrics
    from transformers import TrainerCallback
    
    class CurriculumCallback(TrainerCallback):
        def __init__(self, scheduler):
            self.scheduler = scheduler
        
        def on_step_end(self, args, state, control, **kwargs):
            self.scheduler.step()
            
            # Log curriculum state every 10 steps
            if state.global_step % 10 == 0:
                mode = self.scheduler.get_alignment_mode()
                token_pct = self.scheduler.get_token_percentage()
                progress = self.scheduler.get_progress()
                
                print(f"\n[Curriculum] Step {state.global_step}:")
                print(f"  Progress: {progress:.2%}")
                print(f"  Alignment Mode: {mode}")
                print(f"  Token Percentage: {token_pct:.2%}")
                
                # Log to wandb if available
                if args.report_to == "wandb":
                    try:
                        import wandb
                        wandb.log({
                            "curriculum/progress": progress,
                            "curriculum/alignment_mode": ["ce", "top1_kl", "topk_kl", "full_kl"].index(mode),
                            "curriculum/token_percentage": token_pct,
                        }, step=state.global_step)
                    except:
                        pass
    
    trainer.add_callback(CurriculumCallback(curriculum_scheduler))
    
    # 9. Train
    print("Starting GRPO Training with Curriculum Learning...")
    trainer.train()
    
    # Save
    model.save_lora(f"ckpts/{args.run_name}/final_lora")
    print(f"Training complete! Model saved to ckpts/{args.run_name}/final_lora")

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    
    # Model paths
    parser.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    
    # Data
    parser.add_argument("--train_file", type=str, default="./data/date/train.jsonl")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--response_column", type=str, default="response")
    parser.add_argument("--max_train_samples", type=int, default=None)
    
    # Training params
    parser.add_argument("--teacher_lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_teacher_samples", type=int, default=4)
    
    # Curriculum params
    parser.add_argument("--curriculum_warmup_steps", type=int, default=0)
    
    # Reward weights
    parser.add_argument("--w_verified", type=float, default=1.0)
    parser.add_argument("--w_alignment", type=float, default=0.8)
    parser.add_argument("--w_token_selection", type=float, default=0.05)
    parser.add_argument("--w_length", type=float, default=0.05)
    parser.add_argument("--w_answer_pred", type=float, default=0.15)
    
    # Length regularization params
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--length_baseline", type=float, default=5.0)
    parser.add_argument("--lambda_penalty", type=float, default=0.1)
    parser.add_argument("--n_answer_tokens", type=int, default=10)
    
    # LoRA/Unsloth
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    
    # Logging
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="grpo_curriculum")
    parser.add_argument("--save_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    # Load config if exists
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