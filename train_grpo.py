from unsloth import FastLanguageModel
import os
import torch
import argparse
import yaml
import warnings
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your custom data loader
from load_data import load_data_source
import pdb

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------------------------

def get_correctness_reward(prompts, completions, answer, **kwargs):
    """
    Reward function checking if the completion matches the ground truth.
    Adapted to handle your specific dataset format.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        # Simple exact match or subset match
        # You can enhance this with the Regex logic from your test.py if using GSM8K
        clean_completion = completion[0]['content'].strip().lower()
        clean_gt = ground_truth.strip().lower()
        
        if clean_gt in clean_completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

class StudentDifficultyReward:
    """
    Custom callable class to maintain the state of the Student Model
    without reloading it every step.
    """
    def __init__(self, student_model_name, device="cuda"):
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Ensure logits are returned
        print(f"Loading Student Model for Reward Calculation: {student_model_name}")
        # Load Student in 4bit to save VRAM since it's only for inference
        self.student, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=student_model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(self.student)
        self.device = device

    def __call__(self, prompts, completions, **kwargs):
        """
        Calculates how 'hard' the completion is for the student.
        Higher Student Loss (Perplexity) => Higher Difficulty Reward.
        """
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Ensure logits are returned
        rewards = []
        
        # We need to process this batch. 
        # Note: This runs in the Python loop, so batching here is manual.
        # For maximum speed, we process the list directly.
        
        # Format inputs: Prompt + Generated Response
        full_texts = [p[0]['content'] + c[0]['content'] for p, c in zip(prompts, completions)]
        
        inputs = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.student(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids # Calculate loss on the sequence
            )
            # The model returns mean loss per batch usually, but we need per-sample.
            # So we manually compute cross entropy per row if needed, 
            # OR simplistic approach: Use the loss directly if batch_size=1, 
            # but here we approximate "Difficulty" via Negative Log Likelihood
            
            # Accurate per-sample loss calculation:
            logits = outputs.logits[:, :-1, :]
            labels = inputs.input_ids[:, 1:]
            
            # Cross Entropy without reduction
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss.view(labels.size())
            
            # Average loss per sequence (Perplexity-ish)
            # Mask padding
            mask = inputs.attention_mask[:, 1:]
            active_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
            
            # Reward logic: 
            # If Student Loss is High -> Difficulty is High -> Reward is Positive
            # You can scale this by args.difficulty_weight
            difficulty_scores = active_loss.detach().float().cpu().numpy().tolist()
            
        return difficulty_scores

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def train_teacher(args):
    # 1. Configuration
    max_seq_length = args.max_length
    lora_rank = args.lora_r

    # 2. Load Teacher Model (The one being trained) using Unsloth
    print(f"Loading Teacher Model: {args.teacher_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.teacher_model,
        max_seq_length = max_seq_length,
        load_in_4bit = args.load_in_4bit, 
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.9, # Leave room for Student Model
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    # 3. Data Preparation
    # Load your custom Example objects
    raw_examples = load_data_source(
        args.train_file if args.train_file else args.dataset_name, 
        split="train", 
        prompt_col=args.prompt_column, 
        resp_col=args.response_column, 
        limit=args.max_train_samples
    )

    # Convert to HuggingFace Dataset format required by TRL
    # GRPOTrainer expects a dataset with columns "prompt" and "answer" (or similar)
    dataset_dict = {
        "prompt": [],
        "answer": []
    }
    
    # Pre-format prompts using the chat template structure
    # The GRPOTrainer sends these to the model
    system_prompt = "You are a helpful assistant. Please reason step by step."
    
    for ex in raw_examples:
        # Structure for Chat Templates
        dataset_dict["prompt"].append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.prompt}
        ])
        dataset_dict["answer"].append(ex.response)

    hf_dataset = Dataset.from_dict(dataset_dict)

    # 4. Initialize Reward Functions
    # Initialize the student model wrapper
    student_difficulty_scorer = StudentDifficultyReward(args.student_model)
    
    # Define the lambda for difficulty to apply weight
    def weighted_difficulty(prompts, completions, **kwargs):
        scores = student_difficulty_scorer(prompts, completions, **kwargs)
        return [s * args.difficulty_weight for s in scores]

    # 5. Training Arguments (GRPOConfig)
    training_args = GRPOConfig(
        output_dir = f"ckpts/{args.run_name}",
        run_name = args.run_name,
        learning_rate = args.teacher_lr,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 4, 
        num_generations = args.num_teacher_samples, # trajectories per prompt
        max_prompt_length = max_seq_length // 2,
        max_completion_length = args.max_new_tokens,
        num_train_epochs = args.num_epochs,
        save_steps = args.save_steps if args.save_steps > 0 else 100,
        max_grad_norm = 1.0,
        report_to = "wandb" if args.use_wandb else "none",
        # use_vllm = True, # CRITICAL for speedup
        # vllm_gpu_memory_utilization = 0.25, # Adjust based on VRAM (Student + Teacher + VLLM needs to fit)
    )

    # 6. Trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_correctness_reward, # Checks ground truth
            weighted_difficulty     # Checks student difficulty
        ],
        args = training_args,
        train_dataset = hf_dataset,
    )

    # 7. Train
    print("Starting GRPO Training...")
    trainer.train()
    
    # Save
    model.save_lora(f"ckpts/{args.run_name}/final_lora")

# -----------------------------------------------------------------------------
# Argument Parsing (Kept similar to your original)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    
    # Defaults (can be overridden by config file)
    parser.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="./data/date/train.jsonl")
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--response_column", type=str, default="response")
    
    # Training Params
    parser.add_argument("--teacher_lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_teacher_samples", type=int, default=4)
    parser.add_argument("--difficulty_weight", type=float, default=0.5)
    
    # LoRA/Unsloth
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    
    # Logging
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="grpo_teacher_run")
    parser.add_argument("--save_steps", type=int, default=20)
    
    # Max samples
    parser.add_argument("--max_train_samples", type=int, default=None)

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