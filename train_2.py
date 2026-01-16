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
import pdb

# Try importing Unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

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
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

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
        total_loss = jsd_per_position.sum()
        num_positions = mask.sum().clamp(min=1)
    else:
        total_loss = jsd_per_position.sum()
        num_positions = jsd_per_position.numel()

    return total_loss / num_positions

# -----------------------------------------------------------------------------
# Data Handling (Updated for JSONL support + Dataset Formatting)
# -----------------------------------------------------------------------------

@dataclass
class Example:
    prompt: str
    response: str

def format_arc_challenge(row) -> Tuple[str, str]:
    """Formats ARC-Challenge dictionary choices into a text prompt."""
    q = row.get("question", "")
    choices = row.get("choices", {})
    
    # Handle HuggingFace dataset struct format (list of labels, list of texts)
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])])
    else:
        # Fallback if raw json is different
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
    
    # 1. Load the raw dataset
    if os.path.isfile(path_or_name):
        if path_or_name.endswith(".jsonl"):
            # JSONL files are usually line-separated objects, no "field" needed
            ds = load_dataset("json", data_files={split: path_or_name}, split=split)
        else:
            # Standard JSON might be wrapped in "instances" (based on your original code)
            # We try loading with field first, fallback to standard if that fails
            try:
                ds = load_dataset("json", data_files={split: path_or_name}, field="instances", split=split)
            except Exception:
                print("Could not load with field='instances', trying flat JSON...")
                ds = load_dataset("json", data_files={split: path_or_name}, split=split)
    else:
        # Load from Hugging Face Hub
        ds = load_dataset(path_or_name, split=split)

    if limit:
        ds = ds.select(range(min(len(ds), limit)))

    # 2. Determine formatting strategy based on name/path
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
        # Generic fallback
        def generic_formatter(row):
            return row.get(prompt_col, ""), row.get(resp_col, "")
        formatter = generic_formatter
        print(f"-> Using generic format: {prompt_col} -> {resp_col}")

    # 3. Process examples
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
    # Training Collator: We want explicit input_ids for supervision.
    # Note: We do NOT rely on tokenizer padding here because we use pad_sequence later (which right-pads).
    
    # 1. Encode user prompt
    user_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", return_dict=True)
    
    # 2. Encode full conversation
    full = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], return_tensors="pt", return_dict=True)
    
    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]

    # 3. STRICT TRUNCATION
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
        # specific list to hold raw prompts
        raw_prompts = [] 
        
        for ex in batch:
            ids, attn, lbls = chat_template_pair(tokenizer, ex.prompt, ex.response, max_length)
            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(lbls)
            # Store the raw prompt
            raw_prompts.append(ex.prompt)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        # Return the dictionary with the new "prompts" key
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels,
            "prompts": raw_prompts
        }
    return collate

def generate_on_policy(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str], *, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0, use_unsloth: bool = False) -> List[str]:
    # GENERATION REQUIRES LEFT PADDING
    # We ensure tokenizer.padding_side is set to 'left' globally in main.
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in prompts],
        return_tensors="pt", 
        return_dict=True, 
        padding=True, # This will use tokenizer.padding_side (which is 'left')
        return_attention_mask=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    if use_unsloth and HAS_UNSLOTH:
        FastLanguageModel.for_inference(model) 
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=top_k > 0, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        FastLanguageModel.for_training(model) 
    else:
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=top_k > 0, top_k=top_k if top_k > 0 else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    responses = []
    for text, prompt in zip(decoded, prompts):
        if prompt in text: reply = text.split(prompt)[-1].strip()
        else: reply = text
        responses.append(reply)
    return responses

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
# Training Loop
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_unsloth and not HAS_UNSLOTH:
        raise ImportError("Unsloth requested but not installed.")

    if args.run_name is None:
        prefix = "unsloth" if args.use_unsloth else "hf"
        s_name = args.student_model.split("/")[-1]
        args.run_name = f"{prefix}_distill_{s_name}_lr{args.lr}"

    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    if args.use_wandb:
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": {"name": args.run_name, "id": None}})

    if accelerator.is_main_process:
        print(f"Teacher: {args.teacher_model} | Student: {args.student_model}")
        print(f"Unsloth Optimized: {args.use_unsloth} (4bit: {args.load_in_4bit})")

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    if args.use_unsloth:
        # Load Student
        student, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.student_model,
            max_seq_length=args.max_length,
            dtype=None, 
            load_in_4bit=args.load_in_4bit,
        )
        
        # Add Adapters
        student = FastLanguageModel.get_peft_model(
            student,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

        # Load Teacher
        teacher, _ = FastLanguageModel.from_pretrained(
            model_name=args.teacher_model,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit, 
        )
        FastLanguageModel.for_inference(teacher) 

    else:
        # Standard HF
        tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True, padding_side="left")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        student.resize_token_embeddings(len(tokenizer))
        teacher.eval()
        teacher.requires_grad_(False)

    # -------------------------------------------------------------------------
    # IMPORTANT: Tokenizer Config for Generation
    # -------------------------------------------------------------------------
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # FORCE LEFT PADDING for generation
    # The collator uses pad_sequence which effectively right-pads for training (correct)
    # The generator uses apply_chat_template which respects this setting (correct)
    tokenizer.padding_side = "left"
    
    if accelerator.is_main_process:
        print(f"Tokenizer Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"Tokenizer Padding Side: {tokenizer.padding_side}")

    # -------------------------------------------------------------------------
    # Optimizer & Data
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    train_source = args.train_file if args.train_file else args.dataset_name
    val_source = args.val_file if args.val_file else args.dataset_name
    val_split = "train" if args.val_file else "test"

    train_examples = load_data_source(train_source, split="train", prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_train_samples)
    val_examples = load_data_source(val_source, split=val_split, prompt_col=args.prompt_column, resp_col=args.response_column, limit=args.max_val_samples)

    collate = collate_fn_builder(tokenizer, args.max_length)
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    student, teacher, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        student, teacher, optimizer, train_dataloader, val_dataloader
    )

    global_step = 0
    total_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm.tqdm(range(total_steps), disable=not accelerator.is_main_process)

    for epoch in range(args.num_epochs):
        student.train()
        for batch in train_dataloader:
            with accelerator.accumulate(student):
                loss_dict = {}
                
                # --- On-Policy Branch ---
                if random.random() < args.lambda_on_policy:
                    # FIX: Use the prompts from the CURRENT batch, not random samples
                    prompts = batch["prompts"]

                    with torch.no_grad():
                        # Generate responses using the student model on the current prompts
                        student_responses = generate_on_policy(
                            accelerator.unwrap_model(student),
                            tokenizer,
                            prompts,
                            max_new_tokens=args.max_new_tokens,
                            temperature=1.0, # You might want to use args.temperature here
                            use_unsloth=args.use_unsloth
                        )
                    
                    # Create new temporary examples using the generated responses
                    new_examples = [Example(prompt=p, response=r) for p, r in zip(prompts, student_responses)]
                    
                    # Re-collate these new examples into tensors
                    # Note: We reuse the existing 'collate' function defined earlier
                    new_batch = collate(new_examples)
                    
                    input_ids = new_batch["input_ids"].to(student.device)
                    attention_mask = new_batch["attention_mask"].to(student.device)
                    labels = new_batch["labels"].to(student.device)

                    # Compute logits on the generated (on-policy) data
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
                    # Use the ground truth data provided in the original batch
                    # Note: We must exclude "prompts" from the kwargs passed to the model
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
                
                if accelerator.is_main_process:
                    loss_dict["loss/total"] = loss.item()
                    loss_dict["lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(loss_dict, step=global_step)

                if global_step % args.eval_steps == 0:
                    val_ce_loss = evaluate_student(student, val_dataloader, accelerator)
                    if accelerator.is_main_process:
                        print(f" Step {global_step} | Val CE Loss: {val_ce_loss:.4f}")
                        accelerator.log({"val/ce_loss": val_ce_loss}, step=global_step)

    if accelerator.is_main_process:
        output_dir = f"distilled_model_{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        if args.use_unsloth:
            accelerator.unwrap_model(student).save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            accelerator.unwrap_model(student).save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
        if args.use_wandb: accelerator.end_training()

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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--lambda_on_policy", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="distillation-project")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--use_unsloth", type=str2bool, default=True)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(args, key): setattr(args, key, value)

    if isinstance(args.lr, str): args.lr = float(args.lr)
    if isinstance(args.beta, str): args.beta = float(args.beta)
    if isinstance(args.temperature, str): args.temperature = float(args.temperature)
    if isinstance(args.lambda_on_policy, str): args.lambda_on_policy = float(args.lambda_on_policy)
    if isinstance(args.use_wandb, str): args.use_wandb = str2bool(args.use_wandb)
    if isinstance(args.use_unsloth, str): args.use_unsloth = str2bool(args.use_unsloth)
    if isinstance(args.load_in_4bit, str): args.load_in_4bit = str2bool(args.load_in_4bit)

    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
