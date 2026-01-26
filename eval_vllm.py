"""
Evaluation script for distilled models using vLLM.
Strictly loads local datasets from a 'data/' directory.
Saves results to a structured output directory.

Directory Structure Expectation:
    data/
    ├── gsm8k/
    │   ├── test.jsonl
    │   └── ...

Output Structure:
    output_dir/
    ├── gsm8k/
    │   └── model_name/
    │       └── eval_results.json

Usage:
    python eval.py --model_path /path/to/model --datasets gsm8k arc_challenge --output_dir results
"""

import argparse
import json
import os
import re
import shutil
import gc
import tempfile
import glob
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fractions import Fraction

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer
from peft import PeftModel

# -----------------------------------------------------------------------------
# 1. Import Utilities & Formatters
# -----------------------------------------------------------------------------

try:
    from load_data import (
        format_anli, format_arc_challenge, format_commonsense_qa,
        format_date, format_gsm8k, format_math, format_strategy_qa,
        format_table_mwp
    )
    print("Successfully imported formatters from load_data.py.")
except ImportError:
    print("CRITICAL WARNING: Could not import formatters. Please ensure 'load_data.py' is present.")
    def dummy_fmt(x): return x.get('prompt', ''), x.get('response', '')
    format_anli = format_arc_challenge = format_commonsense_qa = dummy_fmt
    format_date = format_gsm8k = format_math = dummy_fmt
    format_strategy_qa = format_table_mwp = dummy_fmt

os.environ["VLLM_USE_V1"] = "1"

# -----------------------------------------------------------------------------
# 2. Advanced Math Normalization
# -----------------------------------------------------------------------------

def normalize_math_answer(text: str) -> str:
    """
    Normalize mathematical answers to handle various formatting differences.
    
    Handles:
    - LaTeX formatting (\\frac, \\text, \\sqrt, etc.)
    - Text units (students, cm, km, degrees, etc.)
    - Spacing differences
    - Fraction representations (1/2 vs \\frac{1}{2})
    - Interval notation
    - Degree symbols
    - List formatting
    """
    if not text or text == "N/A":
        return text
    
    original = text
    text = str(text).strip()
    
    # Remove common LaTeX commands and their braces
    latex_patterns = [
        (r'\\text\{[^}]*\}', ''),  # Remove text units like \text{ cm}
        (r'\\,', ''),               # Thin space
        (r'\\;', ''),               # Medium space
        (r'\\:', ''),               # Medium space
        (r'\\!', ''),               # Negative thin space
        (r'\\ ', ' '),              # Escaped space
        (r'\\mathrm\{([^}]*)\}', r'\1'),  # \mathrm{text} -> text
        (r'\\operatorname\{([^}]*)\}', r'\1'),  # \operatorname{text} -> text
    ]
    
    for pattern, replacement in latex_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Remove text units (students, cm, km, degrees, etc.)
    text = re.sub(r'\s*\\text\s*\{[^}]*\}', '', text)
    text = re.sub(r'\s+(students?|cm|km|meters?|m|ft|inches?|in|degrees?|°|hours?|minutes?|seconds?|years?|days?)\b', '', text, flags=re.IGNORECASE)
    
    # Normalize degree symbols
    text = re.sub(r'\\circ|°|degrees?', '', text, flags=re.IGNORECASE)
    
    # Convert \frac{a}{b} to a/b
    def replace_frac(match):
        num = match.group(1)
        denom = match.group(2)
        return f"{num}/{denom}"
    
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    text = re.sub(r'\\dfrac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    text = re.sub(r'\\tfrac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    
    # Remove remaining backslashes (for incomplete LaTeX)
    text = text.replace('\\', '')
    
    # Normalize spacing around operators and brackets
    text = re.sub(r'\s*([,\[\]\(\)])\s*', r'\1', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to evaluate fractions to decimals for comparison
    # This handles cases like "1/4" vs "0.25"
    try:
        # Check if it's a simple fraction
        if '/' in text and not any(c in text for c in ['[', ']', '(', ')', ',', 'sqrt', 'x', 'in']):
            parts = text.split('/')
            if len(parts) == 2:
                frac = Fraction(text)
                # Return both fraction and decimal forms for matching
                decimal = float(frac)
                # Store as tuple internally for flexible matching
                return f"{text}|{decimal}"
    except:
        pass
    
    # For interval notation, normalize spacing: x in [-2, 7] -> [-2,7]
    if 'in' in text.lower():
        # Extract just the interval part
        interval_match = re.search(r'[\[\(][-\d.,\s]+[\]\)]', text)
        if interval_match:
            interval = interval_match.group(0)
            interval = re.sub(r'\s*,\s*', ',', interval)
            return interval
    
    return text

def math_answers_equal(pred: str, ref: str) -> bool:
    """
    Compare two mathematical answers with robust normalization.
    """
    if not pred or not ref or pred == "N/A":
        return False
    
    # Normalize both
    norm_pred = normalize_math_answer(pred)
    norm_ref = normalize_math_answer(ref)
    
    # Direct string match after normalization
    if norm_pred == norm_ref:
        return True
    
    # Handle fraction/decimal equivalence
    if '|' in norm_pred or '|' in norm_ref:
        pred_forms = norm_pred.split('|')
        ref_forms = norm_ref.split('|')
        
        for pf in pred_forms:
            for rf in ref_forms:
                try:
                    if abs(float(pf) - float(rf)) < 1e-6:
                        return True
                except:
                    if pf == rf:
                        return True
    
    # Try numeric comparison
    try:
        # Extract all numbers and compare
        pred_nums = re.findall(r'-?\d+\.?\d*', norm_pred)
        ref_nums = re.findall(r'-?\d+\.?\d*', norm_ref)
        
        if pred_nums and ref_nums and len(pred_nums) == len(ref_nums):
            all_match = True
            for p, r in zip(pred_nums, ref_nums):
                try:
                    if abs(float(p) - float(r)) > 1e-6:
                        all_match = False
                        break
                except:
                    if p != r:
                        all_match = False
                        break
            if all_match:
                return True
    except:
        pass
    
    # Fallback: strip all non-alphanumeric and compare
    pred_clean = re.sub(r'[^a-zA-Z0-9.]', '', norm_pred)
    ref_clean = re.sub(r'[^a-zA-Z0-9.]', '', norm_ref)
    
    return pred_clean == ref_clean

# -----------------------------------------------------------------------------
# 3. Extraction Logic
# -----------------------------------------------------------------------------

def extract_last_boxed(text: str) -> Optional[str]:
    """Extracts the content of the LAST \\boxed{...}."""
    box_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    for start_idx in reversed(box_starts):
        open_braces = 1
        content_start = start_idx + 7
        for i in range(content_start, len(text)):
            char = text[i]
            if char == '{': open_braces += 1
            elif char == '}': open_braces -= 1
            if open_braces == 0: return text[content_start:i]
    return None

def extract_mcq_prediction(text: str) -> str:
    """
    Extract MCQ answer (A-F).
    Handles:
    - "The best/correct/final answer is (A)"
    - "The answer is **A**"
    - "The option is C."
    - "Answer: A"
    - Fallback to last bracketed/bolded letter
    """
    # 1. Broad "Answer is" pattern (covers "best answer", "correct answer", "option is")
    # We use findall and take the LAST match to handle chain-of-thought reasoning 
    # where the model might discuss why "A is wrong" before concluding "B is correct".
    # Regex breakdown:
    #   (?:answer|option|choice)  -> keyword
    #   (?: is)?                  -> optional verb (e.g., "The answer: A")
    #   (?:[:\s\*\(\[\{]*)        -> junk separators (spaces, colons, *, [, (, {)
    #   ([A-F])                   -> The target letter
    #   \b                        -> Word boundary (prevents matching "A" in "Apple")
    matches = re.findall(r"(?:answer|option|choice)(?: is)?\s*(?:[:\s\*\(\[\{]*)\s*([A-F])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # 2. Explicit "Answer:" header (common in some finetunes)
    match = re.search(r"Answer:\s*(?:[:\s\*\(\[\{]*)\s*([A-F])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 3. Fallback: Look for the last occurrence of specific patterns
    # Handles: (A), [A], {A}, **A**, **A.**
    matches = re.findall(r"(?:[\(\[\{]|\*\*)\s*([A-F])\s*(?:[\)\]\}]|\*\*|\.)", text)
    if matches:
        return matches[-1].upper()
        
    return "N/A"

def extract_yesno_prediction(text: str) -> str:
    """Extract Yes/No answer."""
    snippet = text[-50:].lower() if len(text) > 50 else text.lower()
    
    if "true" in snippet and "false" not in snippet: return "True"
    if "false" in snippet and "true" not in snippet: return "False"
    
    matches = re.findall(r"\b(true|false)\b", snippet)
    if matches:
        return matches[-1].capitalize()
        
    return "N/A"

def extract_anli_prediction(text: str) -> str:
    """Extract ANLI prediction (True/False/Neither)."""
    snippet = text[-100:].lower()
    
    if "true" in snippet or "entailment" in snippet or "yes" in snippet: return "True"
    if "false" in snippet or "contradiction" in snippet or "no" in snippet: return "False"
    if "neither" in snippet or "neutral" in snippet: return "Neither"
    
    return "N/A"

def extract_prediction(text: str, eval_type: str) -> str:
    """Extract prediction based on evaluation type."""
    if eval_type == "math":
        res = extract_last_boxed(text)
        if res: return res.strip()
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return nums[-1] if nums else "N/A"
        
    elif eval_type == "mcq":
        return extract_mcq_prediction(text)
        
    elif eval_type == "yesno":
        return extract_yesno_prediction(text)
        
    elif eval_type == "anli":
        return extract_anli_prediction(text)
        
    return text.strip()

# -----------------------------------------------------------------------------
# 4. Reference Extraction
# -----------------------------------------------------------------------------

def normalize_anli_ref(ref: str) -> str:
    """Normalize ANLI ground truth."""
    r = str(ref).lower().strip()
    if r in ["entailment", "0"]: return "True"
    if r in ["neutral", "1"]: return "Neither"
    if r in ["contradiction", "2"]: return "False"
    return ref.capitalize()

def extract_reference(row: Dict, eval_type: str) -> str:
    """
    Extract ground truth from dataset row.
    Handles distinct key names for different dataset types (e.g., 'label' for ANLI).
    """
    if eval_type == "mcq":
        ref = row.get("answerKey", "")
    elif eval_type == "anli":
        # ANLI uses 'label' and needs normalization (entailment -> True)
        raw_ref = row.get("label", row.get("answer", ""))
        ref = normalize_anli_ref(raw_ref)
    else:
        # Default fallback for Math/others usually found in 'answer'
        ref = str(row.get("answer", ""))
        
    return ref.strip()

# -----------------------------------------------------------------------------
# 5. Dataset Registry
# -----------------------------------------------------------------------------

DATASET_REGISTRY = {
    "gsm8k":          {"formatter": format_gsm8k,          "type": "math"},
    "math":           {"formatter": format_math,           "type": "math"},
    "table_mwp":      {"formatter": format_table_mwp,      "type": "math"},
    "arc_challenge":  {"formatter": format_arc_challenge,  "type": "mcq"},
    "commonsense_qa": {"formatter": format_commonsense_qa, "type": "mcq"},
    "date":           {"formatter": format_date,           "type": "mcq"},
    "strategy_qa":    {"formatter": format_strategy_qa,    "type": "yesno"},
    "anli":           {"formatter": format_anli,           "type": "anli"},
    "default": {
        "formatter": lambda x: (str(x.get('prompt', '')), str(x.get('response', ''))),
        "type": "text"
    }
}

@dataclass
class EvalExample:
    prompt: str
    reference: str
    metadata: Dict[str, Any]

# -----------------------------------------------------------------------------
# 6. Data Loading & Evaluation
# -----------------------------------------------------------------------------
def chat_template_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

def find_local_data_file(data_dir: str) -> Optional[str]:
    """Find test/validation file in dataset directory."""
    priority_files = ["test.jsonl", "test.json", "validation.jsonl", "validation.json", "val.jsonl"]
    for fname in priority_files:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path): return path
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    return jsonl_files[0] if jsonl_files else None

def load_local_benchmark(dataset_name: str, data_root: str = "data", limit: int = None):
    """Load dataset from local directory."""
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        potential_matches = [d for d in os.listdir(data_root) if dataset_name in d]
        if potential_matches:
            dataset_dir = os.path.join(data_root, potential_matches[0])
            print(f"Redirecting {dataset_name} -> {potential_matches[0]}")
        else:
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    data_file = find_local_data_file(dataset_dir)
    if not data_file:
        raise FileNotFoundError(f"No suitable .jsonl file found in {dataset_dir}")
    
    # print(f"Loading {dataset_name} from: {data_file}")
    # try:
    #     dataset = load_dataset("json", data_files={"test": data_file}, split="test")
    # except:
    #     dataset = load_dataset("json", data_files={"test": data_file}, split="test", field="instances")

    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        if data_file.endswith(".jsonl"):
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
    
    dataset = data

    if limit: dataset = dataset.select(range(min(len(dataset), limit)))

    conf = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY["default"])
    if dataset_name not in DATASET_REGISTRY:
        for k in DATASET_REGISTRY:
            if dataset_name in k:
                conf = DATASET_REGISTRY[k]
                break
    
    formatter = conf["formatter"]
    eval_type = conf["type"]
    
    examples = []
    for row in dataset:
        try:
            prompt, _ = formatter(row)
            reference = extract_reference(row, eval_type)
            examples.append(EvalExample(prompt=prompt, reference=reference, metadata=row))
        except Exception:
            continue
            
    return examples, eval_type

def evaluate_model(model: LLM, examples: List[EvalExample], eval_type: str, sampling_params: SamplingParams, tokenizer: PreTrainedTokenizer):
    """Evaluate model on dataset."""
    prompts = []
    for ex in examples:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            p = chat_template_prompt(tokenizer, ex.prompt)
        else:
            p = ex.prompt
        prompts.append(p)
    outputs = model.generate(prompts, sampling_params)
    
    results = []
    correct_count = 0
    
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text
        prediction = extract_prediction(generated_text, eval_type)
        reference = ex.reference
        
        # Check correctness with improved math handling
        is_correct = False
        
        if eval_type == "math":
            is_correct = math_answers_equal(prediction, reference)
        else:
            # Case-insensitive string match for MCQ, YESNO, ANLI
            is_correct = str(prediction).lower().strip() == str(reference).lower().strip()
            
        if is_correct: 
            correct_count += 1
            
        results.append({
            "prompt": ex.prompt,
            "generated": generated_text,
            "prediction": prediction,
            "reference": reference,
            "correct": is_correct
        })
        
    accuracy = correct_count / len(examples) if examples else 0
    return accuracy, results

# -----------------------------------------------------------------------------
# 7. Model Merging & Main
# -----------------------------------------------------------------------------

def merge_if_needed(model_path: str, base_model: str) -> str:
    """Merge LoRA adapter with base model if needed."""
    if not base_model: return model_path
    print(f"\n[Merge] Merging adapter '{model_path}' into base '{base_model}'...")
    temp_dir = tempfile.mkdtemp(prefix="merged_")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        model.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()
    
    final_model_path = merge_if_needed(args.model_path, args.base_model)
    model_slug = os.path.basename(args.model_path.rstrip("/"))

    tokenizer_path = args.base_model or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"Loading vLLM from {final_model_path}...")
    llm = LLM(
        model=final_model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85
    )
    
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    for dataset_name in args.datasets:
        print(f"\n{'='*40}\nProcessing: {dataset_name}\n{'='*40}")
        try:
            examples, eval_type = load_local_benchmark(dataset_name, args.data_root, args.limit)
            if not examples:
                print(f"Skipping {dataset_name} (empty).")
                continue
                
            accuracy, results = evaluate_model(llm, examples, eval_type, sampling_params, tokenizer)
            print(f"Accuracy for {dataset_name}: {accuracy:.2%}")
            
            output_path = os.path.join(args.output_dir, dataset_name, model_slug)
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, "eval_results.json")
            
            with open(output_file, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "model": args.model_path,
                    "accuracy": accuracy,
                    "details": results
                }, f, indent=2)
            print(f"Saved results to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    if args.base_model and os.path.exists(final_model_path):
        shutil.rmtree(final_model_path)

if __name__ == "__main__":
    main()