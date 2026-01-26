import os
import torch
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Callable, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import pdb

@dataclass
class Example:
    prompt: str
    response: str

def format_anli(row) -> Tuple[str, str]:
    """Formats ANLI premise and hypothesis."""
    premise = row.get("premise", "")
    hypothesis = row.get("hypothesis", "")
    
    prompt = (
        f'Given that "{premise}"\n'
        f'Question: {hypothesis} True, False, or Neither?\n\n'
        f'Please reason step by step, and conclude with your final answer.'
    )
    return prompt, row.get("response", "")

def format_arc_challenge(row) -> Tuple[str, str]:
    """Formats ARC-Challenge with proper multiple choice prompt."""
    question = row.get("question", "")
    choices = row.get("choices", {})
    
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = choices["label"]
        texts = choices["text"]
        num_choices = len(labels)
        
        # Determine choice letters string (e.g., "A, B, C, and D")
        if num_choices == 4:
            choice_str = "A, B, C, and D"
        elif num_choices == 5:
            choice_str = "A, B, C, D, and E"
        elif num_choices == 6:
            choice_str = "A, B, C, D, E, and F"
        else:
            choice_str = ", ".join(labels[:-1]) + f", and {labels[-1]}"
        
        # Format choices
        formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        
        prompt = (
            f"Given the following question and {num_choices} candidate answers ({choice_str}), choose the best answer.\n"
            f"Question: {question}\n"
            f"{formatted_choices}\n"
            f"Please reason step by step, and conclude with your choice. Your response should end with "
            f'"The best answer is []" where the [] is one of {choice_str}.'
        )
    else:
        prompt = f"Question: {question}\n{str(choices)}"
    
    return prompt, row.get("response", "")

def format_commonsense_qa(row) -> Tuple[str, str]:
    """Formats CommonsenseQA (5 choices)."""
    question = row.get("question", "")
    choices = row.get("choices", {})
    
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = choices["label"]
        texts = choices["text"]
        formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        
        prompt = (
            f"Given the following question and five candidate answers (A, B, C, D, and E), choose the best answer.\n"
            f"Question: {question}\n"
            f"{formatted_choices}\n"
            f"Please reason step by step, and conclude with your choice. Your response should end with "
            f'"The best answer is []" where the [] is one of A, B, C, D, or E.'
        )
    else:
        prompt = f"Question: {question}"
    
    return prompt, row.get("response", "")

def format_date(row) -> Tuple[str, str]:
    """Formats date reasoning questions."""
    question = row.get("question", "")
    choices = row.get("choices", {})
    
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = choices["label"]
        texts = choices["text"]
        num_choices = len(labels)
        
        # Determine choice letters string
        if num_choices == 6:
            choice_str = "A, B, C, D, E, and F"
        else:
            choice_str = ", ".join(labels[:-1]) + f", and {labels[-1]}"
        
        formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        
        prompt = (
            f"Given the following question and {num_choices} candidate answers ({choice_str}), choose the best answer.\n"
            f"Question: {question}\n"
            f"{formatted_choices}\n"
            f"Please reason step by step, and conclude with your choice. Your response should end with "
            f'"The best answer is []" where the [] is one of {choice_str}.'
        )
    else:
        prompt = f"Question: {question}"
    
    return prompt, row.get("response", "")

def format_gsm8k(row) -> Tuple[str, str]:
    """Formats GSM8K math word problems."""
    question = row.get("question", "")
    prompt = f"Question: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return prompt, row.get("response", "")

def format_math(row) -> Tuple[str, str]:
    """Formats MATH dataset problems."""
    question = row.get("question", "")
    prompt = f"Question: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return prompt, row.get("response", "")

def format_strategy_qa(row) -> Tuple[str, str]:
    """Formats StrategyQA yes/no questions."""
    question = row.get("question", "")
    prompt = f"Question: True or False: {question}\n\nPlease reason step by step, and conclude with either \"True\" or \"False\"."
    return prompt, row.get("response", "")

def format_table_mwp(row) -> Tuple[str, str]:
    """Formats table-based math word problems."""
    question = row.get("question", "")
    table_title = row.get("table_title", "")
    table = row.get("table", "")
    
    prompt = (
        f'Read the following table and regarding "{table_title}" and then answer a question:\n\n'
        f'{table}\n\n'
        f'Question: {question}\n\n'
        f'Please reason step by step, and put your final answer within \\boxed{{}}.'
    )
    return prompt, row.get("response", "")

def load_data_source(path_or_name: str, split: str = "train", prompt_col: str = "instruction", resp_col: str = "response", limit: Optional[int] = None) -> List[Example]:
    print(f"Loading data from: {path_or_name}...")
    
    ds = None
    # if os.path.isfile(path_or_name):
        # ATTEMPT 1: Try Standard HuggingFace Loading
        # try:
        #     if path_or_name.endswith(".jsonl"):
        #         ds = load_dataset("json", data_files={split: path_or_name}, split=split)
        #     else:
        #         ds = load_dataset("json", data_files={split: path_or_name}, field="instances", split=split)
        # except Exception as e:
            # ATTEMPT 2: Fallback to Python JSON loading (Schema-agnostic)
    # print(f"Warning: load_dataset failed ({e}). Falling back to standard Python json/jsonl load.")
    data = []
    with open(path_or_name, "r", encoding="utf-8") as f:
        if path_or_name.endswith(".jsonl"):
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
    # else:
    #     ds = load_dataset(path_or_name, split=split)
    if limit:
        try:
            ds = ds.select(range(min(len(ds), limit)))
        except Exception:
            ds = ds[:limit]

    lower_name = path_or_name.lower()
    
    # Detect dataset type and use appropriate formatter
    if "anli" in lower_name:
        formatter = format_anli
        print("-> Detected ANLI format.")
    elif "arc" in lower_name and "challenge" in lower_name:
        formatter = format_arc_challenge
        print("-> Detected ARC Challenge format.")
    elif "commonsense" in lower_name or "csqa" in lower_name:
        formatter = format_commonsense_qa
        print("-> Detected CommonsenseQA format.")
    elif "date" in lower_name:
        formatter = format_date
        print("-> Detected Date format.")
    elif "gsm8k" in lower_name:
        formatter = format_gsm8k
        print("-> Detected GSM8K format.")
    elif "math" in lower_name and "gsm" not in lower_name:
        formatter = format_math
        print("-> Detected MATH format.")
    elif "strategy" in lower_name or "strategyqa" in lower_name:
        formatter = format_strategy_qa
        print("-> Detected StrategyQA format.")
    elif "table" in lower_name and "mwp" in lower_name:
        formatter = format_table_mwp
        print("-> Detected Table MWP format.")
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
        except Exception as e:
            print(f"Warning: Failed to format row: {e}")
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