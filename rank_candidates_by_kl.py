"""
rank_candidates_by_kl.py
========================

This script takes a JSONL file containing prompts and a list of
candidate responses per prompt (typically produced by
``generate_candidates.py``) and ranks each candidate by the forward
KL divergence between a teacher and a student model.  Candidates are
sorted in ascending order of KL (lowest divergence first) and
partitioned into ``num_buckets`` buckets where the i‑th bucket
contains the i‑th ranked response for every prompt.

The output consists of separate JSONL files (``rank_0.jsonl``,
``rank_1.jsonl``, …) written to ``output_dir``.  Optionally, a shell
script with example ``train_student.py`` commands can be generated
using ``--write_training_commands``.

Usage example:

```
python rank_candidates_by_kl.py \
    --teacher_model unsloth/Qwen2.5-3B-Instruct \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --candidate_file ./candidates/date_train.jsonl \
    --output_dir ./kl_partition_results/date \
    --write_training_commands
```

The script assumes that both teacher and student share an identical
tokenizer.  If either path refers to a LoRA adapter the adapter is
merged into the ``base_model`` using the same logic as in
``kl_ranked_training.py``.  Forward KL is computed over the full
conversation consisting of the prompt followed by the candidate
response.  See ``utils.forward_kl_div_loss`` for details on the KL
computation.
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc

from utils import forward_kl_div_loss


def merge_if_needed(model_path: str, base_model: str) -> str:
    """
    Merge a LoRA adapter into the base model if ``model_path`` points
    to an adapter directory.  Otherwise return ``model_path`` as is.
    The returned directory may be a temporary directory; callers
    should clean it up when finished.
    """
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return model_path
    import tempfile
    import shutil
    print(f"[Merge] Merging adapter '{model_path}' into base '{base_model}'...")
    temp_dir = tempfile.mkdtemp(prefix="merged_model_")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        adapter = PeftModel.from_pretrained(base, model_path)
        adapter = adapter.merge_and_unload()
        print(f"Saving merged model to {temp_dir}...")
        adapter.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        del adapter, base
        gc.collect()
        torch.cuda.empty_cache()
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir)
        raise


def compute_kl(
    teacher_model: AutoModelForCausalLM,
    student_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
) -> float:
    """
    Compute the mean forward KL divergence KL(teacher || student) over a
    single prompt/response pair.  The conversation is encoded using
    the chat template of the tokenizer.  KL is computed across all
    tokens except the final one (since there is no next token to
    predict).

    Parameters
    ----------
    teacher_model, student_model : AutoModelForCausalLM
        Teacher and student models in evaluation mode.
    tokenizer : AutoTokenizer
        Tokenizer shared by both models.
    prompt, response : str
        User instruction and candidate reply.

    Returns
    -------
    float
        Mean per‑token forward KL divergence (lower is easier).
    """
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    encoded = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        return_dict=True,
    ).to(teacher_model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        t_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        s_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
    t_logits = t_out.logits[:, :-1, :]
    s_logits = s_out.logits[:, :-1, :]
    kl_value = forward_kl_div_loss(
        student_logits=s_logits,
        teacher_logits=t_logits,
        mask=None,
        temperature=1.0,
        reduction="mean",
    ).item()
    return kl_value


def main():
    parser = argparse.ArgumentParser(description="Rank candidate responses by KL divergence and partition into buckets")
    parser.add_argument(
        "--teacher_model",
        type=str,
        required=True,
        help="Path or identifier for the teacher model (may be LoRA adapter)",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Path or identifier for the student model (may be LoRA adapter)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model for merging adapters; defaults to teacher_model",
    )
    parser.add_argument(
        "--candidate_file",
        type=str,
        required=True,
        help="JSONL file produced by generate_candidates.py containing prompts and response lists",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write ranked JSONL buckets",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="instruction",
        help="Name of the prompt column in the candidate file",
    )
    parser.add_argument(
        "--responses_field",
        type=str,
        default="responses",
        help="Name of the field containing the list of candidate responses",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Optional: number of candidates per prompt. If not provided, inferred from the data",
    )
    parser.add_argument(
        "--write_training_commands",
        action="store_true",
        help="If set, write a run_training.sh script with example student training commands",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Merge adapters if necessary
    base_model = args.base_model if args.base_model is not None else args.teacher_model
    teacher_path = merge_if_needed(args.teacher_model, base_model)
    student_path = merge_if_needed(args.student_model, base_model)

    print(f"Loading teacher model from {teacher_path}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Containers for buckets (initialized after inferring num_generations)
    buckets: List[List[dict]] = []

    # Process each line in candidate_file
    print(f"Processing candidate file {args.candidate_file}...")
    with open(args.candidate_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            record = json.loads(line)
            prompt = record[args.prompt_column]
            responses: List[str] = record[args.responses_field]
            # Infer number of generations from first record if not provided
            if args.num_generations is None:
                num_gens = len(responses)
                args.num_generations = num_gens
                buckets = [[] for _ in range(num_gens)]
                print(f"Inferred num_generations={num_gens} from first record.")
            # Ensure we have expected number of responses
            if args.num_generations is not None and len(responses) != args.num_generations:
                raise ValueError(
                    f"Record {line_idx} contains {len(responses)} responses but expected {args.num_generations}."
                )
            # Compute KL scores for each candidate
            kl_scores: List[Tuple[float, str]] = []
            for response in responses:
                try:
                    kl_val = compute_kl(
                        teacher_model=teacher_model,
                        student_model=student_model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        response=response,
                    )
                except Exception as e:
                    print(f"Warning: KL computation failed for prompt index {line_idx} due to {e}. Assigning inf.")
                    kl_val = float("inf")
                kl_scores.append((kl_val, response))
            # Sort by ascending KL
            kl_scores.sort(key=lambda x: x[0])
            # Assign each response to the corresponding bucket
            for rank, (kl_val, response) in enumerate(kl_scores):
                if rank >= args.num_generations:
                    break
                buckets[rank].append(
                    {
                        args.prompt_column: prompt,
                        "response": response,
                        "kl_value": kl_val,
                        "source": record.get("source", "teacher_generated"),
                    }
                )
            if (line_idx + 1) % 50 == 0:
                print(f"Processed {line_idx + 1} prompts...")

    # Write each bucket to JSONL
    jsonl_paths = []
    for i, bucket in enumerate(buckets):
        out_path = os.path.join(args.output_dir, f"rank_{i}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in bucket:
                json_item = {
                    args.prompt_column: item[args.prompt_column],
                    "response": item["response"],
                    "source": item["source"],
                }
                f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
        jsonl_paths.append(out_path)
        print(f"Saved {len(bucket)} samples to {out_path}")

    # Optionally write training commands
    if args.write_training_commands:
        script_path = os.path.join(args.output_dir, "run_training.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n\n")
            for i, path in enumerate(jsonl_paths):
                run_name = f"student_rank_{i}"
                cmd = (
                    f"python train_student.py "
                    f"--teacher_model {args.teacher_model} "
                    f"--student_model {args.student_model} "
                    f"--train_file {path} "
                    f"--num_epochs 1 "
                    f"--run_name {run_name} "
                )
                f.write(cmd + "\n")
            f.write("\n# Modify --num_epochs or other hyperparameters as needed.\n")
        os.chmod(script_path, 0o755)
        print(f"Wrote training command script to {script_path}")

    # Clean up temporary merged directories
    if teacher_path != args.teacher_model and "merged_model_" in teacher_path:
        import shutil
        print(f"Cleaning up temporary merged model directory {teacher_path}...")
        shutil.rmtree(teacher_path)
    if student_path != args.student_model and "merged_model_" in student_path:
        import shutil
        print(f"Cleaning up temporary merged model directory {student_path}...")
        shutil.rmtree(student_path)


if __name__ == "__main__":
    main()