"""
generate_candidates.py
======================

This script separates the response generation phase from the KL‑based
ranking logic used in ``kl_ranked_training.py``.  It produces a
candidate set of completions for each prompt in a dataset using a
teacher model.  The output is a JSONL file where each line contains
the original instruction (prompt) and a list of sampled responses.

By decoupling generation from downstream processing, you can run
vLLM or HuggingFace generation in isolation, freeing up GPU memory
for subsequent stages (e.g. KL scoring and student training).  This
is especially useful when using vLLM which may reserve a large
fraction of GPU memory.

Example usage:

```
python generate_candidates.py \
    --teacher_model unsloth/Qwen2.5-3B-Instruct \
    --input_file data/date/train.jsonl \
    --output_file ./candidates/date_train.jsonl \
    --num_generations 5 \
    --temperature 0.7 \
    --max_new_tokens 256 \
    --use_vllm
```

The generated file can then be passed to ``rank_candidates_by_kl.py`` to
compute per‑candidate KL divergences and partition the dataset into
curriculum buckets.

```
python rank_candidates_by_kl.py \
    --teacher_model unsloth/Qwen2.5-3B-Instruct \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --candidate_file ./candidates/date_train.jsonl \
    --output_dir ./kl_partition_results/date \
    --write_training_commands
```

The two‑stage workflow allows you to generate responses in a memory
efficient environment (possibly on a different machine) and then
perform KL ranking separately.

``generate_candidates.py`` supports both HuggingFace and vLLM backends
via the ``--use_vllm`` flag.  If vLLM is unavailable and the flag is
set, the script will raise an error.  Without ``--use_vllm`` the
script falls back to HuggingFace's ``model.generate``.
"""

import argparse
import json
import os
from typing import List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
import pdb

from load_data import load_data_source

# Try importing vLLM for optional high‑throughput generation.
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False


def merge_if_needed(model_path: str, base_model: str) -> str:
    """
    If ``model_path`` points to a LoRA adapter directory, merge it into
    the ``base_model`` weights and return a temporary directory
    containing the merged model and tokenizer.  Otherwise return
    ``model_path`` unchanged.  Any temporary directory created by this
    function should be removed by the caller when no longer needed.

    Parameters
    ----------
    model_path : str
        Path to a full model or a LoRA adapter directory.
    base_model : str
        Path or identifier of the base model to merge adapters into.

    Returns
    -------
    str
        Path to a merged model directory or the original ``model_path``.
    """
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return model_path
    import tempfile
    import shutil
    print(f"[Merge] Merging adapter '{model_path}' into base '{base_model}'...")
    temp_dir = tempfile.mkdtemp(prefix="merged_model_")
    try:
        # Load base model and adapter
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


def generate_responses_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_generations: int,
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    """
    Generate ``num_generations`` diverse replies using a HuggingFace model.

    This function mirrors the behaviour of ``generate_responses`` in
    ``kl_ranked_training.py``.  For each prompt, the chat template is
    applied and the input is replicated ``num_generations`` times so
    that a single ``model.generate`` call produces multiple sampled
    sequences.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The autoregressive model used to generate responses.
    tokenizer : AutoTokenizer
        Tokenizer providing a chat template.
    prompt : str
        User instruction.
    num_generations : int
        Number of candidate completions to sample.
    temperature : float
        Non‑zero softmax temperature controlling diversity.
    max_new_tokens : int
        Maximum number of tokens to generate per completion.

    Returns
    -------
    List[str]
        A list of ``num_generations`` assistant responses (plain text).
    """
    assert temperature > 0.0, "Temperature must be non‑zero for sampling"
    # Apply chat template for a single user prompt
    messages = [[{"role": "user", "content": prompt}]]
    input_texts = [
        tokenizer.apply_chat_template(m, tokenize=False) for m in messages * num_generations
    ]
    encodings = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **encodings,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    replies: List[str] = []
    for text in decoded_texts:
        # Remove the prompt if it appears in the decoded sequence
        if prompt in text:
            reply = text.split(prompt, 1)[-1].strip()
        else:
            reply = text.strip()
        replies.append(reply)
    return replies


def generate_responses_vllm(
    llm: "LLM",
    prompt: str,
    num_generations: int,
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    """
    Generate multiple responses using a vLLM engine for a single prompt.

    Parameters
    ----------
    llm : vllm.LLM
        Initialised vLLM engine for the teacher model.
    prompt : str
        User instruction.
    num_generations : int
        Number of completions to sample.
    temperature : float
        Sampling temperature (>0).
    max_new_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    List[str]
        A list of ``num_generations`` assistant responses.
    """
    assert temperature > 0.0, "Temperature must be non‑zero for vLLM sampling"
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([prompt], sampling_params)
    request_output = outputs[0]
    replies: List[str] = []
    for out in request_output.outputs:
        text = out.text
        if prompt in text:
            reply = text.split(prompt, 1)[-1].strip()
        else:
            reply = text.strip()
        replies.append(reply)
    return replies


def main():
    parser = argparse.ArgumentParser(description="Generate multiple responses per prompt and save candidates to JSONL")
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="unsloth/Qwen2.5-3B-Instruct",
        help="Identifier or path for the teacher model (may be LoRA adapter)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen2.5-3B-Instruct",
        help="Base model to merge LoRA adapters into if needed. If omitted, teacher_model is assumed to be a full model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/date/train.jsonl",
        help="Path to the training JSONL file containing prompts",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="candidates/date/candidates.jsonl",
        help="Where to write the candidate responses JSONL",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="instruction",
        help="Name of the prompt column in the input JSONL",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help=(
            "Number of responses to sample per prompt per pass. If combined with --num_passes, the total "
            "responses per prompt will be num_generations * num_passes."
        ),
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        default=1,
        help=(
            "Repeat the generation process over the entire dataset this many times. Each pass samples "
            "num_generations responses per prompt. A value >1 avoids replicating the prompt internally "
            "and may offer better memory usage."
        ),
    )
    parser.add_argument(
        "--dataset_pass",
        action="store_true",
        help=(
            "If set, each pass generates the entire dataset in a single batch."
            " For vLLM this calls llm.generate(prompts, SamplingParams(n=num_generations)). "
            "For HuggingFace this uses num_return_sequences=num_generations in a single call."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (>0)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per response",
    )
    parser.add_argument(
        "--max_data_samples",
        type=int,
        default=None,
        help="If set, limit the number of prompts processed to this many examples (useful for debugging).",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation instead of HuggingFace. Requires vLLM to be installed.",
    )
    args = parser.parse_args()

    # Create directory for output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {args.input_file}...")
    examples = load_data_source(
        args.input_file,
        split="train",
        prompt_col=args.prompt_column,
        resp_col="response",
    )
    prompts = [ex.prompt for ex in examples]
    # If max_data_samples is specified, limit the prompts
    if args.max_data_samples is not None:
        prompts = prompts[: args.max_data_samples]
        print(f"Loaded {len(prompts)} prompts (truncated by max_data_samples={args.max_data_samples}).")
    else:
        print(f"Loaded {len(prompts)} prompts.")

    # Prepare the teacher model (merge LoRA if necessary)
    base_model = args.base_model if args.base_model is not None else args.teacher_model
    teacher_path = merge_if_needed(args.teacher_model, base_model)

    # Optionally set up vLLM engine
    llm = None
    tokenizer = None
    model = None
    if args.use_vllm:
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "--use_vllm was specified but vLLM is not installed. Install vLLM or omit this flag."
            )
        print(f"Initializing vLLM engine for model {teacher_path}...")
        llm = LLM(
            model=teacher_path,
            # tensor_parallel_size=1,
            # trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
    else:
        print(f"Loading HuggingFace model from {teacher_path} for generation...")
        tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Prepare a list to accumulate responses per prompt.  Initialize with empty lists.
    total_prompts = len(prompts)
    responses_dict: List[List[str]] = [[] for _ in range(total_prompts)]

    # Outer loop: repeat generation num_passes times. Each pass may operate
    # either on the entire dataset at once (dataset_pass=True) or per
    # prompt. Dataset-level generation can be faster on vLLM because it
    # batches all prompts in a single call similar to generate_dataset_vllm.
    for pass_idx in range(args.num_passes):
        print(f"Starting generation pass {pass_idx + 1}/{args.num_passes}...")
        if args.dataset_pass:
            # -- dataset-level generation --
            if args.use_vllm:
                # vLLM: generate n completions for each prompt in one call
                sampling_params = SamplingParams(
                    n=args.num_generations,
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                )
                outputs = llm.generate(prompts, sampling_params)
                # outputs is a list of RequestOutput objects corresponding to prompts
                for idx, request_output in enumerate(outputs):
                    prompt_text = prompts[idx]
                    for out in request_output.outputs:
                        text = out.text
                        if prompt_text in text:
                            reply = text.split(prompt_text, 1)[-1].strip()
                        else:
                            reply = text.strip()
                        responses_dict[idx].append(reply)
                print(
                    f"[Pass {pass_idx + 1}/{args.num_passes}] Generated {args.num_generations} responses for all {total_prompts} prompts."
                )
            else:
                # HuggingFace: use num_return_sequences to generate multiple completions per prompt in a single call
                # Build the chat template for each prompt in the batch
                input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False
                    )
                    for p in prompts
                ]
                encodings = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **encodings,
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # Group outputs per prompt: each prompt has num_generations outputs
                for idx in tqdm(range(total_prompts), desc="Processing generated outputs"):
                    prompt_text = prompts[idx]
                    start = idx * args.num_generations
                    end = start + args.num_generations
                    group = decoded_texts[start:end]
                    for text in group:
                        if prompt_text in text:
                            reply = text.split(prompt_text, 1)[-1].strip()
                        else:
                            reply = text.strip()
                        responses_dict[idx].append(reply)
                print(
                    f"[Pass {pass_idx + 1}/{args.num_passes}] Generated {args.num_generations} responses for all {total_prompts} prompts."
                )
        else:
            # -- per prompt generation --
            for idx, prompt in tqdm(enumerate(prompts), desc="Generating responses per prompt"):
                if args.use_vllm:
                    replies = generate_responses_vllm(
                        llm=llm,
                        prompt=prompt,
                        num_generations=args.num_generations,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    replies = generate_responses_hf(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        num_generations=args.num_generations,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                responses_dict[idx].extend(replies)
                # Periodic progress logging for per-prompt mode
                if (idx + 1) % 10 == 0:
                    print(
                        f"[Pass {pass_idx + 1}/{args.num_passes}] Generated for {idx + 1}/{total_prompts} prompts..."
                    )

    # Write accumulated responses to file
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for idx, prompt in enumerate(prompts):
            record = {
                args.prompt_column: prompt,
                "responses": responses_dict[idx],
                "source": "teacher_generated",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Saved candidate responses for {total_prompts} prompts to {args.output_file} "
        f"(total responses per prompt = {len(responses_dict[0]) if responses_dict else 0})."
    )

    # Clean up merged model directory if created
    if teacher_path != args.teacher_model and "merged_model_" in teacher_path:
        import shutil
        print(f"Cleaning up temporary merged model directory {teacher_path}...")
        shutil.rmtree(teacher_path)


if __name__ == "__main__":
    main()