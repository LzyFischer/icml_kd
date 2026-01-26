import os
import time
import subprocess
import threading
from queue import Queue
from datetime import datetime

# ================= Configuration Area =================

# 1. Experiment Parameters
DATASETS = ["anli", "date", "math", "arc_challenge", "commonsense_qa", "strategy_qa"]

FAMILIES = [
    {
        "name": "qwen",
        # Used for 'base' ablation AND as base for LoRA merging
        "teacher_base": "unsloth/Qwen2.5-3B-Instruct", 
        "teacher_dir_prefix": "teacher" 
    },
    {
        "name": "gemma",
        "teacher_base": "unsloth/gemma-3-1b-it",
        "teacher_dir_prefix": "teacher_gemma"
    }
]

# Ablations to evaluate: 'base' plus your trained variations
# ABLATIONS = ["base", "", "no_curriculum", "no_length", "sft"] 
ABLATIONS = ["", "sft"] 

# 2. Resource Configuration
GPU_IDS = [0, 1, 2, 3]  
LOG_ROOT = "./logs_teacher_eval"
DATA_ROOT = "./data" 
OUTPUT_DIR = "./results_teacher_final"

# ===================================================

os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print_lock = threading.Lock()

def log(msg):
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_command(cmd, env_vars, log_file):
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env_vars,
            stdout=f,
            stderr=subprocess.STDOUT
        )
        process.wait()
    return process.returncode

def worker(gpu_queue, job_queue):
    while not job_queue.empty():
        try:
            job = job_queue.get_nowait()
        except:
            break

        gpu_id = gpu_queue.get() 
        
        dataset = job['dataset']
        family = job['family']
        ablation = job['ablation']
        
        # --- PATH CONSTRUCTION ---
        if ablation == "base":
            # 1. Base Model Evaluation
            # We evaluate the teacher_base directly. 
            # No --base_model needed because it's not an adapter.
            model_path = family['teacher_base']
            base_model_arg = "" 
            is_local = False
            run_id = f"{family['name']}_base_teacher"
        else:
            # 2. Checkpoint Evaluation
            # We evaluate the local LoRA checkpoint.
            # We MUST provide --base_model so eval_vllm.py can merge it.
            ablation_suffix = f"_{ablation}" if ablation else ""
            teacher_dir_name = f"{family['teacher_dir_prefix']}{ablation_suffix}"
            model_path = f"ckpts/{dataset}/{teacher_dir_name}/final_lora"
            base_model_arg = f"--base_model {family['teacher_base']}"
            is_local = True
            run_id = f"{family['name']}{ablation_suffix}_teacher"

        log_file = os.path.join(LOG_ROOT, f"{dataset}_{run_id}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        log(f"GPU {gpu_id} CHECK: {dataset} | {run_id} | Path: {model_path}")

        try:
            # Skip if local checkpoint is missing
            if is_local and not os.path.exists(model_path):
                log(f"WARNING: Checkpoint missing {model_path}. Skipping.")
                gpu_queue.put(gpu_id)
                job_queue.task_done()
                continue

            # ================= EVALUATION =================
            log(f"GPU {gpu_id} START EVAL: {dataset} | {run_id}")
            
            cmd_eval = (
                f"python eval_vllm.py "
                f"--model_path {model_path} "
                f"{base_model_arg} " 
                f"--datasets {dataset} "
                f"--data_root {DATA_ROOT} "
                f"--output_dir {OUTPUT_DIR}"
            )
            
            if run_command(cmd_eval, env, log_file) == 0:
                log(f"GPU {gpu_id} DONE: {dataset} | {run_id}")
            else:
                log(f"GPU {gpu_id} FAIL: {dataset} | {run_id} (See {log_file})")

        except Exception as e:
            log(f"EXCEPTION on GPU {gpu_id}: {e}")
        finally:
            gpu_queue.put(gpu_id)
            job_queue.task_done()

def main():
    job_queue = Queue()
    
    for ablation in ABLATIONS:
        for dataset in DATASETS:
            for family in FAMILIES:
                job = {
                    "dataset": dataset,
                    "family": family,
                    "ablation": ablation
                }
                job_queue.put(job)
    
    total_jobs = job_queue.qsize()
    log(f"Total teacher evaluation jobs scheduled: {total_jobs}")
    
    gpu_queue = Queue()
    for gid in GPU_IDS:
        gpu_queue.put(gid)
        
    threads = []
    num_threads = len(GPU_IDS)
    
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(gpu_queue, job_queue))
        t.daemon = True
        t.start()
        threads.append(t)
        
    job_queue.join()
    log("All teacher evaluations completed.")

if __name__ == "__main__":
    main()