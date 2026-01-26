import os
import time
import subprocess
import threading
from queue import Queue
from datetime import datetime
import pdb

# ================= 配置区域 =================

# 1. 实验参数
DATASETS = ["anli", "date", "math", "arc_challenge", "commonsense_qa", "strategy_qa"]
FAMILIES = [
    {
        "name": "qwen",
        "student": "unsloth/Qwen2.5-0.5B-Instruct",
        "teacher_base": "unsloth/Qwen2.5-3B-Instruct",
        "teacher_dir_prefix": "teacher" 
    },
    {
        "name": "gemma",
        "student": "unsloth/gemma-3-270m-it",
        "teacher_base": "unsloth/gemma-3-1b-it",
        "teacher_dir_prefix": "teacher_gemma"
    }
]

# [修改点 1] 在这里加入了 "base"，脚本会自动识别并处理
ABLATIONS = ["",  "sft", "base", "no_curriculum", "no_length"] 
# ABLATIONS = ["base"] 
# ABLATIONS = ["base"] # 如果只想跑 base，可以解开这行注释

METHODS = ["sft", "kl", "seqkd", "onpolicy"]
# METHODS = ["seqkd"]

# 2. 资源配置
GPU_IDS = [0,1,2,3]  # 可用 GPU 列表
LOG_ROOT = "./logs_batch_experiment"
DATA_ROOT = "./data" # 数据根目录

# ===========================================

os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs("temp_data_seqkd", exist_ok=True)

print_lock = threading.Lock()

def log(msg):
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_epoch(dataset):
    if dataset in ["anli", "date"]:
        return 2
    return 1

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
        method = job['method']
        
        ablation_suffix = f"_{ablation}" if ablation else ""
        run_id = f"{family['name']}{ablation_suffix}_{method}"
        
        # [修改点 2] Base Model 的路径逻辑处理
        if ablation == "base":
            # 如果是 base 实验，直接使用 config 里的 teacher_base (HF 路径)
            teacher_ckpt = family['teacher_base']
            is_local_ckpt = False
        else:
            # 否则使用本地 ckpts 路径
            teacher_dir_name = f"{family['teacher_dir_prefix']}{ablation_suffix}"
            teacher_ckpt = f"ckpts/{dataset}/{teacher_dir_name}/final_lora"
            is_local_ckpt = True
        
        log_file = os.path.join(LOG_ROOT, f"{dataset}_{run_id}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 显式定义数据路径
        train_file = os.path.join(DATA_ROOT, dataset, "train.jsonl")
        val_file = os.path.join(DATA_ROOT, dataset, "test.jsonl") 
        
        if not os.path.exists(val_file):
            if os.path.exists(os.path.join(DATA_ROOT, dataset, "test.json")):
                 val_file = os.path.join(DATA_ROOT, dataset, "test.json")

        log(f"GPU {gpu_id} START: {dataset} | {run_id}")

        try:
            # [修改点 3] 只在是本地 checkpoint 时检查路径是否存在
            if is_local_ckpt and not os.path.exists(teacher_ckpt):
                log(f"WARNING: Teacher missing {teacher_ckpt}. Skipping.")
                gpu_queue.put(gpu_id)
                job_queue.task_done()
                continue

            epoch = get_epoch(dataset)
            final_model_path = ""

            # ================= 执行实验逻辑 =================
            
            if method == "sft":
                run_name = f"{run_id}"
                cmd = (
                    f"python train_student.py "
                    f"--teacher_model {teacher_ckpt} "
                    f"--student_model {family['student']} "
                    f"--train_file {train_file} "  
                    f"--val_file {val_file} "      
                    f"--run_name {run_name} "
                    f"--loss_type sft "
                    f"--student_mode supervised "
                    f"--num_epochs {epoch} "
                    f"--use_wandb False"
                )
                if run_command(cmd, env, log_file) == 0:
                    final_model_path = f"ckpts/{dataset}/student_model_{run_name}"

            elif method == "kl":
                run_name = f"{run_id}"
                cmd = (
                    f"python train_student.py "
                    f"--teacher_model {teacher_ckpt} "
                    f"--student_model {family['student']} "
                    f"--train_file {train_file} "  
                    f"--val_file {val_file} "      
                    f"--run_name {run_name} "
                    f"--loss_type forward "
                    f"--student_mode supervised "
                    f"--num_epochs {epoch} "
                    f"--use_wandb False"
                )
                if run_command(cmd, env, log_file) == 0:
                    final_model_path = f"ckpts/{dataset}/student_model_{run_name}"

            elif method == "seqkd":
                # 1. Generate
                gen_file = f"temp_data_seqkd/{dataset}_{run_id}.jsonl"
                cmd_gen = (
                    f"python generate_dataset_vllm.py "
                    f"--base_model {family['teacher_base']} " # Base Teacher
                    f"--model_path {teacher_ckpt} "           # Teacher CKPT (Base时即为Base Teacher)
                    f"--input_file {train_file} " 
                    f"--output_file {gen_file} "
                    f"--temperature 1.0"
                )
                run_command(cmd_gen, env, log_file)
                
                # 2. Train
                run_name = f"{run_id}"
                cmd_train = (
                    f"python train_student.py "
                    f"--teacher_model {teacher_ckpt} "
                    f"--student_model {family['student']} "
                    f"--train_file {gen_file} "    
                    f"--val_file {val_file} "      
                    f"--run_name {run_name} "
                    f"--loss_type sft "
                    f"--student_mode supervised " # 注意：这里 SeqKD 通常是 supervised 或 teacher_generated
                    f"--num_epochs {epoch} "
                    f"--use_wandb False"
                )
                # 修正：SeqKD 在 train_student 中通常用 teacher_generated 模式读取数据
                # 但如果你的 generated data 格式和 original 一样（prompt/response），supervised 也可以
                # 为了保险，还是用 explicit teacher_generated 逻辑如果你的脚本支持
                # 这里保持你原代码逻辑，如有需要请改为 teacher_generated
                
                with open(log_file, "a") as f:
                    subprocess.run(cmd_train, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
                
                final_model_path = f"ckpts/{dataset}/student_model_{run_name}"

            elif method == "onpolicy":
                cmd = (
                    f"bash run_iterative_pipeline.sh "
                    f"-i 2 "
                    f"-b {family['student']} "
                    f"-t {teacher_ckpt} "
                    f"-f {train_file} " 
                    f"-v {val_file} "   
                    f"-n {run_id} "
                )
                with open(log_file, "a") as f:
                    p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    last_line = ""
                    for line in p.stdout:
                        f.write(line)
                        line = line.strip()
                        if line: last_line = line
                    p.wait()
                
                if p.returncode == 0 and os.path.exists(last_line):
                    final_model_path = last_line
                else:
                    log(f"ERROR: On-policy failed for {run_id}")

            # ================= Evaluation =================
            if final_model_path and os.path.exists(final_model_path):
                log(f"GPU {gpu_id} EVAL: {dataset} | {run_id}")
                cmd_eval = (
                    f"python eval_vllm.py "
                    f"--model_path {final_model_path} "
                    f"--base_model {family['student']} "
                    f"--datasets {dataset} "
                    f"--data_root {DATA_ROOT} "
                    f"--output_dir results_final"
                )
                with open(log_file, "a") as f:
                    subprocess.run(cmd_eval, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
                log(f"GPU {gpu_id} DONE: {dataset} | {run_id}")
            else:
                log(f"GPU {gpu_id} FAIL: {dataset} | {run_id} (No Model Produced)")

        except Exception as e:
            log(f"EXCEPTION on GPU {gpu_id}: {e}")
        finally:
            gpu_queue.put(gpu_id)
            job_queue.task_done()

def main():
    job_queue = Queue()
    
    # 将 Base 任务放在前面或者混合在里面
    for ablation in ABLATIONS:
        for dataset in DATASETS:
            for family in FAMILIES:
                for method in METHODS:
                    job = {
                        "dataset": dataset,
                        "family": family,
                        "ablation": ablation,
                        "method": method
                    }
                    job_queue.put(job)
    
    total_jobs = job_queue.qsize()
    log(f"Total jobs scheduled: {total_jobs}")
    
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
    log("All jobs completed.")

if __name__ == "__main__":
    main()