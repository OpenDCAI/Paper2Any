#!/usr/bin/env python3
import os
import sys
import yaml
import time
import subprocess
import signal
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "conf" / "model_servers.yaml"
LOG_DIR = ROOT_DIR / "logs"

def load_config():
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_pids_on_port(port):
    """获取占用指定端口的进程PID"""
    try:
        # lsof -t -i:port 返回 PID
        output = subprocess.check_output(f"lsof -t -i:{port}", shell=True).decode().strip()
        if output:
            return [int(pid) for pid in output.split('\n') if pid]
    except subprocess.CalledProcessError:
        pass
    return []

def kill_process_on_port(port):
    """杀死占用指定端口的进程"""
    pids = get_pids_on_port(port)
    for pid in pids:
        print(f"Killing process on port {port} (PID: {pid})")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

def ensure_log_dir():
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True)

def start_mineru(config):
    print("\n[MinerU] Starting services...")
    mineru_cfg = config.get("mineru", {})
    if not mineru_cfg:
        print("[MinerU] No config found, skipping.")
        return

    model_path = mineru_cfg.get("model_path")
    gpu_util = mineru_cfg.get("gpu_utilization", 0.2)
    
    # 1. Start Backends
    backends = []
    instance_count = 0
    
    for instance_group in mineru_cfg.get("instances", []):
        gpu_id = instance_group["gpu_id"]
        ports = instance_group["ports"]
        
        for port in ports:
            instance_count += 1
            kill_process_on_port(port)
            
            log_file = LOG_DIR / f"mineru_backend_{instance_count}.log"
            cmd = [
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                "nohup", "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--served-model-name", "mineru",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--logits-processors", "mineru_vl_utils:MinerULogitsProcessor",
                "--gpu-memory-utilization", str(gpu_util),
                "--trust-remote-code",
                "--enforce-eager"
            ]
            
            full_cmd = " ".join(cmd)
            print(f"Starting MinerU Backend #{instance_count} on GPU {gpu_id} Port {port}...")
            
            with open(log_file, "w") as f:
                subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT_DIR)
            
            backends.append(f"http://127.0.0.1:{port}")
            time.sleep(5)  # Stagger start

    # 2. Start Load Balancer
    lb_cfg = mineru_cfg.get("load_balancer", {})
    if lb_cfg and backends:
        lb_port = lb_cfg["port"]
        lb_host = lb_cfg.get("host", "127.0.0.1")
        lb_name = lb_cfg.get("name", "MinerU LB")
        
        kill_process_on_port(lb_port)
        
        backends_str = " ".join(backends)
        log_file = LOG_DIR / "mineru_lb.log"
        
        cmd = [
            "nohup", "python3", "dataflow_agent/toolkits/model_servers/generic_lb.py",
            "--port", str(lb_port),
            "--host", lb_host,
            "--name", f'"{lb_name}"',
            "--backends", backends_str
        ]
        
        full_cmd = " ".join(cmd)
        print(f"Starting MinerU LB on {lb_host}:{lb_port}...")
        
        with open(log_file, "w") as f:
            subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT_DIR)

def start_sam(config):
    print("\n[SAM] Starting services...")
    sam_cfg = config.get("sam", {})
    if not sam_cfg:
        print("[SAM] No config found, skipping.")
        return

    # 1. Start Backends
    backends = []
    instance_count = 0
    
    for instance_group in sam_cfg.get("instances", []):
        gpu_id = instance_group["gpu_id"]
        ports = instance_group["ports"] # Changed to list of ports to match YAML structure
        
        for port in ports:
            instance_count += 1
            kill_process_on_port(port)
            
            log_file = LOG_DIR / f"sam_backend_{instance_count}.log"
            
            # Using env to set CUDA_VISIBLE_DEVICES
            cmd = [
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                "nohup", "uvicorn", "dataflow_agent.toolkits.model_servers.sam_server:app",
                "--port", str(port),
                "--host", "0.0.0.0"
            ]
            
            full_cmd = " ".join(cmd)
            print(f"Starting SAM Backend #{instance_count} on GPU {gpu_id} Port {port}...")
            
            with open(log_file, "w") as f:
                subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT_DIR)
            
            backends.append(f"http://127.0.0.1:{port}")
            time.sleep(1)

    # 2. Start Load Balancer
    lb_cfg = sam_cfg.get("load_balancer", {})
    if lb_cfg and backends:
        lb_port = lb_cfg["port"]
        lb_host = lb_cfg.get("host", "127.0.0.1")
        lb_name = lb_cfg.get("name", "SAM LB")
        
        kill_process_on_port(lb_port)
        
        backends_str = " ".join(backends)
        log_file = LOG_DIR / "sam_lb.log"
        
        cmd = [
            "nohup", "python3", "dataflow_agent/toolkits/model_servers/generic_lb.py",
            "--port", str(lb_port),
            "--host", lb_host,
            "--name", f'"{lb_name}"',
            "--backends", backends_str
        ]
        
        full_cmd = " ".join(cmd)
        print(f"Starting SAM LB on {lb_host}:{lb_port}...")
        
        with open(log_file, "w") as f:
            subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT_DIR)

def start_ocr(config):
    print("\n[OCR] Starting services...")
    ocr_cfg = config.get("ocr", {})
    if not ocr_cfg:
        print("[OCR] No config found, skipping.")
        return

    port = ocr_cfg.get("port", 8003)
    host = ocr_cfg.get("host", "0.0.0.0")
    workers = ocr_cfg.get("workers", 4)
    # device logic can be added if OCR server supports it via env var, currently passing empty CUDA_VISIBLE_DEVICES for cpu
    
    kill_process_on_port(port)
    
    log_file = LOG_DIR / "ocr_server.log"
    
    cmd = [
        "CUDA_VISIBLE_DEVICES=''",
        "nohup", "uvicorn", "dataflow_agent.toolkits.model_servers.ocr_server:app",
        "--port", str(port),
        "--host", host,
        "--workers", str(workers)
    ]
    
    full_cmd = " ".join(cmd)
    print(f"Starting OCR Server on {host}:{port} with {workers} workers...")
    
    with open(log_file, "w") as f:
        subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT_DIR)

def main():
    print(f"Working Directory: {ROOT_DIR}")
    ensure_log_dir()
    config = load_config()
    
    start_mineru(config)
    start_sam(config)
    start_ocr(config)
    
    print("\nAll services started! Check logs in logs/ directory.")

if __name__ == "__main__":
    main()
