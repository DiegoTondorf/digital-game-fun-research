#!/usr/bin/env python3
# 00_run_pipeline.py — orchestrates steps and logs output (TEE)


'''
Step 00: Pipeline Orchestrator - Orchestrates all pipeline steps, manages config, and logs output.

• Runs each pipeline step as a subprocess in order
• Manages environment configuration and logging
• Captures and logs outputs for reproducibility

Inputs: Project scripts, environment variables, configuration files.
Outputs: Step logs, orchestrated outputs, reproducibility logs.
'''

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from pipeline.Z_utils.common import log, ensure_dir, find_project_root

# --- Config with env override ---
def get_env(key: str, default):
    val = os.getenv(key)
    if val is None:
        return default
    # Try to cast to type of default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(val)
        except Exception:
            return default
    return val





A_ROOT = find_project_root()
A_GAT = A_ROOT / "pipeline/A_gathering"
B_SAM = A_ROOT / "pipeline/B_sampling"
C_QUA = A_ROOT / "pipeline/C_quality_control"
D_DES = A_ROOT / "pipeline/D_descriptive_stats"
E_ANA = A_ROOT / "pipeline/E_analysis"
F_DAT = A_ROOT / "pipeline/F_data_preparation"

CONFIG: Dict[str, object] = {
    "CLEAN_DATA_BEFORE_RUN": get_env("PIPELINE_CLEAN_DATA", False),
    "PYTHON": get_env("PIPELINE_PYTHON", sys.executable or "python"),
    "AUTO_INSTALL": get_env("PIPELINE_AUTO_INSTALL", True),
    "REQUIRED_PKGS": {
        "requests": "requests",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    },
    "RUN_STEP_01": get_env("PIPELINE_RUN_STEP_01", True), # Fetch from SteamSpy
    "RUN_STEP_02": get_env("PIPELINE_RUN_STEP_02", True), # Prepare metrics and rank
    "RUN_STEP_03": get_env("PIPELINE_RUN_STEP_03", True), # Filter and clean
    "RUN_STEP_04": get_env("PIPELINE_RUN_STEP_04", True), # Select pairs
    "RUN_STEP_05": get_env("PIPELINE_RUN_STEP_05", True), # Fetch reviews
    "RUN_STEP_06": get_env("PIPELINE_RUN_STEP_06", True), # Sample LLM reviews
    "RUN_STEP_07": get_env("PIPELINE_RUN_STEP_07", True), # Quality control
    "RUN_STEP_08": get_env("PIPELINE_RUN_STEP_08", True), # Descriptive stats
    "RUN_STEP_09": get_env("PIPELINE_RUN_STEP_09", True), # Dictionary analysis
    "RUN_STEP_10E": get_env("PIPELINE_RUN_STEP_10E", False), # LLM scoring (export prompts)
    "RUN_STEP_10B": get_env("PIPELINE_RUN_STEP_10B", False), # LLM scoring (batch run)
    "RUN_STEP_10I": get_env("PIPELINE_RUN_STEP_10I", False), # LLM scoring (ingest responses)
    "RUN_STEP_11": get_env("PIPELINE_RUN_STEP_11", True), # Embeddings
    "RUN_STEP_12": get_env("PIPELINE_RUN_STEP_12", True), # Cluster + dimensionality reduction
    "RUN_STEP_13": get_env("PIPELINE_RUN_STEP_13", True), # LLM cluster scoring
    "RUN_STEP_14": get_env("PIPELINE_RUN_STEP_14", False), # 
    "SCRIPTS": {
        "01": str(A_GAT / "01_steamspy_fetch.py"),
        "02": str(A_GAT / "02_prepare_metrics.py"),
        "03": str(A_GAT / "03_filter_clean.py"),
        "04": str(A_GAT / "04_select_pairs.py"),
        "05": str(A_GAT / "05_fetch_reviews.py"),
        "06": str(B_SAM / "06_sample_llm_reviews.py"),
        "07": str(C_QUA / "07_quality_control.py"),
        "08": str(D_DES / "08_descriptive_stats.py"),
        "09": str(E_ANA / "09_dictionary_analysis.py"),
        "10e": str(E_ANA / "10_llm_scoring.py"),
        "10b": str(E_ANA / "10b_llm_run_ollama.py"), #10b_llm_run_ollama.py | #10a_llm_dryrun.py | 10b_llm_run_google.py
        "10i": str(E_ANA / "10_llm_scoring.py"),
        "11": str(E_ANA / "11_embeddings.py"),
        "12": str(E_ANA / "12_cluster_reduce.py"),
        "13": str(E_ANA / "13_llm_cluster_scoring.py"),
        "14": str(F_DAT / "14_prepare_latex_data.py"),
    },
    "STEP_ARGS": {
        "10e": ["--export-only"],
        "10i": ["--ingest-only"]
    },
    "LOG_DIR": get_env("PIPELINE_LOG_DIR", "logs"),
}



# --- Helper functions ---
def pip_available(py_exe: str) -> bool:
    try:
        proc = subprocess.run([py_exe, "-m", "pip", "--version"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def have_module(name: str) -> bool:
    try:
        import importlib.util as _iu
        return _iu.find_spec(name) is not None
    except Exception:
        return False

def install_package(py_exe: str, pip_name: str) -> bool:
    log(f"[INFO] Installing missing dependency: {pip_name}")
    proc = subprocess.run([py_exe, "-m", "pip", "install", "--upgrade", pip_name],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        log(f"[OK] Installed: {pip_name}")
        return True
    log(f"[ERROR] pip install failed for {pip_name}\n{proc.stdout}\n{proc.stderr}")
    return False

def preflight_dependencies(py_exe: str, required: Dict[str, str], auto_install: bool) -> bool:
    ensure_dir(Path(CONFIG["LOG_DIR"]))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pflog = Path(CONFIG["LOG_DIR"]) / f"preflight_{ts}.log"
    with pflog.open("w", encoding="utf-8") as f:
        f.write("# PRE-FLIGHT DEPENDENCY CHECK\n")
        f.write(f"# PYTHON: {py_exe}\n")
        pip_ok = pip_available(py_exe)
        f.write(f"# pip available: {pip_ok}\n")
        all_ok = True
        for mod, pipn in required.items():
            present = have_module(mod)
            f.write(f"[CHECK] {mod}: {'OK' if present else 'MISSING'}\n")
            if present: continue
            all_ok = False
            if auto_install and pip_ok:
                ok = install_package(py_exe, pipn)
                f.write(f"[INSTALL] {pipn}: {'OK' if ok else 'FAIL'}\n")
                if ok:
                    present = have_module(mod)
                    f.write(f"[RECHECK] {mod}: {'OK' if present else 'MISSING'}\n")
                if not present: all_ok = False
            else:
                all_ok = False
        f.write(f"\n=== RESULT: {'OK' if all_ok else 'MISSING DEPENDENCIES'} ===\n")
    log(f"[INFO] Pre-flight log: {pflog}")
    return all_ok

def run_step(step_key: str, extra_args: List[str] = None) -> int:
    scripts: Dict[str, str] = CONFIG["SCRIPTS"]
    script = scripts.get(step_key)
    if not script or not Path(script).exists():
        log(f"[ERR] Script missing for step {step_key}: {script}"); return 1
    ensure_dir(Path(CONFIG["LOG_DIR"]))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(CONFIG["LOG_DIR"]) / f"step_{step_key}_{ts}.log"
    extra = extra_args or []
    step_args = CONFIG.get("STEP_ARGS", {})
    if isinstance(step_args, dict):
        maybe = step_args.get(step_key, [])
        if isinstance(maybe, (list, tuple)): extra = list(maybe)
    cmd = [CONFIG["PYTHON"], script] + extra
    log(f"\n=== Running step {step_key}: {script}")
    log(f" Log: {log_path.resolve()}")
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# CMD: {' '.join(map(str, cmd))}\n")
        logf.write(f"# CWD: {Path.cwd()}\n")
        logf.write("# --- STDOUT/STDERR (tee) ---\n"); logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip())
            logf.write(line)
        proc.wait()
        rc = proc.returncode
    log(f"=== Step {step_key} exited with code {rc} ===")
    return rc

def main() -> None:
    # Optional cleanup of data/ folder before run
    if CONFIG.get("CLEAN_DATA_BEFORE_RUN", False):
        data_dir = Path("data")
        if data_dir.exists() and data_dir.is_dir():
            import shutil
            log("[INFO] Cleaning up data/ folder before pipeline run...")
            shutil.rmtree(data_dir)
            log("[OK] data/ folder deleted.")

    py_exe = str(CONFIG["PYTHON"])
    ok = preflight_dependencies(py_exe, CONFIG["REQUIRED_PKGS"], CONFIG["AUTO_INSTALL"])
    if not ok: sys.exit(2)

    sequence: List[str] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10e", "10b", "10i", "11", "12", "13", "14"]
    for key in sequence:
        if not CONFIG.get(f"RUN_STEP_{key.upper()}", False):
            continue
        extra = []
        if key == "10b":
            extra = ["--profile", "fast", "--limit", "15", "--max-chars", "800"]
        rc = run_step(key, extra)
        if rc != 0:
            log(f"[STOP] Pipeline stopped due to error in step {key}.")
            sys.exit(rc)
    log("\n[OK] Pipeline finished.")
    

if __name__ == "__main__":
    main()