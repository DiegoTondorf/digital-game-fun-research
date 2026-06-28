#!/usr/bin/env python3
import os
import shutil


'''
Step 10b: LLM Run (Ollama) - Runs LLM prompts using the Ollama backend and manages results.

• Handles model selection and pulling
• Dispatches prompts in parallel
• Extracts and validates JSON responses
• Supports resumable and appendable runs.

Inputs: Prompts JSONL file, configuration files.
Outputs: Responses JSONL file, debug logs.
'''

import json
import os
import sys
import time
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Track all running Ollama subprocesses
OLLAMA_PROCS = []
from pathlib import Path
from pipeline.Z_utils.common import ensure_dir, log as masked_log, find_project_root
from typing import Dict, Any, Optional, Tuple, List


ROOT = find_project_root()
DATA = ROOT / "data"


# --- Config with env override ---
def get_env(key: str, default):
    val = os.getenv(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(val)
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except Exception:
            return default
    return val

CONFIG: Dict[str, Any] = {
    "PROMPTS_JSONL": get_env("STEP10B_PROMPTS_JSONL", DATA / "analysis/10/requests/prompts.jsonl"),
    "RESPONSES_JSONL": get_env("STEP10B_RESPONSES_JSONL", DATA / "analysis/10/responses/responses.jsonl"),

    "INPUTS_DIR": get_env("STEP10B_INPUTS_DIR", "analysis_inputs/"),
    "PROMPTS_DIR": get_env("STEP10B_PROMPTS_DIR", "analysis_inputs/prompts/"),
    "CONFIG_DIR": get_env("STEP10B_CONFIG_DIR", "analysis_inputs/config/"),

    "PROFILES": {
        "fast":   {"candidate_models": ["phi3", "phi3:latest"], "workers": 2, "prompt_variant": "short",    "max_chars": 800},
        "medium": {"candidate_models": ["mistral:7b-instruct", "phi3", "phi3:latest"], "workers": 2, "prompt_variant": "standard", "max_chars": 1600},
        "complex":{"candidate_models": ["mistral:7b-instruct", "phi3", "phi3:latest"], "workers": 1, "prompt_variant": "standard", "max_chars": 2400},
    },
    "DEFAULT_PROFILE": "fast",

    "MODEL": get_env("STEP10B_MODEL", os.getenv("OLLAMA_MODEL", "")),
    "WORKERS": get_env("STEP10B_WORKERS", int(os.getenv("OLLAMA_WORKERS", "0"))),

    "REQUEST_TIMEOUT_SEC": get_env("STEP10B_TIMEOUT", int(os.getenv("OLLAMA_TIMEOUT", "900"))),
    "DISPATCH_SLEEP": get_env("STEP10B_SLEEP", float(os.getenv("OLLAMA_SLEEP", "0.0"))),
    "LIMIT_ITEMS": get_env("STEP10B_LIMIT", int(os.getenv("OLLAMA_LIMIT", "0")) or None),

    "HEARTBEAT_SEC": get_env("STEP10B_HEARTBEAT_SEC", 30),
    "APPEND_MODE": get_env("STEP10B_APPEND_MODE", True),
    "RESUME_FROM_EXISTING": get_env("STEP10B_RESUME_FROM_EXISTING", True),
    "FSYNC_EVERY": get_env("STEP10B_FSYNC_EVERY", 50),
    "FORCE_CLI_ONLY": get_env("STEP10B_FORCE_CLI_ONLY", True),

    "RUNNER_LOG": get_env("STEP10B_RUNNER_LOG", "../logs/step_10b_runner_debug.log"),
}

SHORT_INSTRUCTION = """[TASK]
Classify this review into Flow, Utility, Nostalgia, or None. Return STRICT JSON only.
[RULES]
- Scores in [0,1], sum to 1.0 +/- 0.01.
[OUTPUT]
{
  "appid":"{{APPID}}",
  "reviewId": {{REVIEW_ID_OR_NULL}},
  "Flow": 0..1, "Utility": 0..1, "Nostalgia": 0..1, "None": 0..1,
}
[INPUT]
appid: {{APPID}}
reviewId: {{REVIEW_ID_OR_NULL}}
review: {{REVIEW_TEXT}}
[OUTPUT JSON ONLY]"""


# Centralized log function (stderr, masked)
def clog(msg: str) -> None:
    masked_log(msg, verbose=True)

STOP_REQUESTED = False
def _on_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    clog("[10b] SIGINT received: finishing in-flight requests and stopping...")

def list_ollama_models() -> List[str]:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        names = []
        for line in (proc.stdout or "").splitlines():
            parts = line.strip().split()
            if parts and parts[0]:
                names.append(parts[0])
        return names
    except Exception:
        return []

def model_present_any(model: str) -> bool:
    target = (model or "").strip().lower().split("@")[0]
    names = [n.lower() for n in list_ollama_models()]
    for n in names:
        if n == target or n.startswith(target.split(":")[0]):
            return True
    return False

def cli_pull(model: str, timeout: int) -> bool:
    clog(f"[10b/pull] `ollama pull {model}` …")
    try:
        proc = subprocess.Popen(["ollama", "pull", model], stdout=sys.stderr, stderr=sys.stderr, text=True)
        start = time.time()
        while proc.poll() is None:
            if timeout and (time.time() - start > timeout):
                try: proc.kill()
                except Exception: pass
                clog("[10b/pull] Timed out; aborting pull.")
                return False
            time.sleep(0.5)
        return proc.returncode == 0
    except Exception as ex:
        clog(f"[10b/pull] exception: {ex}")
        return False

def ensure_model_available(model: str) -> bool:
    if model_present_any(model):
        clog(f"[10b] Model `{model}` present.")
        return True
    if model.startswith("phi3"):  # auto-pull phi3 only
        return cli_pull(model, 1800)
    return False

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Step 10b — Ollama runner (Windows-friendly, CLI-only, UTF-8 stdin)")
    ap.add_argument("--profile", type=str, default=os.getenv("OLLAMA_PROFILE", ""), help="fast|medium|complex")
    ap.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", ""), help="Force model name")
    ap.add_argument("--workers", type=int, default=int(os.getenv("OLLAMA_WORKERS", "0")), help="Override workers")
    ap.add_argument("--limit", type=int, default=int(os.getenv("OLLAMA_LIMIT", "0")), help="Process first N prompts; 0=all")
    ap.add_argument("--max-chars", type=int, default=0, help="Truncate review to this many characters; 0=profile default")
    return ap.parse_args()

def load_profile_overrides(profile_name: str) -> Dict[str, Any]:
    path = os.path.join(str(CONFIG["CONFIG_DIR"]), "profiles.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            allp = json.load(f) or {}
        return allp.get(profile_name, {})
    except Exception:
        return {}

def resolve_profile_and_model(args) -> Tuple[str, int, str, int]:
    profile_name = (args.profile or os.getenv("OLLAMA_PROFILE") or CONFIG["DEFAULT_PROFILE"]).strip().lower()
    profiles = CONFIG["PROFILES"]
    if profile_name not in profiles:
        profile_name = CONFIG["DEFAULT_PROFILE"]
    profile = dict(profiles[profile_name])  # copy

    override = load_profile_overrides(profile_name)
    for k in ("candidate_models", "workers", "prompt_variant", "max_chars"):
        if k in override:
            profile[k] = override[k]

    workers = int(args.workers) if int(args.workers) > 0 else int(CONFIG["WORKERS"]) if int(CONFIG["WORKERS"]) > 0 else int(profile["workers"])
    prompt_variant = str(profile.get("prompt_variant", "standard"))
    max_chars = int(profile.get("max_chars", 0))
    if int(getattr(args, "max_chars", 0) or 0) > 0:
        max_chars = int(args.max_chars)

    forced = (args.model or CONFIG.get("MODEL") or "").strip()
    if forced:
        if not ensure_model_available(forced):
            sys.exit(2)
        return forced, workers, prompt_variant, max_chars

    # Prefer phi3 first; then other present models
    candidates = ["phi3", "phi3:latest"] + [m for m in profile.get("candidate_models", []) if m not in ("phi3", "phi3:latest")]
    for mdl in candidates:
        if (not mdl.startswith("phi3")) and (not model_present_any(mdl)):
            continue
        if ensure_model_available(mdl):
            return mdl, workers, prompt_variant, max_chars

    clog("[10b] No candidate models available. Use --model or install with `ollama pull`.")
    sys.exit(2)

def fill_prompt(instruction: str, appid: str, review_id, review_text: str) -> str:
    rid = "null" if review_id is None else str(int(review_id))
    return (instruction
            .replace("{{APPID}}", str(appid))
            .replace("{{REVIEW_ID_OR_NULL}}", rid)
            .replace("{{REVIEW_TEXT}}", review_text))

def _load_template(prompt_variant: str, inputs_dir: str) -> str:
    pdir = os.path.join(inputs_dir, "prompts")
    fname = "prompt_fast.txt" if prompt_variant == "short" else "prompt_nuanced.txt"
    path = os.path.join(pdir, fname)
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return SHORT_INSTRUCTION

def make_prompt_for_record(rec: dict, prompt_variant: str, max_chars: int, inputs_dir: str) -> str:
    appid = str(rec.get("appid", ""))
    rid = rec.get("reviewId", None)
    text = rec.get("review", rec.get("text", ""))
    if max_chars and isinstance(text, str) and len(text) > max_chars:
        text = text[:max_chars]
    instr = _load_template(prompt_variant, inputs_dir)
    return fill_prompt(instr, appid, rid, text)

def call_ollama_run(model: str, prompt: str, timeout: int) -> Tuple[int, str, str]:
    """
    Run `ollama run <model>` while sending the full prompt via STDIN as UTF-8 BYTES.
    This avoids Windows console code page issues (cp850/cp437) with non-ASCII characters.
    """
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8", errors="strict"),  # <-- send bytes, not text
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out, err

def parse_first_json(s: str) -> Optional[dict]:
    depth = 0; start = -1; in_str = False; esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        frag = s[start:i+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            pass
    return None

def writer_open_append(path: str):
    ensure_dir(Path(path).parent)
    return open(path, "a", buffering=1, encoding="utf-8")  # line-buffered

def write_debug(log_path: str, text: str) -> None:
    ensure_dir(Path(log_path).parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def safe_write_jsonl_line(fp, obj, fsync_every, state):
    line = json.dumps(obj, ensure_ascii=True)
    fp.write(line + "\n")
    fp.flush()
    state["written"] += 1
    if fsync_every > 0 and (state["written"] % fsync_every == 0):
        os.fsync(fp.fileno())

def load_processed_keys(responses_path: str) -> set:
    processed = set()
    p = Path(responses_path)
    if not p.exists():
        return processed
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            appid = str(obj.get("appid", ""))
            rid = obj.get("reviewId", None)
            key = f"{appid}|{rid if rid is not None else 'NULL'}"
            processed.add(key)
    return processed

def format_eta(done, total, start_ts):
    if done == 0 or total == 0:
        return "ETA: --:--"
    elapsed = time.time() - start_ts
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    mm = int(remaining // 60); ss = int(remaining % 60)
    return f"ETA: {mm:02d}:{ss:02d}"

def main() -> None:
    # Ctrl+C handler
    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass

    args = parse_args()

    prompts_path = Path(str(CONFIG["PROMPTS_JSONL"]))
    out_path = Path(str(CONFIG["RESPONSES_JSONL"]))
    if not prompts_path.exists():
        clog(f"[10b] Missing {prompts_path}")
        sys.exit(2)

    # Load prompts
    records: List[dict] = []
    with prompts_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if "appid" in obj and ("review" in obj or "text" in obj):
                obj["appid"] = str(obj["appid"])
                records.append(obj)

    # Limit for smoke tests
    limit = args.limit if args.limit and args.limit > 0 else CONFIG["LIMIT_ITEMS"]
    if limit is not None:
        records = records[: int(limit)]

    # Resume skip
    if bool(CONFIG.get("RESUME_FROM_EXISTING", True)):
        done_keys = load_processed_keys(str(out_path))
        if done_keys:
            def rk(rec): return f"{str(rec.get('appid',''))}|{rec.get('reviewId','NULL') if rec.get('reviewId') is not None else 'NULL'}"
            before = len(records)
            records = [r for r in records if rk(r) not in done_keys]
            clog(f"[10b] Resume: {before-len(records)} already done; {len(records)} pending.")

    if not records:
        clog("[10b] Nothing to do (no prompts or all done).")
        return

    model, workers, prompt_variant, max_chars = resolve_profile_and_model(args)
    fsync_every = int(CONFIG.get("FSYNC_EVERY", 50))
    state = {"done": 0, "written": 0}
    start_ts = time.time()
    clog(f"[10b] Start: model={model} workers={workers} n={len(records)} prompt_variant={prompt_variant} max_chars={max_chars}")

    with writer_open_append(str(out_path)) as fout:
        try:
            def _do_one(rec: dict) -> dict:
                try:
                    prompt = make_prompt_for_record(rec, prompt_variant, max_chars=max_chars, inputs_dir=str(CONFIG["INPUTS_DIR"]))
                    rc, out, err = call_ollama_run(model, prompt, timeout=int(CONFIG["REQUEST_TIMEOUT_SEC"]))
                    if rc != 0:
                        write_debug(str(CONFIG["RUNNER_LOG"]), f"[call] rc={rc}\nstdout:\n{out}\nstderr:\n{err}\n---\nPROMPT_HEAD:\n{prompt[:800]}\n")
                        return {"appid": str(rec.get("appid","")), "reviewId": rec.get("reviewId", None),
                                "_error": f"ollama run rc={rc}", "_stderr": (err or out)[:240]}
                    # Parse JSON (strict or first object)
                    try:
                        obj = json.loads(out)
                    except Exception:
                        obj = parse_first_json(out)
                    if not isinstance(obj, dict):
                        write_debug(str(CONFIG["RUNNER_LOG"]), f"[parse] non-JSON\nstdout:\n{out[:1200]}\n")
                        return {"appid": str(rec.get("appid","")), "reviewId": rec.get("reviewId", None),
                                "_error": "non-JSON output"}
                    obj.setdefault("appid", str(rec.get("appid","")))
                    obj.setdefault("reviewId", rec.get("reviewId", None))
                    return obj
                except Exception as e:
                    write_debug(str(CONFIG["RUNNER_LOG"]), f"[ex] {e}")
                    return {"appid": str(rec.get("appid","")), "reviewId": rec.get("reviewId", None), "_error": f"{str(e)[:512]}"}

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_do_one, r) for r in records]
                next_hb = time.time() + CONFIG["HEARTBEAT_SEC"]
                total = len(futs)
                for fut in as_completed(futs):
                    res = fut.result()
                    safe_write_jsonl_line(fout, res, fsync_every, state)
                    state["done"] += 1
                    pct = 100.0 * state["done"] / total
                    eta = format_eta(state["done"], total, start_ts)
                    clog(f"[ok ] {state['done']}/{total} ({pct:.1f}%) {eta}")
                    if time.time() >= next_hb:
                        clog(f"[hb ] {state['done']}/{total} ({pct:.1f}%) {eta}")
                        next_hb = time.time() + CONFIG["HEARTBEAT_SEC"]
        except KeyboardInterrupt:
            clog("[10b] Ctrl+C: finishing in-flight...")
        finally:
            try: os.fsync(fout.fileno())
            except Exception: pass

    clog(f"[10b] Wrote {state['written']} lines -> {out_path}")

if __name__ == "__main__":
    main()