# pipeline/13_llm_cluster_scoring.py
# -*- coding: utf-8 -*-

'''
Step 13: LLM Cluster Scoring - Scores FUN dimensions at the cluster level using LLMs and merges results.

• Runs LLMs to score clusters for Flow, Utility, Nostalgia, None
• Enforces JSON schema validation and error handling
• Supports retrying failed entries and merging results

Inputs: Cluster representatives, summaries, embeddings, descriptive stats CSVs.
Outputs: Cluster-level scoring tables, merged results, logs.
'''
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from pipeline.Z_utils.common import ensure_dir, log  # project utils

# -----------------------------------------------------------------------------
# Environment & defaults
# -----------------------------------------------------------------------------
SEED = 7

@dataclass
class Env:
    ROOT: Path
    DATA: Path
    # Inputs
    EMB_PATH: Path
    REPS_CSV: Path
    SUMM_CSV: Path
    PER_APP_DESC: Path  # step 08 per_app_descriptive_stats.csv (for pairIndex/type)
    # Work dirs
    OUT_DIR: Path
    OUT_REQ_DIR: Path
    OUT_RESP_DIR: Path
    OUT_TABLES_DIR: Path
    # Outputs
    PROMPTS_JSONL: Path
    RESPONSES_JSONL: Path
    PER_APP_CSV: Path
    PER_PAIR_CSV: Path
    LLM_VS_DICT_CSV: Path
    METADATA_JSON: Path
    # Optional mirror to step10 tables (for 99_latex_pack defaults)
    STEP10_TABLES_DIR: Path
    STEP10_PER_APP: Path
    STEP10_PER_PAIR: Path


def get_env() -> Env:
    root = Path(__file__).parent.parent.resolve()
    data = root / "data"
    # inputs
    emb_path = data / "analysis" / "11" / "embeddings.parquet"
    reps_csv = data / "analysis" / "12" / "cluster_reps.csv"
    summ_csv = data / "analysis" / "12" / "cluster_summaries.csv"
    per_app_desc = data / "analysis" / "08" / "tables" / "per_app_descriptive_stats.csv"

    out_dir = data / "analysis" / "13"
    out_req = out_dir / "requests"
    out_resp = out_dir / "responses"
    out_tables = out_dir / "tables"

    prompts = out_req / "prompts.jsonl"
    responses = out_resp / "responses.jsonl"
    per_app_csv = out_tables / "per_app_llm_agg.csv"
    per_pair_csv = out_tables / "per_pair_deltas_llm.csv"
    llm_vs_dict = out_tables / "llm_vs_dict_shares.csv"
    metadata_json = out_dir / "metadata.json"

    step10_tables = data / "analysis" / "10" / "tables"
    step10_per_app = step10_tables / "per_app_llm_agg.csv"
    step10_per_pair = step10_tables / "per_pair_deltas_llm.csv"

    return Env(
        ROOT=root, DATA=data,
        EMB_PATH=emb_path, REPS_CSV=reps_csv, SUMM_CSV=summ_csv, PER_APP_DESC=per_app_desc,
        OUT_DIR=out_dir, OUT_REQ_DIR=out_req, OUT_RESP_DIR=out_resp, OUT_TABLES_DIR=out_tables,
        PROMPTS_JSONL=prompts, RESPONSES_JSONL=responses,
        PER_APP_CSV=per_app_csv, PER_PAIR_CSV=per_pair_csv,
        LLM_VS_DICT_CSV=llm_vs_dict, METADATA_JSON=metadata_json,
        STEP10_TABLES_DIR=step10_tables, STEP10_PER_APP=step10_per_app, STEP10_PER_PAIR=step10_per_pair
    )


    from pipeline.Z_utils.common import find_project_root  # project utils

    # Always resolve data paths to root, regardless of working directory
DEFAULTS = {
    "model": "phi3",                 # or 'qwen2:7b-instruct-q4_0', 'llama3.1:8b-instruct', etc.
    "host": "http://localhost:11434",
    "timeout": 120,
    "workers": 2,
    "source": "auto",                # auto | summaries | reps
    "limit": None,                   # None or int
    "mirror_to_step10": True,        # mirror per_app/per_pair to step10/tables
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _sha1_str(s: str) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def _confidence_diagnostics(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "confidence" not in df.columns or df["confidence"].dropna().empty:
        return out
    c = pd.to_numeric(df["confidence"], errors="coerce")
    c = c[(c >= 0.0) & (c <= 1.0)].dropna()
    if c.empty:
        return out
    out["count"] = float(len(c))
    out["mean"] = float(c.mean())
    out["std"] = float(c.std(ddof=1)) if len(c) > 1 else 0.0
    out["min"] = float(c.min())
    out["p10"] = float(c.quantile(0.10))
    out["p50"] = float(c.quantile(0.50))
    out["p90"] = float(c.quantile(0.90))
    out["max"] = float(c.max())
    return out


# -----------------------------------------------------------------------------
# Loading inputs
# -----------------------------------------------------------------------------
def read_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        # fallback for JSONL (Step 11 fallback)
        alt = path.with_suffix(".jsonl")
        if alt.exists():
            rows = []
            with alt.open("r", encoding="utf-8") as fh:
                for ln in fh:
                    if ln.strip():
                        rows.append(json.loads(ln))
            if not rows:
                raise SystemExit(f"[13][fatal] Empty embeddings JSONL: {alt.as_posix()}")
            df = pd.DataFrame(rows)
        else:
            raise SystemExit(f"[13][fatal] Embeddings not found: {path.as_posix()} (or {alt.as_posix()})")
    else:
        df = pd.read_parquet(path)
    need = {"appid", "reviewId", "text"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[13][fatal] Embeddings missing columns: {sorted(miss)}")
    return df[["appid", "reviewId", "text"]].copy()


def read_clusters(env: Env) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reps = pd.read_csv(env.REPS_CSV, encoding="utf-8") if env.REPS_CSV.exists() else pd.DataFrame()
    sums = pd.read_csv(env.SUMM_CSV, encoding="utf-8") if env.SUMM_CSV.exists() else pd.DataFrame()
    if reps.empty and sums.empty:
        raise SystemExit("[13][fatal] No cluster inputs found (neither reps nor summaries).")
    return reps, sums


def build_units(env: Env, source: str, limit: Optional[int]) -> pd.DataFrame:
    """
    Return a dataframe of scoring units with columns:
      appid, unit_id, level, text
    level ∈ {'cluster','review'}
    """
    reps, sums = read_clusters(env)
    emb = read_embeddings(env.EMB_PATH)

    # Build from summaries
    df_s = pd.DataFrame(columns=["appid", "unit_id", "level", "text"])
    if not sums.empty:
        # sums: appid, cluster_id, summary_text, summary_n_reviews, source_reviewIds
        ss = sums.copy()
        ss["appid"] = ss["appid"].astype(str)
        ss["unit_id"] = ss.apply(lambda r: f"app:{r['appid']}-cluster:{int(r['cluster_id'])}", axis=1)
        ss["level"] = "cluster"
        ss["text"] = ss["summary_text"].astype(str).fillna("")
        df_s = ss[["appid", "unit_id", "level", "text"]].copy()

    # Build from representatives (single review text)
    df_r = pd.DataFrame(columns=["appid", "unit_id", "level", "text"])
    if not reps.empty:
        rr = reps.copy()
        rr["appid"] = rr["appid"].astype(str)
        rr["unit_id"] = rr.apply(lambda r: f"review:{str(r['rep_reviewId'])}", axis=1)
        rr["level"] = "review"
        # Join to get the actual review text
        emb2 = emb.copy()
        emb2["appid"] = emb2["appid"].astype(str)
        # Map rep_reviewId -> text
        rr = rr.merge(
            emb2[["appid", "reviewId", "text"]],
            left_on=["appid", "rep_reviewId"],
            right_on=["appid", "reviewId"],
            how="left"
        )
        rr["text"] = rr["text"].astype(str).fillna("")
        df_r = rr[["appid", "unit_id", "level", "text"]].copy()

    # Select by source
    if source == "summaries":
        base = df_s
    elif source == "reps":
        base = df_r
    else:  # auto: prefer summaries if available, else reps
        base = df_s if not df_s.empty else df_r

    # Fallback: if auto selected summaries but they're empty, fallback to reps
    if base.empty and source == "auto":
        base = df_r

    if base.empty:
        raise SystemExit(f"[13][fatal] No scoring units found for source='{source}'")

    # Clean/trim
    base["text"] = base["text"].fillna("").map(lambda s: s.strip())
    # Drop near-empty texts
    base = base[base["text"].map(lambda s: len(s) >= 3)].copy()

    # Deterministic order (appid, unit_id asc)
    base.sort_values(["appid", "unit_id"], inplace=True)

    if limit is not None:
        base = base.head(int(limit)).copy()

    return base.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Prompt & schema (Structured Output)
# -----------------------------------------------------------------------------
def build_system_prompt() -> str:
    # Try to load prompt from analysis_inputs/prompt_fast.txt or prompt_nuanced.txt if available
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_dir = os.path.join(base_dir, "analysis_inputs", "prompts")
    fast_path = os.path.join(prompt_dir, "prompt_fast.txt")
    nuanced_path = os.path.join(prompt_dir, "prompt_nuanced.txt")
    prompt = None
    chosen_path = None
    for p in [fast_path, nuanced_path]:
        if os.path.exists(p):
            chosen_path = p
            break
    if chosen_path:
        with open(chosen_path, "r", encoding="utf-8") as fh:
            prompt = fh.read().strip()
        return prompt
    else:
        log(f"[13][fatal] No prompt file found at {fast_path} or {nuanced_path}. Aborting.")
        raise SystemExit(f"[13][fatal] No prompt file found at {fast_path} or {nuanced_path}.")


def build_user_prompt(appid: str, unit_id: str, level: str, text: str) -> str:
    # Keep concise—models follow structure better with short context
    header = f"APPID={appid} | UNIT={unit_id} | LEVEL={level}\n"
    instruction = "Text to rate:\n"
    return header + instruction + text


SCHEMA = {
    "type": "object",
    "properties": {
        "appid": {"type": "string"},
        "unit_id": {"type": "string"},
        "level": {"type": "string", "enum": ["review", "cluster"]},
        "Flow": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "Utility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "Nostalgia": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "None": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["appid", "unit_id", "level", "Flow", "Utility", "Nostalgia", "None"],
}

# NOTE: Ollama's structured outputs `format` parameter enforces this schema,
# returning JSON we can directly parse. See:
# - https://ollama.com/blog/structured-outputs
# - (Python examples) community docs about using `format` with Pydantic schemas.


# -----------------------------------------------------------------------------
# Ollama call (structured output enforced)
# -----------------------------------------------------------------------------

def call_ollama(
    *, host: str, model: str, system: str, user: str, timeout: int = 120, seed: int = SEED, max_retries: int = 3, raw_fail_log_path: Optional[Path] = None, meta: Optional[Dict] = None
) -> Dict:
    # Combine system and user prompt into a single user message for better reliability
    combined_prompt = system.strip() + "\n\n" + user.strip()
    # Prompt length check (Ollama context window is typically 4k-8k tokens)
    if len(combined_prompt) > 8000:
        log(f"[13][warn] Prompt too long ({len(combined_prompt)} chars), truncating to 8000.")
        combined_prompt = combined_prompt[:8000]
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{host.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": combined_prompt},
                    ],
                    "format": SCHEMA,
                    "options": {
                        "temperature": 0,
                        "seed": seed,
                        "num_predict": 128
                    },
                    "keep_alive": "30m",
                    "stream": False,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            content = r.json()["message"]["content"]
            try:
                return json.loads(content)
            except Exception as parse_exc:
                # Log raw output if requested
                if raw_fail_log_path is not None:
                    from datetime import datetime
                    rec = {
                        "error": str(parse_exc),
                        "raw_content": content,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if meta:
                        rec.update(meta)
                    ensure_dir(raw_fail_log_path.parent)
                    with raw_fail_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raise
        except Exception as e:
            if attempt < max_retries:
                log(f"[13][retry] LLM call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
                import time as _time; _time.sleep(delay)
                delay *= 2
            else:
                log(f"[13][error] LLM call failed after {max_retries} attempts: {e}")
                # If this is a parse error, raw output already logged above
                # For other errors (e.g., network), optionally log meta
                if raw_fail_log_path is not None:
                    from datetime import datetime
                    rec = {
                        "error": str(e),
                        "raw_content": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if meta:
                        rec.update(meta)
                    ensure_dir(raw_fail_log_path.parent)
                    with raw_fail_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raise


def renormalize(rec: Dict) -> Dict:
    """Ensure Flow+Utility+Nostalgia+None == 1.0 with clamping and scaling.

    - Values are clamped to [0,1].
    - If the sum is <= 0, defaults to None=1.0.
    - Otherwise rescale so the sum is exactly 1.0 (within float precision).
    """
    keys = ["Flow", "Utility", "Nostalgia", "None"]
    vals = [max(0.0, min(1.0, float(rec.get(k, 0.0)))) for k in keys]
    s = sum(vals)
    if s <= 0:
        # default to None=1.0 if everything zero/invalid
        vals = [0.0, 0.0, 0.0, 1.0]
        s = 1.0
    # scale to sum=1
    vals = [v / s for v in vals]
    for k, v in zip(keys, vals):
        rec[k] = float(v)
    # confidence clamp
    if "confidence" in rec:
        try:
            rec["confidence"] = float(rec["confidence"])
        except Exception:
            rec["confidence"] = 0.0
    rec["confidence"] = max(0.0, min(1.0, rec.get("confidence", 0.0)))
    return rec


def write_jsonl_line(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def score_units(
    df_units: pd.DataFrame, *, model: str, host: str, timeout: int, workers: int,
    prompts_path: Path, responses_path: Path, raw_fail_log_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    df_units: columns [appid, unit_id, level, text]
    """
    ensure_dir(prompts_path.parent)
    ensure_dir(responses_path.parent)

    # Write prompts for reproducibility/debug
    with prompts_path.open("w", encoding="utf-8") as fh:
        for _, r in df_units.iterrows():
            prompt_obj = {
                "appid": str(r["appid"]),
                "unit_id": str(r["unit_id"]),
                "level": str(r["level"]),
                "prompt": build_user_prompt(str(r["appid"]), str(r["unit_id"]), str(r["level"]), r["text"]),
            }
            fh.write(json.dumps(prompt_obj, ensure_ascii=False) + "\n")

    system_prompt = build_system_prompt()

    def _row_get(row, key):
        # Support both dict and namedtuple
        if isinstance(row, dict):
            return row[key]
        return getattr(row, key)

    def _task(row) -> Dict:
        appid = str(_row_get(row, "appid"))
        unit_id = str(_row_get(row, "unit_id"))
        level = str(_row_get(row, "level"))
        user_prompt = build_user_prompt(appid, unit_id, level, _row_get(row, "text"))
        meta = {"appid": appid, "unit_id": unit_id, "level": level, "model": model}
        try:
            out = call_ollama(
                host=host, model=model, system=system_prompt, user=user_prompt, timeout=timeout, seed=SEED, max_retries=3,
                raw_fail_log_path=raw_fail_log_path, meta=meta
            )
            # Attach identifiers & model info so ingest is trivial
            out["appid"] = appid
            out["unit_id"] = unit_id
            out["level"] = level
            out["model"] = model
            # Validate & renormalize shares to ensure strict comparability
            out = renormalize(out)
            # Post-check bounds (defensive): clamp again if small numeric drift
            for k in ["Flow","Utility","Nostalgia","None","confidence"]:
                if k in out:
                    try:
                        v = float(out[k])
                        out[k] = max(0.0, min(1.0, v)) if k != "confidence" else max(0.0, min(1.0, v))
                    except Exception:
                        pass
            return {"ok": True, "rec": out}
        except Exception as e:
            return {"ok": False, "error": str(e), "appid": appid, "unit_id": unit_id, "level": level}

    total = len(df_units)
    start = time.time()

    def _task_with_log(idx, row):
        res = _task(row)
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - (idx + 1)) / rate if rate > 0 else float('inf')
            log(f"[13] progress {idx+1}/{total} | {rate:.2f} u/s | ETA {remaining/60:.1f} min")
        return res

    results = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = [ex.submit(_task_with_log, i, r) for i, r in enumerate(df_units.itertuples(index=False), start=0)]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for i, r in enumerate(df_units.itertuples(index=False), start=0):
            results.append(_task_with_log(i, r._asdict()))

    # Persist responses JSONL and build DataFrame
    ok_rows, err_rows = [], []
    for res in results:
        if res.get("ok"):
            rec = res["rec"]
            write_jsonl_line(responses_path, rec)
            ok_rows.append(rec)
        else:
            err_rows.append(res)

    if err_rows:
        dbg_path = responses_path.parent / "errors.jsonl"
        with dbg_path.open("w", encoding="utf-8") as fh:
            for er in err_rows:
                fh.write(json.dumps(er, ensure_ascii=False) + "\n")
        log(f"[13][warn] {len(err_rows)} errors written to {dbg_path.as_posix()}")
        # Summarize errors at end
        error_types = {}
        for er in err_rows:
            etype = str(er.get('error', 'unknown')).split(':')[0]
            error_types[etype] = error_types.get(etype, 0) + 1
        log(f"[13][summary] Error types: {error_types}")

    if not ok_rows:
        raise SystemExit("[13][fatal] No successful responses from model.")

    df_scores = pd.DataFrame(ok_rows)
    # columns: appid, unit_id, level, Flow, Utility, Nostalgia, None, confidence, model
    return df_scores

def per_app_aggregate(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LLM-scored units to per-app metrics aligned with dictionary outputs.

    Outputs (shares on 0..1 scale)
    ------------------------------
    - Flow_share_mean, Utility_share_mean, Nostalgia_share_mean, None_share_mean
    - confidence_mean (0..1), confidence_var (0..1^2)
    - n_reviews (or clusters) per app
    """
    cols = ["Flow", "Utility", "Nostalgia", "None", "confidence"]
    for c in cols:
        if c in df_scores.columns:
            df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

    # Defensive renormalization at record level
    for i in range(len(df_scores)):
        df_scores.loc[i, ["Flow","Utility","Nostalgia","None"]] = pd.Series(
            renormalize({
                "Flow": df_scores.loc[i, "Flow"],
                "Utility": df_scores.loc[i, "Utility"],
                "Nostalgia": df_scores.loc[i, "Nostalgia"],
                "None": df_scores.loc[i, "None"],
                "confidence": df_scores.loc[i, "confidence"] if "confidence" in df_scores.columns else 0.0,
            })
        )[ ["Flow","Utility","Nostalgia","None"] ]

    df_scores["appid"] = df_scores["appid"].astype(str)
    grp = df_scores.groupby("appid", as_index=False)
    means = grp[["Flow","Utility","Nostalgia","None","confidence"]].mean()
    counts_series = grp.size()
    counts = counts_series.reset_index()
    # Rename last column (count) to n_reviews for compatibility with older pandas
    if len(counts.columns) >= 2:
        counts = counts.rename(columns={counts.columns[-1]: 'n_reviews'})
    conf_var = df_scores.groupby("appid", as_index=False)[["confidence"]].var(ddof=1).rename(columns={"confidence":"confidence_var"})
    out = means.merge(counts, on="appid", how="left").merge(conf_var, on="appid", how="left")
    out.rename(columns={
        "Flow": "Flow_share_mean",
        "Utility": "Utility_share_mean",
        "Nostalgia": "Nostalgia_share_mean",
        "None": "None_share_mean",
        "confidence": "confidence_mean",
    }, inplace=True)
    if 'index' in out.columns:
        out = out.drop(columns=['index'])
    out.sort_values("appid", inplace=True)
    return out


def per_pair_deltas(env: Env, per_app: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not env.PER_APP_DESC.exists():
        log(f"[13][info] per_app_descriptive_stats.csv not found; skipping pair deltas.")
        return None
    meta = pd.read_csv(env.PER_APP_DESC, encoding="utf-8")
    if not {"appid", "pairIndex", "type"}.issubset(set(meta.columns)):
        log(f"[13][info] per_app_descriptive_stats.csv missing columns; skipping pair deltas.")
        return None
    # Prepare joins
    meta["appid"] = meta["appid"].astype(str)
    per_app2 = per_app.copy()
    per_app2["appid"] = per_app2["appid"].astype(str)

    merged = meta[["pairIndex", "type", "appid"]].merge(
        per_app2, on="appid", how="left"
    )
    # Split top/bottom
    top = merged[merged["type"] == "top"].copy()
    bot = merged[merged["type"] == "bottom"].copy()
    # Join on pairIndex
    j = top.merge(
        bot,
        on="pairIndex",
        suffixes=("_top", "_bottom"),
        how="inner",
    )
    if j.empty:
        log("[13][info] No matching top/bottom rows for deltas.")
        return None
    # Determine column set to compute diffs on
    share_cols = [
        ("Flow_share_mean", "Flow"),
        ("Utility_share_mean", "Utility"),
        ("Nostalgia_share_mean", "Nostalgia"),
        ("None_share_mean", "None"),
    ]
    use_prefix = None  # either *_share_mean or legacy names
    if all((f"{c}_top" in j.columns and f"{c}_bottom" in j.columns) for c, _ in share_cols):
        # New naming with *_share_mean present pre-merge
        use_prefix = "_share_mean"
        def col(name):
            return name
    elif all((f"{legacy}_top" in j.columns and f"{legacy}_bottom" in j.columns) for _, legacy in share_cols):
        use_prefix = "legacy"
        def col(name):
            # map e.g., Flow_share_mean -> Flow
            return name.replace("_share_mean", "")
    else:
        log("[13][info] per_pair_deltas: expected share columns not found; skipping deltas.")
        return None

    def top(name):
        return j[f"{col(name)}_top"]
    def bottom(name):
        return j[f"{col(name)}_bottom"]

    diffs = pd.DataFrame({
        "pairIndex": j["pairIndex"],
        "Flow_diff_top_minus_bottom": top("Flow_share_mean") - bottom("Flow_share_mean"),
        "Utility_diff_top_minus_bottom": top("Utility_share_mean") - bottom("Utility_share_mean"),
        "Nostalgia_diff_top_minus_bottom": top("Nostalgia_share_mean") - bottom("Nostalgia_share_mean"),
        "None_diff_top_minus_bottom": top("None_share_mean") - bottom("None_share_mean"),
        # confidence is not a share; use mean confidence if available
        "confidence_diff_top_minus_bottom": (j["confidence_top"] - j["confidence_bottom"]) if ("confidence_top" in j.columns and "confidence_bottom" in j.columns) else 0.0,
        "appid_top": j["appid_top"], "appid_bottom": j["appid_bottom"],
    })
    diffs.sort_values("pairIndex", inplace=True)
    return diffs


def mirror_to_step10(env: Env, *, per_app: pd.DataFrame, per_pair: Optional[pd.DataFrame]) -> None:
    ensure_dir(env.STEP10_TABLES_DIR)
    per_app.to_csv(env.STEP10_PER_APP, index=False, encoding="utf-8")
    log(f"[13] Mirrored per_app -> {env.STEP10_PER_APP.as_posix()}")
    if per_pair is not None:
        per_pair.to_csv(env.STEP10_PER_PAIR, index=False, encoding="utf-8")
        log(f"[13] Mirrored per_pair -> {env.STEP10_PER_PAIR.as_posix()}")


def build_correlation_ready(env: Env, per_app_llm: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Join Step 09 dictionary shares with Step 13 LLM shares for correlation.

    Expected Step 09 source: data/analysis/09/per_app_review_agg.csv with columns:
      appid, Flow_share_mean, Utility_share_mean, Nostalgia_share_mean, None_share_mean (0..1)
    """
    dict_path = env.DATA / "analysis" / "09" / "per_app_review_agg.csv"
    if not dict_path.exists():
        log(f"[13][info] Dict per_app_review_agg.csv not found at {dict_path.as_posix()}; skipping correlation CSV.")
        return None
    d = pd.read_csv(dict_path, encoding="utf-8")
    if "appid" not in d.columns:
        log("[13][info] per_app_review_agg.csv missing 'appid'; skipping correlation CSV.")
        return None
    # Select only share means (0..1) and rename with _dict suffix
    share_cols = ["Flow_share_mean","Utility_share_mean","Nostalgia_share_mean","None_share_mean"]
    have = [c for c in share_cols if c in d.columns]
    if not have:
        log("[13][info] per_app_review_agg.csv has no *_share_mean columns; skipping correlation CSV.")
        return None
    d = d[["appid"] + have].copy()
    d["appid"] = d["appid"].astype(str)
    d = d.rename(columns={c: c.replace("_share_mean","_share_dict") for c in have})

    # Prepare LLM per_app shares
    l = per_app_llm.copy()
    l["appid"] = l["appid"].astype(str)
    l_cols = ["Flow_share_mean","Utility_share_mean","Nostalgia_share_mean","None_share_mean"]
    l = l[["appid"] + [c for c in l_cols if c in l.columns]].copy()
    l = l.rename(columns={c: c.replace("_share_mean","_share_llm") for c in l.columns if c.endswith("_share_mean")})

    j = d.merge(l, on="appid", how="inner")
    if j.empty:
        log("[13][info] No appid overlap between dict and LLM outputs; correlation CSV would be empty.")
        return None
    j.sort_values("appid", inplace=True)
    j.to_csv(env.LLM_VS_DICT_CSV, index=False, encoding="utf-8")
    log(f"[13] Wrote correlation-ready CSV -> {env.LLM_VS_DICT_CSV.as_posix()}")
    return j


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_error_entries(env: Env) -> List[Tuple[str, str, str]]:
    """
    Parse error logs to extract (appid, unit_id, level) of failed entries.
    Looks in data/analysis/13/responses/errors.jsonl and logs/step_13_*.log.
    """
    error_keys = set()
    # 1. Check errors.jsonl (preferred)
    err_path = env.OUT_RESP_DIR / "errors.jsonl"
    if err_path.exists():
        with err_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    obj = json.loads(ln)
                    appid = str(obj.get("appid", ""))
                    unit_id = str(obj.get("unit_id", ""))
                    level = str(obj.get("level", ""))
                    if appid and unit_id and level:
                        error_keys.add((appid, unit_id, level))
                except Exception:
                    continue
    # 2. Optionally, parse logs/step_13_*.log for [error] or [retry] lines
    log_dir = Path(env.ROOT) / "pipeline" / "logs"
    if log_dir.exists():
        for logf in log_dir.glob("step_13_*.log"):
            try:
                with logf.open("r", encoding="utf-8") as fh:
                    for ln in fh:
                        if "[error]" in ln or "[retry]" in ln:
                            # Try to extract appid/unit_id/level from line
                            import re
                            m = re.search(r"appid=([\w\-]+).*unit_id=([\w\-:]+).*level=([\w]+)", ln)
                            if m:
                                error_keys.add((m.group(1), m.group(2), m.group(3)))
            except Exception:
                continue
    return list(error_keys)

def filter_units_by_keys(df_units: pd.DataFrame, keys: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Filter df_units to only those with (appid, unit_id, level) in keys."""
    keyset = set(keys)
    return df_units[df_units.apply(lambda r: (str(r["appid"]), str(r["unit_id"]), str(r["level"])) in keyset, axis=1)].copy()

def merge_new_results(env: Env, new_df: pd.DataFrame) -> None:
    """
    Merge new_df (rechecked results) into the main responses.jsonl and per_app_llm_agg.csv.
    """
    # 1. Update responses.jsonl
    resp_path = env.RESPONSES_JSONL
    if resp_path.exists():
        # Read all old
        old = []
        with resp_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    old.append(json.loads(ln))
                except Exception:
                    continue
        # Remove any with same (appid, unit_id, level) as new
        new_keys = set((str(r["appid"]), str(r["unit_id"]), str(r["level"])) for r in new_df.to_dict("records"))
        old = [r for r in old if (str(r.get("appid","")), str(r.get("unit_id","")), str(r.get("level",""))) not in new_keys]
        # Add new
        all_rows = old + new_df.to_dict("records")
        # Write back
        with resp_path.open("w", encoding="utf-8") as fh:
            for r in all_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 2. Update per_app_llm_agg.csv
    per_app_path = env.PER_APP_CSV
    if per_app_path.exists():
        df = pd.read_csv(per_app_path, encoding="utf-8")
        # Recompute per_app aggregate with new results
        # Read all responses
        all_recs = []
        if resp_path.exists():
            with resp_path.open("r", encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        all_recs.append(json.loads(ln))
                    except Exception:
                        continue
        if all_recs:
            df_scores = pd.DataFrame(all_recs)
            per_app = per_app_aggregate(df_scores)
            per_app.to_csv(per_app_path, index=False, encoding="utf-8")
    # Optionally update deltas and correlation CSVs
    # (skip for now, user can rerun full script for those)

def main(argv: Optional[List[str]] = None) -> None:
    env = get_env()


    ap = argparse.ArgumentParser(description="Step 13 — cluster-level LLM scoring (structured JSON via Ollama)")
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--host", default=DEFAULTS["host"])
    ap.add_argument("--timeout", type=int, default=DEFAULTS["timeout"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--source", choices=["auto", "summaries", "reps"], default=DEFAULTS["source"])
    ap.add_argument("--limit", type=int, default=DEFAULTS["limit"])
    ap.add_argument("--mirror-to-step10", choices=["on", "off"], default="on" if DEFAULTS["mirror_to_step10"] else "off")
    ap.add_argument("--recheck-errors", action="store_true", help="Re-run LLM scoring only for entries that failed in previous runs.")
    args = ap.parse_args(argv)

    log(f"[13] Start — model={args.model} source={args.source} workers={args.workers}")
    log(f"[13] Inputs: reps={env.REPS_CSV.as_posix()} | summaries={env.SUMM_CSV.as_posix()}")
    ensure_dir(env.OUT_REQ_DIR); ensure_dir(env.OUT_RESP_DIR); ensure_dir(env.OUT_TABLES_DIR)

    if args.recheck_errors:
        log("[13] Recheck errors mode: will re-run only failed entries from previous runs.")
        # 1. Parse error logs for failed keys
        error_keys = parse_error_entries(env)
        if not error_keys:
            log("[13][info] No error entries found in logs. Nothing to recheck.")
            return
        # 2. Build all units, then filter to only those with errors
        df_units = build_units(env, source=args.source, limit=None)
        df_units_err = filter_units_by_keys(df_units, error_keys)
        if df_units_err.empty:
            log("[13][info] No matching units found for error keys. Nothing to recheck.")
            return
        log(f"[13] Rechecking {len(df_units_err)} error units.")
        # 3. Warm up model
        try:
            _ = call_ollama(
                host=args.host, model=args.model,
                system="Return {\"ok\": true} only.",
                user="ping",
                timeout=int(args.timeout), seed=SEED
            )
            log("[13] Warm-up complete (model loaded).")
        except Exception as e:
            log(f"[13][warn] Warm-up failed (continuing): {e}")
        # 4. Score only error units, log raw failures
        raw_fail_log_path = env.OUT_RESP_DIR / "responses_failed_raw.jsonl"
        t0 = time.time()
        df_scores = score_units(
            df_units_err,
            model=args.model, host=args.host, timeout=int(args.timeout), workers=int(args.workers),
            prompts_path=env.OUT_REQ_DIR / "prompts_recheck.jsonl", responses_path=env.OUT_RESP_DIR / "responses_recheck.jsonl",
            raw_fail_log_path=raw_fail_log_path
        )
        dt = time.time() - t0
        log(f"[13] Rechecked {len(df_scores)} error responses in {dt:.1f}s")
        # 5. Merge new results into main files
        merge_new_results(env, df_scores)
        # 6. Log which errors were fixed
        fixed_keys = set((str(r["appid"]), str(r["unit_id"]), str(r["level"])) for r in df_scores.to_dict("records"))
        still_failed = set(error_keys) - fixed_keys
        log(f"[13] Fixed {len(fixed_keys)} errors. Still failed: {len(still_failed)}.")
        if still_failed:
            log(f"[13] Still failed keys: {list(still_failed)[:10]}{'...' if len(still_failed)>10 else ''}")
        log(f"[13] Raw failed responses logged to {raw_fail_log_path.as_posix()}")
        log("[13] Done (recheck errors mode).")
        return

    # Default: normal mode
    # Warm up model with a tiny structured request to trigger load/download once
    try:
        _ = call_ollama(
            host=args.host, model=args.model,
            system="Return {\"ok\": true} only.",
            user="ping",
            timeout=int(args.timeout), seed=SEED
        )
        log("[13] Warm-up complete (model loaded).")
    except Exception as e:
        log(f"[13][warn] Warm-up failed (continuing): {e}")

    # Build scoring units
    df_units = build_units(env, source=args.source, limit=args.limit)
    log(f"[13] Scoring units: {len(df_units)}")

    t0 = time.time()
    df_scores = score_units(
        df_units,
        model=args.model, host=args.host, timeout=int(args.timeout), workers=int(args.workers),
        prompts_path=env.PROMPTS_JSONL, responses_path=env.RESPONSES_JSONL
    )
    dt = time.time() - t0
    log(f"[13] Collected {len(df_scores)} responses in {dt:.1f}s")

    # Diagnostics: confidence distribution and anomalies
    conf_diag = _confidence_diagnostics(df_scores)
    if conf_diag:
        log(f"[13][diag] confidence dist: n={conf_diag.get('count',0):.0f} mean={conf_diag.get('mean',0):.3f} "
            f"std={conf_diag.get('std',0):.3f} min={conf_diag.get('min',0):.3f} p10={conf_diag.get('p10',0):.3f} "
            f"p50={conf_diag.get('p50',0):.3f} p90={conf_diag.get('p90',0):.3f} max={conf_diag.get('max',0):.3f}")
    # Check sum anomalies before renormalization (if any survived)
    if {"Flow","Utility","Nostalgia","None"}.issubset(df_scores.columns):
        sums = pd.to_numeric(df_scores["Flow"], errors="coerce").fillna(0) + \
               pd.to_numeric(df_scores["Utility"], errors="coerce").fillna(0) + \
               pd.to_numeric(df_scores["Nostalgia"], errors="coerce").fillna(0) + \
               pd.to_numeric(df_scores["None"], errors="coerce").fillna(0)
        bad = ((sums - 1.0).abs() > 1e-6).sum()
        if bad:
            log(f"[13][warn] {int(bad)} records had shares not summing to 1.0 before renormalization (tolerance 1e-6)")

    # Aggregate per-app -> shares (0..1) aligned with dictionary outputs
    per_app = per_app_aggregate(df_scores)
    per_app.to_csv(env.PER_APP_CSV, index=False, encoding="utf-8")
    log(f"[13] Wrote per-app aggregate -> {env.PER_APP_CSV.as_posix()}")

    # Correlation-ready join with Step 09
    _ = build_correlation_ready(env, per_app)

    # Pairwise deltas (if pairing table exists)
    deltas = per_pair_deltas(env, per_app)
    if deltas is not None:
        deltas.to_csv(env.PER_PAIR_CSV, index=False, encoding="utf-8")
        log(f"[13] Wrote per-pair deltas -> {env.PER_PAIR_CSV.as_posix()}")

    # Optional mirroring for 99_latex_pack defaults
    if args.mirror_to_step10 == "on":
        mirror_to_step10(env, per_app=per_app, per_pair=deltas)

    # Emit metadata for scientific notes & reproducibility
    meta = {
        "model": args.model,
        "host": args.host,
        "seed": SEED,
        "schema_sha1": _sha1_str(json.dumps(SCHEMA, sort_keys=True)),
        "system_prompt_sha1": _sha1_str(build_system_prompt()),
        "workers": int(args.workers),
        "timeout": int(args.timeout),
        "source": args.source,
        "n_units": int(len(df_units)),
        "confidence_diag": conf_diag,
        "notes": [
            "Model outputs may reflect bias; scores are compositional (sum to 1.0) which can mask absolute intensity.",
            "Confidence is model-reported and not calibrated; compare across models with caution.",
            "Shares are normalized to sum exactly to 1.0 per record for comparability with dictionary-based shares.",
        ],
    }
    try:
        ensure_dir(env.METADATA_JSON.parent)
        with env.METADATA_JSON.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        log(f"[13] Wrote metadata -> {env.METADATA_JSON.as_posix()}")
    except Exception as e:
        log(f"[13][warn] Failed to write metadata: {e}")

    log("[13] Done.")


if __name__ == "__main__":
    main()