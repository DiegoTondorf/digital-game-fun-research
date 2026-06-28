#!/usr/bin/env python3
# Step 10 — Export prompts and ingest LLM responses into tables/charts.

'''
Step 10a: LLM Dry Run - Simulates LLM scoring pipeline without actual LLM API calls.

• Tests prompt export and response parsing logic
• No actual LLM API calls are made
• Useful for debugging and validation of pipeline logic

Inputs: Sampled reviews CSV, pairs CSV, prompts JSONL, configuration files.
Outputs: Tables, charts, summary text, debug files (simulated).
'''

import argparse
import json
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import matplotlib
from pipeline.Z_utils.common import log as masked_log, ensure_dir, find_project_root
matplotlib.use("Agg")
import matplotlib.pyplot as plt



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

CONFIG: Dict[str, object] = {
    # Inputs
    "SAMPLED_REVIEWS_CSV": get_env("STEP10_SAMPLED_REVIEWS_CSV", DATA / "reviews_sampled_all.csv"),
    "PAIRS_CSV": get_env("STEP10_PAIRS_CSV", DATA / "steamspy_top_bottom_pairs.csv"),

    # Inputs folder (outside data/)
    "INPUTS_DIR": get_env("STEP10_INPUTS_DIR", ROOT / "analysis_inputs/"),
    "PROMPTS_DIR": get_env("STEP10_PROMPTS_DIR", ROOT / "analysis_inputs/prompts/"),
    "CONFIG_DIR": get_env("STEP10_CONFIG_DIR", ROOT / "analysis_inputs/config/"),

    # Outputs
    "REQUESTS_DIR": get_env("STEP10_REQUESTS_DIR", DATA / "analysis/10/requests/"),
    "RESPONSES_DIR": get_env("STEP10_RESPONSES_DIR", DATA / "analysis/10/responses/"),
    "TABLES_DIR": get_env("STEP10_TABLES_DIR", DATA / "analysis/10/tables/"),
    "CHARTS_DIR": get_env("STEP10_CHARTS_DIR", DATA / "analysis/10/charts/"),
    "DEBUG_DIR": get_env("STEP10_DEBUG_DIR", DATA / "analysis/10/debug/"),
    "SUMMARY_TXT": get_env("STEP10_SUMMARY_TXT", DATA / "analysis/10/summary.txt"),

    # Prompts I/O
    "PROMPTS_JSONL": get_env("STEP10_PROMPTS_JSONL", DATA / "analysis/10/requests/prompts.jsonl"),
    "PROMPT_FAST_SNAPSHOT": get_env("STEP10_PROMPT_FAST_SNAPSHOT", DATA / "analysis/10/requests/prompt_fast_snapshot.txt"),
    "PROMPT_NUANCED_SNAPSHOT": get_env("STEP10_PROMPT_NUANCED_SNAPSHOT", DATA / "analysis/10/requests/prompt_nuanced_snapshot.txt"),

    # Tables
    "PER_REVIEW_CSV": get_env("STEP10_PER_REVIEW_CSV", DATA / "analysis/10/tables/per_review_llm.csv"),
    "PER_APP_CSV": get_env("STEP10_PER_APP_CSV", DATA / "analysis/10/tables/per_app_llm_agg.csv"),
    "PER_PAIR_CSV": get_env("STEP10_PER_PAIR_CSV", DATA / "analysis/10/tables/per_pair_deltas_llm.csv"),

    # Behavior
    "GENERATE_PROMPTS": get_env("STEP10_GENERATE_PROMPTS", True),
    "VALIDATE_AND_INGEST": get_env("STEP10_VALIDATE_AND_INGEST", True),
    "STRICT_JSON": get_env("STEP10_STRICT_JSON", True),
    "TIE_BAND": get_env("STEP10_TIE_BAND", 0.05),
    "SUM_TOL": get_env("STEP10_SUM_TOL", 0.01),
    "NO_CLIP": get_env("STEP10_NO_CLIP", True),
    "VERBOSE": get_env("STEP10_VERBOSE", True),

    # Charts
    "CHART": {"format": get_env("STEP10_CHART_FORMAT", "pdf"), "dpi": get_env("STEP10_CHART_DPI", 120), "style": get_env("STEP10_CHART_STYLE", "default"), "figsize": [8, 5]},
}


# Centralized log function
def log(level: str, msg: str):
    masked_log(f"[{level}] {msg}", verbose=CONFIG.get("VERBOSE", True))

# ======================
# Minimal response rules
# ======================
REQUIRED_RESPONSE_KEYS = {
    "appid", "reviewId", "Flow", "Utility", "Nostalgia", "None",
    "confidence", "dominant", "rationale_brief"
}
ALLOWED_DOM = {"Flow", "Utility", "Nostalgia", "None", "tie"}
FUN_KEYS = ("Flow", "Utility", "Nostalgia", "None")

# ======================
# Helpers
# ======================

def stable_int_id(s: str) -> int:
    return int(md5(s.encode("utf-8")).hexdigest()[:12], 16)

def get_review_text(row: pd.Series) -> str:
    for c in ("review", "text", "content", "body"):
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""

def coerce_review_id(row: pd.Series) -> Optional[int]:
    for c in ("reviewId", "recommendationid", "id"):
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v.isdigit():
                return int(v)
            return stable_int_id(v)
    txt = get_review_text(row)
    app = str(row.get("appid", "")).strip()
    return stable_int_id(f"{app}\n{txt[:128]}") if txt else None

def read_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sampled reviews not found: {path}")
    # Chunked reading for large files
    chunks = []
    for chunk in pd.read_csv(path, encoding="utf-8", chunksize=100000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    if "appid" not in df.columns:
        raise RuntimeError("Input sampled CSV must contain 'appid'.")
    df["appid"] = df["appid"].astype(str)
    if "reviewId" not in df.columns:
        df["reviewId"] = df.apply(coerce_review_id, axis=1)
    if "review" not in df.columns:
        df["review"] = df.apply(get_review_text, axis=1)
    if "charLen" not in df.columns:
        df["charLen"] = df["review"].map(lambda s: len(str(s)) if pd.notna(s) else 0)
    if "isPositive" not in df.columns:
        df["isPositive"] = df.get("voted_up", pd.Series([None] * len(df))).astype("float64")
    df = df.sort_values(by=["appid", "reviewId"], kind="mergesort").reset_index(drop=True)
    return df

def read_pairs(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log("INFO", f"Pairs file not found: {path}")
        return None
    # Chunked reading for large files
    chunks = []
    for chunk in pd.read_csv(path, encoding="utf-8", chunksize=100000):
        chunks.append(chunk)
    p = pd.concat(chunks, ignore_index=True)
    if "appid" not in p.columns:
        return None
    p["appid"] = p["appid"].astype(str)
    if "pairIndex" in p.columns:
        p["pairIndex"] = pd.to_numeric(p["pairIndex"], errors="coerce").astype("Int64")
    else:
        p["pairIndex"] = pd.Series([pd.NA] * len(p), dtype="Int64")
    if "type" not in p.columns:
        p["type"] = ""
    return p.drop_duplicates(subset=["appid"], keep="first")

def _read_prompt_templates(inputs_dir: Path) -> Dict[str, str]:
    fast = inputs_dir / "prompts" / "prompt_fast.txt"
    nuan = inputs_dir / "prompts" / "prompt_nuanced.txt"
    t_fast = fast.read_text(encoding="utf-8") if fast.exists() else ""
    t_nuan = nuan.read_text(encoding="utf-8") if nuan.exists() else ""
    return {"fast": t_fast, "nuanced": t_nuan}

def _is_readable(s: str) -> bool:
    """Deterministic readability check (cheap & offline)."""
    if not s:
        return False
    txt = str(s)
    letters = sum(ch.isalpha() for ch in txt)
    words_alpha = sum(1 for w in txt.split() if any(c.isalpha() for c in w))
    bad = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}'\"-_/\\@#&%+*=<>|") for ch in txt)
    frac_bad = bad / max(len(txt), 1)
    return (words_alpha >= 3) and (letters >= 20) and (frac_bad <= 0.30)

# ======================
# Prompts export
# ======================
def export_prompts(samples: pd.DataFrame,
                   out_path: Path,
                   fast_snap: Path,
                   nuanced_snap: Path,
                   inputs_dir: Path) -> Tuple[int, int]:
    """Write minimal prompts.jsonl with only {appid, reviewId, review}. Snapshot templates."""
    tmpls = _read_prompt_templates(inputs_dir)
    fast_snap.parent.mkdir(parents=True, exist_ok=True)
    fast_snap.write_text(tmpls.get("fast", ""), encoding="utf-8")
    nuanced_snap.write_text(tmpls.get("nuanced", ""), encoding="utf-8")

    n_write = 0
    n_unreadable = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unreadable_rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in samples.iterrows():
            appid = r["appid"]
            rid = int(r["reviewId"]) if pd.notna(r["reviewId"]) else None
            review = str(r["review"]) if pd.notna(r["review"]) else ""
            if not _is_readable(review):
                n_unreadable += 1
                one_line = " ".join(review.split())
                unreadable_rows.append({
                    "appid": appid,
                    "reviewId": rid,
                    "charLen": len(review),
                    "review": review,
                    "review_1line": one_line[:240]
                })
                continue
            obj = {"appid": appid, "reviewId": rid, "review": review}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_write += 1
    dbg_dir = Path(str(CONFIG["DEBUG_DIR"]))
    dbg_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens it nicely
    pd.DataFrame(unreadable_rows).to_csv(dbg_dir / "unreadable.csv", index=False, encoding="utf-8-sig")
    return n_write, n_unreadable

# ======================
# Responses ingestion
# ======================
def _float01(x) -> Optional[float]:
    try:
        v = float(x)
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None

def parse_response_line(line: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        obj = json.loads(line)
    except Exception as ex:
        return None, f"invalid_json:{ex}"
    if not isinstance(obj, dict):
        return None, "not_object"
    missing = [k for k in REQUIRED_RESPONSE_KEYS if k not in obj]
    if missing:
        return None, f"missing_keys:{','.join(missing)}"
    obj["appid"] = str(obj.get("appid"))
    rid = obj.get("reviewId", None)
    if rid is None:
        obj["reviewId"] = None
    else:
        try:
            obj["reviewId"] = int(rid)
        except Exception:
            return None, "reviewId_not_int_or_null"
    for k in FUN_KEYS + ("confidence",):
        v = _float01(obj.get(k))
        if v is None:
            return None, f"out_of_range:{k}"
        obj[k] = v
    dom = str(obj.get("dominant", "")).strip()
    if dom not in ALLOWED_DOM:
        return None, "invalid_dominant"
    rb = obj.get("rationale_brief", "")
    if not isinstance(rb, str):
        return None, "rationale_not_string"
    obj["dominant"] = dom
    return obj, None

def sum_and_renormalize(obj: dict, sum_tol: float, renorm_log: List[dict]) -> Tuple[dict, bool, float]:
    s = sum(obj[k] for k in FUN_KEYS)
    if abs(1.0 - s) < 1e-12:
        return obj, False, s
    if abs(1.0 - s) <= sum_tol:
        if s == 0:
            obj["None"] = 1.0
            for k in ("Flow", "Utility", "Nostalgia"):
                obj[k] = 0.0
            renorm_log.append({"appid": obj["appid"], "reviewId": obj["reviewId"], "reason": "sum_zero_forced_None"})
            return obj, True, s
        for k in FUN_KEYS:
            obj[k] = obj[k] / s
        renorm_log.append({"appid": obj["appid"], "reviewId": obj["reviewId"], "old_sum": s, "action": "renormalized"})
        return obj, True, s
    return obj, False, s

def fix_dominance(obj: dict, tie_band: float, domfix_log: List[dict], mismatch_log: List[dict]) -> dict:
    vals = [(k, obj[k]) for k in FUN_KEYS]
    vals.sort(key=lambda kv: kv[1], reverse=True)
    calc = "tie" if (vals[0][1] - vals[1][1]) <= tie_band else vals[0][0]
    if obj.get("dominant") != calc:
        mismatch_log.append({"appid": obj["appid"], "reviewId": obj["reviewId"], "given": obj.get("dominant"), "calc": calc})
        obj["dominant"] = calc
        domfix_log.append({"appid": obj["appid"], "reviewId": obj["reviewId"], "fixed_to": calc})
    return obj

def ingest_responses(
    responses_path: Path,
    samples: pd.DataFrame,
    tie_band: float,
    sum_tol: float,
    debug_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, Path], List[str]]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    invalid_path = debug_dir / "invalid_responses.jsonl"
    missing_path = debug_dir / "missing_responses.csv"
    unmatched_path = debug_dir / "unmatched_responses.csv"
    renorm_path = debug_dir / "renormalizations.jsonl"
    domfix_path = debug_dir / "dominance_fixes.jsonl"
    mismatch_path = debug_dir / "mismatches.jsonl"

    n_lines = 0
    valid, invalid = [], []
    renorm_log, domfix_log, mismatch_log = [], [], []

    if not responses_path.exists():
        raise FileNotFoundError(f"Responses JSONL not found: {responses_path}")

    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n_lines += 1
            obj, reason = parse_response_line(s)
            if obj is None:
                invalid.append({"reason": reason or "invalid", "line": s})
                continue
            # rationale rule: <=30 tokens, no quotes
            rb = obj.get("rationale_brief", "")
            tokens = len(str(rb).split()) if isinstance(rb, str) else 9999
            if (not isinstance(rb, str)) or ('"' in rb) or (tokens > 30):
                invalid.append({"reason": f"rationale_invalid(len={tokens})", "line": s})
                continue
            obj, changed, s0 = sum_and_renormalize(obj, sum_tol, renorm_log)
            if abs(1.0 - sum(obj[k] for k in FUN_KEYS)) > sum_tol:
                invalid.append({"reason": f"sum_out_of_tolerance:{s0}", "line": s})
                continue
            obj = fix_dominance(obj, tie_band, domfix_log, mismatch_log)
            valid.append(obj)

    # Debug logs
    with invalid_path.open("w", encoding="utf-8") as f:
        for r in invalid: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with renorm_path.open("w", encoding="utf-8") as f:
        for r in renorm_log: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with domfix_path.open("w", encoding="utf-8") as f:
        for r in domfix_log: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with mismatch_path.open("w", encoding="utf-8") as f:
        for r in mismatch_log: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vr = pd.DataFrame(valid)

    # Guard: if empty, make sure required merge keys exist
    if vr.empty:
        vr = pd.DataFrame(columns=["appid", "reviewId"])

    # Deduplicate by (appid, reviewId)
    if not vr.empty:
        vr = vr.drop_duplicates(subset=["appid", "reviewId"], keep="first")

    # Join with samples for readability
    samp = samples[["appid", "reviewId", "review", "charLen", "isPositive"]].copy()
    samp["readable"] = samp["review"].map(_is_readable).astype(bool)
    samp = samp.drop(columns=["review"])

    merged = samp.merge(vr, on=["appid", "reviewId"], how="left")

    # Report missing and unmatched
    missing = merged[merged["Flow"].isna()][["appid", "reviewId"]].copy()
    unmatched = vr.merge(samp[["appid", "reviewId"]], on=["appid", "reviewId"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"][["appid", "reviewId"]].copy()

    if not missing.empty: missing.to_csv(missing_path, index=False, encoding="utf-8")
    else: missing_path.write_text("", encoding="utf-8")
    if not unmatched.empty: unmatched.to_csv(unmatched_path, index=False, encoding="utf-8")
    else: unmatched_path.write_text("", encoding="utf-8")

    counters = {
        "lines_total": n_lines,
        "valid": len(valid),
        "invalid": len(invalid),
        "renormalized": len(renorm_log),
        "dominance_fixed": len(domfix_log),
        "missing": int(missing.shape[0]),
        "unmatched": int(unmatched.shape[0]),
    }
    dbg_paths = {
        "invalid": invalid_path,
        "missing": missing_path,
        "unmatched": unmatched_path,
        "renormalizations": renorm_path,
        "dominance_fixes": domfix_path,
        "mismatches": mismatch_path,
    }
    passthrough_keys: List[str] = []
    if not vr.empty:
        for c in vr.columns:
            if c not in REQUIRED_RESPONSE_KEYS:
                passthrough_keys.append(c)
    return merged, counters, dbg_paths, passthrough_keys

# ======================
# Tables
# ======================
def write_per_review(merged: pd.DataFrame, out_path: Path, passthrough: List[str]) -> pd.DataFrame:
    df = merged.copy()
    df["Flow_llm"] = df["Flow"]; df["Utility_llm"] = df["Utility"]
    df["Nostalgia_llm"] = df["Nostalgia"]; df["None_llm"] = df["None"]
    df["dominant_llm"] = df["dominant"]
    if "pairIndex" not in df.columns: df["pairIndex"] = pd.NA
    if "type" not in df.columns: df["type"] = pd.NA
    base_cols = [
        "appid", "reviewId", "charLen", "isPositive",
        "Flow_llm", "Utility_llm", "Nostalgia_llm", "None_llm",
        "confidence", "dominant_llm", "rationale_brief",
        "readable",
        "pairIndex", "type",
    ]
    for c in base_cols:
        if c not in df.columns: df[c] = pd.NA
    extra_cols = [c for c in sorted(set(passthrough)) if c not in base_cols and c in df.columns]
    pr = df[base_cols + extra_cols].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pr.to_csv(out_path, index=False, encoding="utf-8")
    return pr

def per_app_aggregate(per_review: pd.DataFrame, pairs: Optional[pd.DataFrame], out_path: Path) -> pd.DataFrame:
    df = per_review.copy()
    agg_mean = df.groupby("appid", as_index=False)[["Flow_llm","Utility_llm","Nostalgia_llm","None_llm","confidence"]].mean()
    agg_mean = agg_mean.rename(columns={
        "Flow_llm":"Flow_llm_mean","Utility_llm":"Utility_llm_mean",
        "Nostalgia_llm":"Nostalgia_llm_mean","None_llm":"None_llm_mean",
        "confidence":"confidence_mean"
    })
    dom_counts = df.groupby(["appid", "dominant_llm"], as_index=False).size()
    dom_pivot = dom_counts.pivot(index="appid", columns="dominant_llm", values="size").fillna(0.0)
    dom_pivot = dom_pivot.div(dom_pivot.sum(axis=1), axis=0).fillna(0.0)
    dom_pivot.columns = [f"dominant_rate::{str(c)}" for c in dom_pivot.columns]
    dom_pivot = dom_pivot.reset_index()
    for lab in ["Flow","Utility","Nostalgia","None","tie"]:
        col = f"dominant_rate::{lab}"
        if col not in dom_pivot.columns: dom_pivot[col] = 0.0
    fun_any = df.assign(_fun=(df["None_llm"] < 0.5).astype(float)).groupby("appid", as_index=False)["_fun"].mean()
    fun_any = fun_any.rename(columns={"_fun": "pct_fun_any"})
    pa = agg_mean.merge(dom_pivot, on="appid", how="left").merge(fun_any, on="appid", how="left")
    if pairs is not None:
        pa = pa.merge(pairs[["appid","pairIndex","type"]], on="appid", how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pa.to_csv(out_path, index=False, encoding="utf-8")
    return pa

def per_pair_deltas(pa: pd.DataFrame, out_path: Path):
    if "pairIndex" not in pa.columns or "type" not in pa.columns:
        out_path.write_text("", encoding="utf-8"); return None
    top = pa[pa["type"].astype(str).str.lower()=="top"].copy()
    bot = pa[pa["type"].astype(str).str.lower()=="bottom"].copy()
    mean_cols = ["Flow_llm_mean","Utility_llm_mean","Nostalgia_llm_mean","None_llm_mean","confidence_mean","pct_fun_any"]
    dom_cols = [c for c in pa.columns if c.startswith("dominant_rate::")]
    top = top[["pairIndex","appid"]+mean_cols+dom_cols].rename(columns={"appid":"appid_top"})
    bot = bot[["pairIndex","appid"]+mean_cols+dom_cols].rename(columns={"appid":"appid_bottom"})
    m = top.merge(bot, on="pairIndex", how="inner", suffixes=("_top","_bottom"))
    if m.empty:
        out_path.write_text("", encoding="utf-8"); return None
    for c in mean_cols + dom_cols:
        m[f"delta_{c}"] = m[f"{c}_top"] - m[f"{c}_bottom"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_path, index=False, encoding="utf-8")
    return m

# ======================
# Charts
# ======================
def make_charts(per_review: pd.DataFrame, per_app: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    style = str(CONFIG["CHART"]["style"]); dpi = int(CONFIG["CHART"]["dpi"])
    fmt = str(CONFIG["CHART"]["format"]); figsize = CONFIG["CHART"]["figsize"]
    figsize = (float(figsize[0]), float(figsize[1])) if isinstance(figsize, (list,tuple)) else (8,5)
    try: plt.style.use(style)
    except Exception: pass

    # Histograms
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    cats = ["Flow_llm", "Utility_llm", "Nostalgia_llm", "None_llm"]
    titles = ["Flow", "Utility", "Nostalgia", "None"]
    for ax, col, ttl in zip(axs.flatten(), cats, titles):
        if col in per_review.columns:
            ax.hist(per_review[col].dropna().values, bins=20, color="#4472C4", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_title(ttl); ax.set_xlabel("Score (0–1)"); ax.set_ylabel("Frequency"); ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(charts_dir / f"scores_hist_overall.{fmt}", format=fmt)
    plt.close(fig)

    # Dominant stacked
    order = per_app.sort_values(by=["pairIndex","appid"], kind="mergesort") if "pairIndex" in per_app.columns else per_app.sort_values(by=["appid"], kind="mergesort")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bottom = pd.Series([0.0]*len(order), index=order.index)
    palette = {
        "dominant_rate::Flow": "#2E7FE8", "dominant_rate::Utility": "#ED7D31",
        "dominant_rate::Nostalgia": "#70AD47", "dominant_rate::None": "#7F7F7F",
        "dominant_rate::tie": "#9E480E",
    }
    for col in ["dominant_rate::Flow","dominant_rate::Utility","dominant_rate::Nostalgia","dominant_rate::None","dominant_rate::tie"]:
        if col in order.columns:
            vals = order[col].fillna(0.0).values
            ax.bar(order["appid"].astype(str).values, vals, bottom=bottom.values, color=palette.get(col, "#999999"), label=col.split("::")[1])
            bottom = bottom + order[col].fillna(0.0)
    ax.set_title("Dominant Category Share by App (LLM)")
    ax.set_ylabel("Share"); ax.set_ylim(0, 1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order["appid"].astype(str).values, rotation=90, fontsize=7)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(charts_dir / f"dominant_share_by_app.{fmt}", format=fmt)
    plt.close(fig)

# ======================
# Summary
# ======================
def write_summary(summary_path: Path, n_sampled: int, n_prompts: int,
                  counters: Optional[Dict[str, int]],
                  dbg_paths: Optional[Dict[str, Path]],
                  per_app: Optional[pd.DataFrame],
                  n_unreadable: int) -> None:
    lines = []
    lines.append("# Step 10 — LLM Scoring Summary")
    lines.append(f"Sampled reviews      : {n_sampled}")
    lines.append(f"Prompts written      : {n_prompts}")
    lines.append(f"Excluded (unreadable): {n_unreadable}")
    if counters:
        lines.append(f"Responses parsed (lines) : {counters.get('lines_total', 0)}")
        lines.append(f"Valid responses           : {counters.get('valid', 0)}")
        lines.append(f"Invalid responses         : {counters.get('invalid', 0)}")
        lines.append(f"Renormalized (sum tol)    : {counters.get('renormalized', 0)}")
        lines.append(f"Dominance fixed           : {counters.get('dominance_fixed', 0)}")
        lines.append(f"Missing responses         : {counters.get('missing', 0)}")
        lines.append(f"Unmatched responses       : {counters.get('unmatched', 0)}")
    if per_app is not None and not per_app.empty:
        lines.append("")
        lines.append("App-level means (averaged across apps):")
        for c in ["Flow_llm_mean", "Utility_llm_mean", "Nostalgia_llm_mean", "None_llm_mean", "confidence_mean", "pct_fun_any"]:
            if c in per_app.columns:
                lines.append(f" {c:22s}= {per_app[c].mean():.4f}")
    if dbg_paths:
        lines.append("")
        lines.append("Debug files:")
        for k, p in dbg_paths.items():
            lines.append(f" {k:18s}: {p}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ======================
# CLI & main
# ======================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 10 — LLM scoring (prompts + offline ingestion).")
    ap.add_argument("--export-only", action="store_true", help="Only export prompts; skip ingestion.")
    ap.add_argument("--ingest-only", action="store_true", help="Only ingest responses; assume prompts already exported.")
    ap.add_argument("--responses", type=str, default=None, help="Override responses.jsonl path.")
    ap.add_argument("--export-limit", type=int, default=0, help="Write at most N prompts to requests/prompts.jsonl (0 = all).")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    samples_path = Path(str(CONFIG["SAMPLED_REVIEWS_CSV"]))
    pairs_path = Path(str(CONFIG["PAIRS_CSV"]))
    charts_dir = Path(str(CONFIG["CHARTS_DIR"]))
    debug_dir = Path(str(CONFIG["DEBUG_DIR"]))
    summary_path = Path(str(CONFIG["SUMMARY_TXT"]))
    prompts_path = Path(str(CONFIG["PROMPTS_JSONL"]))
    fast_snap = Path(str(CONFIG["PROMPT_FAST_SNAPSHOT"]))
    nuanced_snap = Path(str(CONFIG["PROMPT_NUANCED_SNAPSHOT"]))
    per_review_csv = Path(str(CONFIG["PER_REVIEW_CSV"]))
    per_app_csv = Path(str(CONFIG["PER_APP_CSV"]))
    per_pair_csv = Path(str(CONFIG["PER_PAIR_CSV"]))
    responses_path = Path(args.responses) if args.responses else Path(str(CONFIG["RESPONSES_DIR"])) / "responses.jsonl"

    samples = read_samples(samples_path)
    n_sampled = samples.shape[0]
    log("INFO", f"Loaded sampled reviews: {n_sampled} from {samples_path}")

    n_prompts = 0
    n_unreadable = 0
    if args.ingest_only:
        log("INFO", "--ingest-only: skipping prompt export.")
    else:
        if bool(CONFIG["GENERATE_PROMPTS"]):
            samples_use = samples
            if getattr(args, "export_limit", 0) and args.export_limit > 0:
                samples_use = samples.head(int(args.export_limit))
            n_prompts, n_unreadable = export_prompts(
                samples_use, prompts_path, fast_snap, nuanced_snap, Path(str(CONFIG["INPUTS_DIR"]))
            )
            log("OK", f"Prompts written: {n_prompts} -> {prompts_path}")
            log("OK", f"Prompt snapshots -> {fast_snap}, {nuanced_snap}")
        else:
            log("INFO", "GENERATE_PROMPTS=False; skipping prompt export.")

    counters = None
    dbg_paths = None
    per_review = None
    per_app = None

    if args.export_only:
        log("INFO", "--export-only: skipping ingestion.")
    else:
        if bool(CONFIG["VALIDATE_AND_INGEST"]):
            if not responses_path.exists():
                log("INFO", f"No responses at {responses_path}. Prompts ready; run 10b (Ollama).")
            else:
                log("INFO", f"Ingesting responses from {responses_path}")
                merged, counters, dbg_paths, passthrough = ingest_responses(
                    responses_path=responses_path,
                    samples=samples,
                    tie_band=float(CONFIG["TIE_BAND"]),
                    sum_tol=float(CONFIG["SUM_TOL"]),
                    debug_dir=debug_dir,
                )
                pairs = read_pairs(pairs_path)
                if pairs is not None:
                    merged = merged.merge(pairs[["appid","pairIndex","type"]], on="appid", how="left")
                per_review = write_per_review(merged, per_review_csv, passthrough)
                per_app = per_app_aggregate(per_review, pairs, per_app_csv)
                _ = per_pair_deltas(per_app, per_pair_csv)
                make_charts(per_review, per_app, charts_dir)
            log("OK", "Tables and charts written under data/analysis/10/")
        else:
            log("INFO", "VALIDATE_AND_INGEST=False; skipping ingestion.")

    write_summary(summary_path, n_sampled, n_prompts, counters, dbg_paths, per_app, n_unreadable)
    log("OK", f"Summary -> {summary_path}")
    log("OK", "Step 10 completed.")

if __name__ == "__main__":
    main()