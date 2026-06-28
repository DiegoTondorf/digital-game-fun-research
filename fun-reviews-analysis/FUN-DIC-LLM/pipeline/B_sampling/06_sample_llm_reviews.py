# 06_sample_llm_reviews.py
# coding: utf-8

'''
Step 06: LLM Review Sampling - Performs deterministic stratified sampling for LLM scoring.

• Samples reviews per app for LLM evaluation
• Produces per-app and aggregate sampled datasets
• Generates sampling summaries for transparency

Inputs: reviews_per_app/{appid}.csv (from Step 05)
Outputs: reviews_sampled/{appid}_sample.csv, reviews_sampled_all.csv, reviews_sampled/{appid}_sampling_summary.csv, reviews_sampled_summary_all.csv
'''


import math
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pipeline.Z_utils.common import asc, log, ensure_dir, find_project_root


ROOT = find_project_root()
DATA = ROOT / "data"

# ======================
# CONFIG
# ======================

# Pipeline Step 06: Sample LLM reviews
# All paths are relative to the repo root for reproducibility
# Helper functions are imported from utils/common.py

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

CONFIG = {
    "REVIEWS_PER_APP_DIR": str(get_env("SAMPLE_REVIEWS_PER_APP_DIR", DATA / "reviews_per_app/")),
    "SAMPLES_PER_APP_DIR": str(get_env("SAMPLE_SAMPLES_PER_APP_DIR", DATA / "reviews_sampled/")),
    "SAMPLES_MASTER_CSV": str(get_env("SAMPLE_SAMPLES_MASTER_CSV", DATA / "reviews_sampled_all.csv")),
    "SUMMARY_MASTER_CSV": str(get_env("SAMPLE_SUMMARY_MASTER_CSV", DATA / "reviews_sampled_summary_all.csv")),
    "MAX_SAMPLE_PER_APP": get_env("SAMPLE_MAX_SAMPLE_PER_APP", 150),
    "LENGTH_BUCKET_METHOD": get_env("SAMPLE_LENGTH_BUCKET_METHOD", "tertiles"),
    "VERBOSE": get_env("SAMPLE_VERBOSE", True),
}

# Columns preserved from Step 05 (minimum set; extra columns are carried through)
BASE_COLS = [
    "pairIndex","type","appid","name",
    "rankAcceptance","rankFiltered",
    "acceptanceScore","totalEnglishReviews",
    "reviewId","language","review",
    "isPositive","voted_up",
    "timestamp_created","timestamp_updated",
    "authorSteamId","playtime_forever","playtime_last_two_weeks",
    "received_for_free","written_during_early_access","steam_purchase",
    "comment_count","votes_up","votes_funny","weighted_vote_score",
    "charLen",
]

# New columns added by this step
NEW_COLS = [
    "lengthBucket",            # 'short' | 'medium' | 'long'
    "sampleIndexWithinApp",    # 1..m (deterministic order in final sample)
]

# Final sample columns
SAMPLE_COLS = BASE_COLS + NEW_COLS

# ASCII-safe logging

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ======================
# IO helpers
# ======================

def list_app_csvs(root: str) -> List[Path]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.suffix.lower()==".csv" and x.is_file()])

def read_app_reviews(app_csv: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(app_csv, encoding="utf-8-sig")
    # Type coercion / defaults (min set)
    df["isPositive"] = df["isPositive"].fillna(0).astype(int)
    df["timestamp_created"] = df["timestamp_created"].fillna(0).astype(int)
    df["charLen"] = df["charLen"].fillna(0).astype(int)
    return df.to_dict(orient="records")

def write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]):
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, columns=cols, index=False, encoding="utf-8")

def append_master(path: Path, rows: List[Dict[str, Any]], cols: List[str]):
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    header = (not path.exists()) or (path.stat().st_size == 0)
    df.to_csv(path, mode="a", header=header, columns=cols, index=False, encoding="utf-8")

# ======================
# Stats helpers
# ======================
def pctile_bounds(values: List[int]) -> Tuple[int, int]:
    """
    Return integer cut points (p33, p66) using nearest-rank on sorted values.
    For m >= 3, p33 = values[floor(0.333*(m-1))], p66 = values[floor(0.666*(m-1))].
    For small m, degrade gracefully.
    """
    m = len(values)
    if m == 0:
        return (0, 0)
    if m == 1:
        return (values[0], values[0])
    v = sorted(values)
    q1_idx = int(math.floor(0.333 * (m - 1)))
    q2_idx = int(math.floor(0.666 * (m - 1)))
    return (v[q1_idx], v[q2_idx])

def bucket_label(cl: int, p33: int, p66: int) -> str:
    if cl <= p33:
        return "short"
    elif cl <= p66:
        return "medium"
    else:
        return "long"

def sort_key_time_id(r: Dict[str, Any]):
    # Deterministic: oldest first, tie by reviewId
    return (int(r.get("timestamp_created", 0) or 0), str(r.get("reviewId","")))

def even_center_indices(n: int, k: int) -> List[int]:
    """
    Center-of-bin deterministic selection.
    For n items and k <= n picks:
       step = n / k
       idx[i] = floor((i + 0.5) * step)
    Guarantees spread across time (old/mid/new).
    """
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    step = n / float(k)
    idx = []
    for i in range(k):
        j = int(math.floor((i + 0.5) * step))
        if j >= n:  # clamp just in case of float edge
            j = n - 1
        idx.append(j)
    # ensure strictly increasing by nudging forward if any ties (rare)
    for t in range(1, len(idx)):
        if idx[t] <= idx[t-1]:
            idx[t] = min(idx[t-1] + 1, n - 1)
    # if nudging caused duplicates at the tail and we have room at the head, adjust backward
    for t in range(len(idx)-2, -1, -1):
        if idx[t] >= idx[t+1]:
            idx[t] = max(idx[t+1] - 1, 0)
    # final safety clamp unique+sorted
    idx = sorted(set(idx))
    # if we lost items due to set(), re-fill by scanning gaps (should be rare)
    while len(idx) < k:
        # insert missing around ideal centers
        candidates = list(range(n))
        for c in candidates:
            if c not in idx:
                idx.append(c)
                if len(idx) == k:
                    break
        idx = sorted(set(idx))[:k]
    return idx

def largest_remainder_alloc(totals: List[int], cap: int) -> List[int]:
    """
    Proportional allocation with largest remainder.
      inputs: totals for each stratum (length 6), cap target
      returns: integer targets summing to <= min(sum(totals), cap)
    """
    S = sum(totals)
    if S == 0 or cap <= 0:
        return [0]*len(totals)
    cap = min(cap, S)
    raw = [ (cap * t) / S for t in totals ]
    base = [ int(math.floor(x)) for x in raw ]
    rems = [ (raw[i] - base[i], i) for i in range(len(totals)) ]
    need = cap - sum(base)
    rems.sort(reverse=True)  # highest fractional parts first
    for r, i in rems:
        if need <= 0: break
        base[i] += 1
        need -= 1
    return base

# ======================
# Per-app sampling
# ======================
def sample_for_app(app_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (sampled_rows, summary_rows) for a single app.
    """
    if not app_rows:
        return [], []

    meta = {
        "appid": str(app_rows[0].get("appid","")).strip(),
        "name": str(app_rows[0].get("name","")).strip(),
        "pairIndex": int(float(app_rows[0].get("pairIndex", 0) or 0)),
        "type": str(app_rows[0].get("type","")).strip(),
        "rankAcceptance": int(float(app_rows[0].get("rankAcceptance", 0) or 0)),
        "rankFiltered": int(float(app_rows[0].get("rankFiltered", 0) or 0)),
        "acceptanceScore": float(app_rows[0].get("acceptanceScore", 0.0) or 0.0),
        "totalEnglishReviews": int(float(app_rows[0].get("totalEnglishReviews", 0) or 0)),
    }

    # If app has <= cap, take all (and still annotate buckets)
    cap = CONFIG["MAX_SAMPLE_PER_APP"]
    N = len(app_rows)
    cap = min(cap, N)

    # Split by sentiment
    pos = [r for r in app_rows if int(r.get("isPositive",0)) == 1]
    neg = [r for r in app_rows if int(r.get("isPositive",0)) == 0]

    # Compute length tertiles within each sentiment group
    def buckets(group_rows: List[Dict[str, Any]]):
        if not group_rows:
            return [], [], []
        lens = [int(r.get("charLen",0)) for r in group_rows]
        p33, p66 = pctile_bounds(lens)
        s, m, l = [], [], []
        for r in group_rows:
            lb = bucket_label(int(r.get("charLen",0)), p33, p66)
            if lb == "short":  s.append(r)
            elif lb == "medium": m.append(r)
            else: l.append(r)
        return s, m, l

    pos_s, pos_m, pos_l = buckets(pos)
    neg_s, neg_m, neg_l = buckets(neg)

    strata = [
        ("pos","short", pos_s),
        ("pos","medium",pos_m),
        ("pos","long",  pos_l),
        ("neg","short", neg_s),
        ("neg","medium",neg_m),
        ("neg","long",  neg_l),
    ]

    totals = [len(g) for _,_,g in strata]
    targets = largest_remainder_alloc(totals, cap)

    # Deterministic selection per stratum
    selected_by_stratum: List[List[Dict[str, Any]]] = []
    shortfalls = 0

    for (sent, lb, group), t in zip(strata, targets):
        grp_sorted = sorted(group, key=sort_key_time_id)
        n = len(grp_sorted)
        k = min(t, n)
        if k <= 0:
            selected_by_stratum.append([])
            continue
        idx = even_center_indices(n, k)
        chosen = [grp_sorted[i] for i in idx]
        # mark bucket for chosen
        for r in chosen:
            r["lengthBucket"] = lb
        selected_by_stratum.append(chosen)
        if k < t:
            shortfalls += (t - k)

    # If shortfalls (e.g., some strata too small), fill from remaining reviews deterministically
    selected_ids = set(r.get("reviewId","") for slab in selected_by_stratum for r in slab)
    remaining_pool = []
    for _,_,group in strata:
        # keep unsampled in time order
        grp_sorted = sorted(group, key=sort_key_time_id)
        for r in grp_sorted:
            if r.get("reviewId","") not in selected_ids:
                remaining_pool.append(r)
    if shortfalls > 0 and remaining_pool:
        # evenly spaced picks across remaining_pool
        remain_sorted = remaining_pool
        add_idx = even_center_indices(len(remain_sorted), min(shortfalls, len(remain_sorted)))
        # assign their lengthBucket as computed previously
        additions = [remain_sorted[i] for i in add_idx]
        for r in additions:
            if "lengthBucket" not in r:
                # if not already set (should be set), recompute light-weight
                # put into bucket by charLen against its sentiment group's tertiles
                # This is rare; strata already bucketed earlier.
                r["lengthBucket"] = "medium"
        # Merge additions into selected_by_stratum (append to their matching stratum list)
        # For traceability we just extend the first stratum; summaries still reflect final counts.
        selected_by_stratum[0].extend(additions)

    # Flatten and trim to cap (safety)
    selected = []
    for slab in selected_by_stratum:
        selected.extend(slab)
    # Deterministic final order: time then id
    selected = sorted(selected, key=sort_key_time_id)
    selected = selected[:cap]

    # Annotate sample index and ensure all required fields exist
    for i, r in enumerate(selected, start=1):
        r["sampleIndexWithinApp"] = i
        if "lengthBucket" not in r:
            r["lengthBucket"] = "medium"  # fallback

    # Build summary rows
    def tspan_days(rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        t1 = int(rows[0].get("timestamp_created",0))
        t2 = int(rows[-1].get("timestamp_created",0))
        return max(0, (t2 - t1) // 86400)

    # Counts per stratum before and after selection
    counts_before = { (sent,lb): len(group) for (sent,lb,group) in strata }
    counts_selected = {}
    for sent,lb,group in strata:
        # count selected items that belong to this (sent,lb)
        cnt = 0
        # We recognize sentiment by isPositive in row
        for r in selected:
            sflag = "pos" if int(r.get("isPositive",0))==1 else "neg"
            if sflag==sent and r.get("lengthBucket","") == lb:
                cnt += 1
        counts_selected[(sent,lb)] = cnt

    summary_rows = []
    for (sent, lb, group), t in zip(strata, targets):
        grp_sorted = sorted(group, key=sort_key_time_id)
        sel_cnt = counts_selected[(sent,lb)]
        row = {
            "appid": meta["appid"],
            "name": meta["name"],
            "pairIndex": meta["pairIndex"],
            "type": meta["type"],
            "sentiment": sent,
            "lengthBucket": lb,
            "n_total_stratum": len(group),
            "n_target_stratum": t,
            "n_selected_stratum": sel_cnt,
            "time_span_days_stratum": tspan_days(grp_sorted),
        }
        summary_rows.append(row)

    # App-level summary
    app_summary = {
        "appid": meta["appid"],
        "name": meta["name"],
        "pairIndex": meta["pairIndex"],
        "type": meta["type"],
        "n_reviews_app": N,
        "n_cap": CONFIG["MAX_SAMPLE_PER_APP"],
        "n_selected_app": len(selected),
        "n_pos": len([r for r in selected if int(r.get("isPositive",0))==1]),
        "n_neg": len([r for r in selected if int(r.get("isPositive",0))==0]),
        "totalEnglishReviews": meta["totalEnglishReviews"],
        "acceptanceScore": meta["acceptanceScore"],
    }
    # Add this as the first line in the per-app summary CSV
    summary_rows.insert(0, app_summary)

    # Console log
    log(f"[APP] {meta['appid']} `{asc(meta['name'])}` reviews={N} cap={CONFIG['MAX_SAMPLE_PER_APP']} selected={len(selected)}")
    for row in summary_rows[1:]:
        log(f"  - {row['sentiment']}/{row['lengthBucket']}: total={row['n_total_stratum']} "
            f"target={row['n_target_stratum']} selected={row['n_selected_stratum']} "
            f"span_days={row['time_span_days_stratum']}")

    return selected, summary_rows

# ======================
# Main
# ======================
def main():
    src_dir = Path(CONFIG["REVIEWS_PER_APP_DIR"])
    dst_dir = Path(CONFIG["SAMPLES_PER_APP_DIR"])
    master_samples = Path(CONFIG["SAMPLES_MASTER_CSV"])
    master_summary = Path(CONFIG["SUMMARY_MASTER_CSV"])

    ensure_dir(dst_dir)
    # Reset master files each run (deterministic)
    if master_samples.exists():
        master_samples.unlink()
    if master_summary.exists():
        master_summary.unlink()

    app_csvs = list_app_csvs(src_dir)
    if not app_csvs:
        raise FileNotFoundError(f"No per-app CSVs found in {src_dir}")

    # Master summary columns (wide enough for both strata and app-level rows)
    summary_cols = [
        "appid","name","pairIndex","type",
        "n_reviews_app","n_cap","n_selected_app","n_pos","n_neg",
        "totalEnglishReviews","acceptanceScore",
        "sentiment","lengthBucket",
        "n_total_stratum","n_target_stratum","n_selected_stratum","time_span_days_stratum",
    ]

    all_summary_rows: List[Dict[str, Any]] = []
    total_selected_count = 0

    for app_csv in app_csvs:
        rows = read_app_reviews(app_csv)
        if not rows:
            log(f"[SKIP] Empty or unreadable: {app_csv.name}")
            continue

        # Sample for this app
        sampled, summary = sample_for_app(rows)

        # Per-app sample CSV
        # Preserve all BASE_COLS and add NEW_COLS, but keep any extra columns if present.
        # We'll write exactly SAMPLE_COLS to keep it tidy.
        # Ensure columns exist in sampled rows
        for r in sampled:
            for c in SAMPLE_COLS:
                if c not in r:
                    r[c] = r.get(c, "")
        per_app_out = dst_dir / f"{rows[0].get('appid','')}_sample.csv"
        write_csv(per_app_out, sampled, SAMPLE_COLS)

        # Append to master samples
        append_master(master_samples, sampled, SAMPLE_COLS)

        # Per-app summary CSV
        per_app_summary = dst_dir / f"{rows[0].get('appid','')}_sampling_summary.csv"
        write_csv(per_app_summary, summary, summary_cols)

        # Append to master summary
        append_master(master_summary, summary, summary_cols)

        all_summary_rows.extend(summary)
        total_selected_count += len(sampled)

    log(f"[OK] Step 06 finished. Sampled total reviews: {total_selected_count}")
    log(f"     Master sample: {master_samples}")
    log(f"     Master summary: {master_summary}")

if __name__ == "__main__":
    main()
