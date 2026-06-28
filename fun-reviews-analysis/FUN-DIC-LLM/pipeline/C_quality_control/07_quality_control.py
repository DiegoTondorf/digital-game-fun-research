# 07_quality_control.py
# coding: utf-8

'''
Step 07: Quality Control - Performs quality control checks across pipeline steps 02–06.

• Aggregates and validates data from previous steps
• Generates summary reports, issues, and optional charts
• Supports reproducible paths and helper utilities

Inputs: Outputs from Steps 02–06 (various CSVs, data files).
Outputs: qc_summary_per_app.csv, qc_issues.csv, qc_report.txt, qc charts (optional).
'''


import json
import math
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pipeline.Z_utils.common import asc, log, ensure_dir, find_project_root


ROOT = find_project_root()
DATA = ROOT / "data"

# Optional charts (matplotlib only)
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

### Pipeline Step 07: Quality Control
# All paths are relative to the repo root for reproducibility
# Helper functions are imported from utils/common.py
###

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
    "STEP2_SORTED": str(get_env("QC_STEP2_SORTED", DATA / "steamspy_scored_sorted.csv")),
    "STEP3_FILTERED": str(get_env("QC_STEP3_FILTERED", DATA / "steamspy_filtered.csv")),
    "STEP4_PAIRS": str(get_env("QC_STEP4_PAIRS", DATA / "steamspy_top_bottom_pairs.csv")),
    "REVIEWS_PER_APP_DIR": str(get_env("QC_REVIEWS_PER_APP_DIR", DATA / "reviews_per_app/")),
    "REVIEWS_ALL": str(get_env("QC_REVIEWS_ALL", DATA / "reviews_all.csv")),
    "RAW_PER_APP_DIR": str(get_env("QC_RAW_PER_APP_DIR", DATA / "raw/reviews/")),
    "CKPT_DIR": str(get_env("QC_CKPT_DIR", DATA / "checkpoints/reviews/")),
    "SAMPLES_PER_APP_DIR": str(get_env("QC_SAMPLES_PER_APP_DIR", DATA / "reviews_sampled/")),
    "SAMPLES_MASTER": str(get_env("QC_SAMPLES_MASTER", DATA / "reviews_sampled_all.csv")),
    "SAMPLES_SUMMARY_MASTER": str(get_env("QC_SAMPLES_SUMMARY_MASTER", DATA / "reviews_sampled_summary_all.csv")),

    "QC_DIR": str(get_env("QC_QC_DIR", DATA / "qc/")),
    "QC_SUMMARY_PER_APP": str(get_env("QC_QC_SUMMARY_PER_APP", DATA / "qc/qc_summary_per_app.csv")),
    "QC_ISSUES": str(get_env("QC_QC_ISSUES", DATA / "qc/qc_issues.csv")),
    "QC_REPORT": str(get_env("QC_QC_REPORT", DATA / "qc/qc_report.txt")),

    "GENERATE_CHARTS": get_env("QC_GENERATE_CHARTS", True),
    "GLOBAL_CHARTS": get_env("QC_GLOBAL_CHARTS", True),
    "PER_APP_CHARTS": get_env("QC_PER_APP_CHARTS", True),
    "CHARTS_DIR": get_env("QC_CHARTS_DIR", DATA / "qc/charts/"),
    "MAX_PER_APP_CHARTS": get_env("QC_MAX_PER_APP_CHARTS", 50),

    "ROUND_DECIMALS": get_env("QC_ROUND_DECIMALS", 6),
    "MIN_TOTAL_REVIEWS": get_env("QC_MIN_TOTAL_REVIEWS", 50),
    "MIN_ENGLISH_REVIEWS": get_env("QC_MIN_ENGLISH_REVIEWS", 50),
    "FLAG_SAMPLE_POS_RATIO_DIFF_PCTPTS": get_env("QC_FLAG_SAMPLE_POS_RATIO_DIFF_PCTPTS", 5.0),
    "FLAG_SAMPLE_TIME_COVERAGE_MIN_FRAC": get_env("QC_FLAG_SAMPLE_TIME_COVERAGE_MIN_FRAC", 0.60),
    "FLAG_SAMPLE_LENGTH_DIST_DIFF_PCTPTS": get_env("QC_FLAG_SAMPLE_LENGTH_DIST_DIFF_PCTPTS", 10.0),

    "JUMP_LABELS": set(get_env("QC_JUMP_LABELS", "free to play,early access").split(",")),
    "NON_GAME_LABELS": set(get_env("QC_NON_GAME_LABELS",
        "utilities,software,web publishing,design and illustration,design & illustration,audio production,video production,photo editing,education,animation and modeling,animation & modeling,typing"
    ).split(",")),

    "VERBOSE": get_env("QC_VERBOSE", True),
}

# ======================
# ASCII-safe logging
# ======================

# ======================
# IO helpers
# ======================

def read_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df.to_dict(orient="records")
    except Exception:
        add_issue("ERROR","IO","",f"Cannot read CSV: {path}")
        return []

def write_rows(path: Path, rows: List[Dict[str, Any]], cols: List[str]):
    ensure_dir(path.parent)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=cols)
    df.to_csv(path, columns=cols, index=False, encoding="utf-8")

def list_csvs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower()==".csv"])

# ======================
# Small utils
# ======================
def to_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def to_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def canon(text: str) -> str:
    s = (text or "").lower().strip()
    for repl in ["&", "/", "\\", "+", "-", "_", ",", ";", "|"]:
        s = s.replace(repl, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s

def recompute_acc_from_counts(row: Dict[str, Any], dec: int) -> float:
    pos = to_int(row.get("posReview", 0))
    neg = to_int(row.get("negReview", 0))
    tot = pos + neg
    if tot <= 0:
        return 0.0
    pos_rate = pos / tot
    val = pos_rate * math.log(tot)
    return round(val, dec)

def q_indices(n: int) -> Tuple[int,int,int]:
    q1 = int(n*0.25); q2 = int(n*0.50); q3 = int(n*0.75)
    return q1, q2, q3

def pct(x: float) -> float:
    return 100.0 * x

# ======================
# QC accumulators
# ======================
ISSUES: List[Dict[str, Any]] = []
SUMMARY_PER_APP: List[Dict[str, Any]] = []

def add_issue(level: str, where: str, appid: str, detail: str):
    ISSUES.append({"level": level, "where": where, "appid": appid, "detail": detail})
    prefix = {"INFO":"[i]","WARN":"[!]","ERROR":"[x]"}.get(level,"[?]")
    log(f"{prefix} {where} app={appid or '-'}: {detail}")

# ======================
# Charts
# ======================
def save_hist(data: List[float], title: str, xlabel: str, out_path: Path, bins: int = 50):
    if not (CONFIG["GENERATE_CHARTS"] and HAS_MPL and data):
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=bins, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_stacked_bar(labels: List[str], pos_counts: List[int], neg_counts: List[int], title: str, out_path: Path):
    if not (CONFIG["GENERATE_CHARTS"] and HAS_MPL):
        return
    ensure_dir(out_path.parent)
    x = range(len(labels))
    plt.figure(figsize=(8,5))
    plt.bar(x, pos_counts, label="Positive", color="#59A14F")
    plt.bar(x, neg_counts, bottom=pos_counts, label="Negative", color="#E15759")
    plt.xticks(x, labels, rotation=0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ======================
# STEP 02 checks (+ chart)
# ======================
def check_step02():
    p = Path(CONFIG["STEP2_SORTED"])
    if not p.exists():
        add_issue("WARN", "Step02", "", f"Missing {p}")
        return [], 0
    rows = read_csv(p)
    n = len(rows)
    dec = CONFIG["ROUND_DECIMALS"]

    # sorted by acceptanceScore DESC
    last = float("inf")
    acc_vals = []
    diffs = 0
    for i, r in enumerate(rows):
        acc_file = round(to_float(r.get("acceptanceScore", 0.0)), dec)
        if acc_file > last + 10**(-dec):  # not non-increasing
            add_issue("ERROR","Step02","",f"Not sorted by acceptanceScore DESC at row {i}")
            break
        last = acc_file
        acc_vals.append(acc_file)

        # recompute acceptance from counts with same rounding policy
        acc_calc = recompute_acc_from_counts(r, dec)
        if acc_calc != acc_file:
            diffs += 1

    if diffs > 0:
        add_issue("INFO","Step02","",f"{diffs} rows differ after round({dec}) (likely roundoff).")

    log(f"[Step02] rows={n}")
    # chart
    if CONFIG["GLOBAL_CHARTS"]:
        save_hist(acc_vals, "Step 02: acceptanceScore distribution", "acceptanceScore",
                  Path(CONFIG["CHARTS_DIR"]) / "step02_acceptance_hist.png", bins=80)
    return rows, n

# ======================
# STEP 03 checks (+ chart + reconstructed slice)
# ======================
def check_step03(step02_rows: List[Dict[str, Any]]):
    p = Path(CONFIG["STEP3_FILTERED"])
    if not p.exists():
        add_issue("WARN","Step03","",f"Missing {p}")
        return [], 0
    rows = read_csv(p)
    n = len(rows)
    dec = CONFIG["ROUND_DECIMALS"]

    # sorted DESC
    last = float("inf")
    acc_vals = []
    for i, r in enumerate(rows):
        acc_file = round(to_float(r.get("acceptanceScore", 0.0)), dec)
        if acc_file > last + 10**(-dec):
            add_issue("ERROR","Step03","",f"Filtered not sorted DESC at row {i}")
            break
        last = acc_file
        acc_vals.append(acc_file)

    # dedup by appid
    seen = set(); dups=0
    for r in rows:
        appid = str(r.get("appid","")).strip()
        if appid in seen:
            dups += 1
        seen.add(appid)
    if dups>0:
        add_issue("ERROR","Step03","",f"Found {dups} duplicate appids in filtered file")


    # Reconstruct Step-03 pre-quartile population from Step-02 using the exact same sequence and pandas logic as Step 03
    if step02_rows:
        import pandas as pd
        df = pd.DataFrame(step02_rows)
        # 1) Sort by acceptanceScore DESC
        df = df.sort_values(by="acceptanceScore", ascending=False)
        # 2) Apply minimum total reviews
        nmin = CONFIG["MIN_TOTAL_REVIEWS"]
        df = df[df["totReview"] >= nmin]
        # 3) De-duplicate by appid (keep first = highest acceptanceScore)
        df = df.drop_duplicates(subset=["appid"], keep="first")
        # 4) Quartile slicing (middle 50%)
        n2 = len(df)
        if n2 >= 4:
            q1 = int(n2 * 0.25)
            q3 = int(n2 * 0.75)
            expected = set(df.iloc[q1:q3]["appid"].astype(str).str.strip())
            got = {str(r.get("appid", "")).strip() for r in rows}
            missing = expected - got
            extra = got - expected
            if missing:
                add_issue("WARN", "Step03", "", f"{len(missing)} expected mid-slice appids missing from Step03 output (after reconstruction)")
            if extra:
                add_issue("WARN", "Step03", "", f"{len(extra)} Step03 appids are extra beyond reconstructed mid-slice (check filters/dedup).")

    log(f"[Step03] rows={n}")
    # chart
    if CONFIG["GLOBAL_CHARTS"]:
        save_hist(acc_vals, "Step 03: acceptanceScore (filtered middle 50%)", "acceptanceScore",
                  Path(CONFIG["CHARTS_DIR"]) / "step03_acceptance_hist.png", bins=80)
    return rows, n

# ======================
# STEP 04 checks (first-genre rules)
# ======================
def check_step04():
    p = Path(CONFIG["STEP4_PAIRS"])
    if not p.exists():
        add_issue("WARN","Step04","",f"Missing {p}")
        return []
    rows = read_csv(p)
    used = set()
    jump = set(CONFIG["JUMP_LABELS"])
    nongen = set(CONFIG["NON_GAME_LABELS"])
    for r in rows:
        appid = str(r.get("appid","")).strip()
        fg = canon(r.get("firstGenre",""))
        if not fg:
            add_issue("WARN","Step04", appid, "Missing firstGenre")
            continue
        if fg in jump or fg in nongen:
            add_issue("ERROR","Step04", appid, f"firstGenre '{fg}' is in jump/non-game sets")
        if fg in used:
            add_issue("ERROR","Step04", appid, f"firstGenre '{fg}' repeated across selections")
        used.add(fg)
    log(f"[Step04] pairs rows={len(rows)} firstGenre_unique={len(used)}")
    return rows

# ======================
# STEP 05 checks (per-app reviews + cross-checks vs raw & ckpt + global chart)
# ======================
def per_app_review_checks():
    root = Path(CONFIG["REVIEWS_PER_APP_DIR"])
    app_index: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        add_issue("WARN","Step05","",f"Missing dir {root}")
        return app_index

    total_all = 0
    pos_all = 0

    for path in list_csvs(root):
        rows = read_csv(path)
        if not rows:
            continue
        appid = str(rows[0].get("appid","")).strip()
        name = str(rows[0].get("name","")).strip()
        total_rows = len(rows)
        total_all += total_rows

        # positives
        pos = sum(1 for r in rows if to_int(r.get("isPositive",0))==1)
        pos_all += pos

        # totalEnglishReviews consistency
        en_vals = { to_int(r.get("totalEnglishReviews",0)) for r in rows }
        if len(en_vals) > 1:
            add_issue("WARN","Step05",appid, f"totalEnglishReviews not constant: {sorted(en_vals)}")
        total_en = max(en_vals) if en_vals else 0
        if total_en < CONFIG["MIN_ENGLISH_REVIEWS"]:
            add_issue("WARN","Step05",appid,f"totalEnglishReviews={total_en} below MIN_ENGLISH_REVIEWS={CONFIG['MIN_ENGLISH_REVIEWS']}")

        # duplicates in reviewId
        seen = set(); dups=0
        for r in rows:
            rid = str(r.get("reviewId","")).strip()
            if rid in seen: dups+=1
            seen.add(rid)
        if dups>0:
            add_issue("ERROR","Step05",appid, f"{dups} duplicate reviewId found")

        # cross-check raw pages vs ckpt, allow 1 extra empty raw file at end
        raw_dir = Path(CONFIG["RAW_PER_APP_DIR"]) / appid
        raw_files = sorted(raw_dir.glob("page_*.json")) if raw_dir.exists() else []
        raw_pages = len(raw_files)
        ckpt_file = Path(CONFIG["CKPT_DIR"]) / f"{appid}.ckpt.json"
        if ckpt_file.exists():
            try:
                j = json.loads(ckpt_file.read_text(encoding="utf-8"))
                ckpt_pages = int(j.get("pages",0))
                # Allow one extra raw file if it is empty (no reviews)
                if raw_pages == ckpt_pages + 1 and raw_files:
                    with open(raw_files[-1], encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            if not data.get("reviews"):
                                pass  # OK: last raw file is empty
                            else:
                                add_issue("WARN","Step05",appid, f"raw pages ({raw_pages}) != ckpt pages ({ckpt_pages})")
                        except Exception:
                            add_issue("WARN","Step05",appid, f"raw pages ({raw_pages}) != ckpt pages ({ckpt_pages}) and last file unreadable")
                elif ckpt_pages != raw_pages:
                    add_issue("WARN","Step05",appid, f"raw pages ({raw_pages}) != ckpt pages ({ckpt_pages})")
            except Exception:
                add_issue("WARN","Step05",appid, "cannot read checkpoint json")
        else:
            add_issue("INFO","Step05",appid, "no checkpoint file (ok if intentional)")

        # lengths
        lens = [to_int(r.get("charLen",0)) for r in rows]
        lens_sorted = sorted(lens)

        app_index[appid] = {
            "appid": appid, "name": name,
            "per_app_count": total_rows,
            "totalEnglishReviews": total_en,
            "positive_ratio_english": (pos/total_rows) if total_rows>0 else 0.0,
            "charLens": lens_sorted,
        }

    # Compare master CSV count vs sum of per-app counts
    master = Path(CONFIG["REVIEWS_ALL"])
    if master.exists():
        master_rows = read_csv(master)
        if len(master_rows) != total_all:
            add_issue("WARN","Step05","",f"reviews_all.csv count {len(master_rows)} != sum of per-app {total_all}")

    # global chart: overall review length histogram
    if CONFIG["GLOBAL_CHARTS"]:
        all_lens = []
        for a in app_index.values():
            all_lens.extend(a["charLens"])
        save_hist(all_lens, "Step 05: Review length (chars) across all apps", "charLen",
                  Path(CONFIG["CHARTS_DIR"]) / "step05_lengths_all.png", bins=80)

    log(f"[Step05] per-app CSVs={len(app_index)} total_reviews={total_all}")
    return app_index

# ======================
# STEP 06 checks (sampling) + per-app charts + flags
# ======================
def check_step06(app_index: Dict[str, Any]):
    root = Path(CONFIG["SAMPLES_PER_APP_DIR"])
    if not root.exists():
        add_issue("WARN","Step06","",f"Missing dir {root}")
        return
    master_sample = Path(CONFIG["SAMPLES_MASTER"])
    if not master_sample.exists():
        add_issue("WARN","Step06","",f"Missing master sample {master_sample}")
    master_summary = Path(CONFIG["SAMPLES_SUMMARY_MASTER"])
    if not master_summary.exists():
        add_issue("WARN","Step06","",f"Missing master summary {master_summary}")

    # Limit per-app charting
    charted = 0
    max_charts = CONFIG["MAX_PER_APP_CHARTS"]

    for sfile in list_csvs(root):
        if not sfile.name.endswith("_sample.csv"):
            continue
        rows = read_csv(sfile)
        if not rows:
            continue

        appid = str(rows[0].get("appid","")).strip()
        name = str(rows[0].get("name","")).strip()
        n_sel = len(rows)
        n_pos = sum(1 for r in rows if to_int(r.get("isPositive",0))==1)
        n_neg = n_sel - n_pos
        pos_ratio_sample = (n_pos / n_sel) if n_sel>0 else 0.0

        # Per-app full info from Step05
        base = app_index.get(appid, {})
        n_full = base.get("per_app_count", 0)
        pos_ratio_full = base.get("positive_ratio_english", 0.0)

        # Flag: positivity deviation
        diff_pp = abs(pct(pos_ratio_sample) - pct(pos_ratio_full))
        if n_full > 0 and diff_pp > CONFIG["FLAG_SAMPLE_POS_RATIO_DIFF_PCTPTS"]:
            add_issue("WARN","Step06",appid,
                      f"sample positivity {pct(pos_ratio_sample):.2f}% vs fetched {pct(pos_ratio_full):.2f}% (diff {diff_pp:.2f} pp)")

        # Time coverage flag
        full_csv = Path(CONFIG["REVIEWS_PER_APP_DIR"]) / f"{appid}.csv"
        t_all = sorted([to_int(r.get("timestamp_created",0)) for r in read_csv(full_csv)]) if full_csv.exists() else []
        t_sample = sorted([to_int(r.get("timestamp_created",0)) for r in rows])
        span_all = (t_all[-1]-t_all[0]) if len(t_all)>=2 else 0
        span_sam = (t_sample[-1]-t_sample[0]) if len(t_sample)>=2 else 0
        frac = (span_sam/span_all) if span_all>0 else 1.0
        if span_all>0 and frac < CONFIG["FLAG_SAMPLE_TIME_COVERAGE_MIN_FRAC"]:
            add_issue("WARN","Step06",appid,
                      f"sample time span covers {frac*100:.1f}% of full timespan (< {CONFIG['FLAG_SAMPLE_TIME_COVERAGE_MIN_FRAC']*100:.0f}%)")

        # Length-bucket share check: sample vs full (approx via tertiles on full lengths)
        full_lens = base.get("charLens", [])
        if full_lens:
            m = len(full_lens)
            p33 = full_lens[int(math.floor(0.333*(m-1)))]
            p66 = full_lens[int(math.floor(0.666*(m-1)))]
            def bucket_len(x): return "short" if x<=p33 else ("medium" if x<=p66 else "long")

            full_bk = {"short":0,"medium":0,"long":0}
            for L in full_lens:
                full_bk[bucket_len(L)] += 1
            full_sh = {k:(v/m) for k,v in full_bk.items()}

            sam_bk = {"short":0,"medium":0,"long":0}
            for r in rows:
                lb = r.get("lengthBucket","").strip() or bucket_len(to_int(r.get("charLen",0)))
                if lb not in sam_bk: lb = bucket_len(to_int(r.get("charLen",0)))
                sam_bk[lb] = sam_bk.get(lb,0)+1
            sam_sh = {k:(sam_bk[k]/n_sel if n_sel>0 else 0.0) for k in ("short","medium","long")}
            for k in ("short","medium","long"):
                diff_len_pp = abs(pct(sam_sh[k]) - pct(full_sh.get(k,0.0)))
                if diff_len_pp > CONFIG["FLAG_SAMPLE_LENGTH_DIST_DIFF_PCTPTS"]:
                    add_issue("WARN","Step06",appid,
                              f"sample {k} share {pct(sam_sh[k]):.1f}% vs full {pct(full_sh.get(k,0.0)):.1f}% (diff {diff_len_pp:.1f} pp)")

        # Per-app charts (optional)
        if CONFIG["PER_APP_CHARTS"] and charted < max_charts:
            # stacked sentiment by lengthBucket (sample)
            labels = ["short","medium","long"]
            pos_counts = [sum(1 for r in rows if (r.get("lengthBucket","")==lab or lab in r.get("lengthBucket","")) and to_int(r.get("isPositive",0))==1) for lab in labels]
            neg_counts = [sum(1 for r in rows if (r.get("lengthBucket","")==lab or lab in r.get("lengthBucket","")) and to_int(r.get("isPositive",0))==0) for lab in labels]
            out = Path(CONFIG["CHARTS_DIR"]) / f"app_{appid}_sample_len_sentiment.png"
            save_stacked_bar(labels, pos_counts, neg_counts,
                             f"App {appid} sample: sentiment by length", out)
            # per-app review length histogram (full)
            out2 = Path(CONFIG["CHARTS_DIR"]) / f"app_{appid}_full_lengths.png"
            save_hist(full_lens, f"App {appid} full: review length (chars)", "charLen", out2, bins=50)
            charted += 1

        # Update app_index with sample info for final summary CSV
        if appid not in app_index:
            app_index[appid] = {"appid": appid, "name": name}
        app_index[appid].update({
            "sample_count": n_sel,
            "sample_pos": n_pos,
            "sample_neg": n_neg,
            "sample_time_span_days": (span_sam//86400) if span_sam>0 else 0
        })

# ======================
# MAIN
# ======================
def main():
    ensure_dir(Path(CONFIG["QC_DIR"]))
    if CONFIG["GENERATE_CHARTS"]:
        ensure_dir(Path(CONFIG["CHARTS_DIR"]))

    # Step 02
    s2_rows, s2_n = check_step02()
    # Step 03
    s3_rows, s3_n = check_step03(s2_rows)
    # Step 04
    pairs_rows = check_step04()
    # Step 05
    per_app = per_app_review_checks()
    # Step 06
    check_step06(per_app)

    # Merge pair context for summary
    pair_map: Dict[str, Dict[str, Any]] = {}
    for r in pairs_rows:
        a = str(r.get("appid","")).strip()
        if a and a not in pair_map:
            pair_map[a] = {"pairIndex": to_int(r.get("pairIndex",0)), "type": r.get("type","")}

    # Build per-app summary rows
    for appid, rec in per_app.items():
        srow = {
            "appid": appid,
            "name": rec.get("name",""),
            "pairIndex": pair_map.get(appid,{}).get("pairIndex",""),
            "type": pair_map.get(appid,{}).get("type",""),
            "totalEnglishReviews": rec.get("totalEnglishReviews",0),
            "fetched_reviews": rec.get("per_app_count",0),
            "english_positive_ratio": rec.get("positive_ratio_english",0.0),
            "sampled_reviews": rec.get("sample_count",""),
            "sample_pos": rec.get("sample_pos",""),
            "sample_neg": rec.get("sample_neg",""),
            "sample_time_span_days": rec.get("sample_time_span_days",""),
        }
        SUMMARY_PER_APP.append(srow)

    # Write outputs
    summary_cols = [
        "appid","name","pairIndex","type",
        "totalEnglishReviews","fetched_reviews","english_positive_ratio",
        "sampled_reviews","sample_pos","sample_neg","sample_time_span_days",
    ]
    write_rows(Path(CONFIG["QC_SUMMARY_PER_APP"]), SUMMARY_PER_APP, summary_cols)

    issues_cols = ["level","where","appid","detail"]
    write_rows(Path(CONFIG["QC_ISSUES"]), ISSUES, issues_cols)

    # Human-readable report
    errors = sum(1 for x in ISSUES if x["level"]=="ERROR")
    warns  = sum(1 for x in ISSUES if x["level"]=="WARN")
    infos  = sum(1 for x in ISSUES if x["level"]=="INFO")
    with open(Path(CONFIG["QC_REPORT"]),"w",encoding="utf-8") as f:
        f.write("=== QC REPORT ===\n")
        f.write(f"Step02 rows: {s2_n}\n")
        f.write(f"Step03 rows: {s3_n}\n")
        f.write(f"Pairs rows (Step04): {len(pairs_rows)}\n")
        f.write(f"Apps with reviews (Step05): {len(SUMMARY_PER_APP)}\n")
        f.write(f"Issues: ERROR={errors}, WARN={warns}, INFO={infos}\n\n")
        if ISSUES:
            f.write("Top issues:\n")
            for x in ISSUES[:500]:
                f.write(f" - {x['level']} {x['where']} app={x['appid']}: {x['detail']}\n")
        f.write("\nCharts directory: {}\n".format(CONFIG["CHARTS_DIR"] if CONFIG["GENERATE_CHARTS"] else "(charts disabled)"))

    log(f"[OK] QC complete. Summary: {CONFIG['QC_SUMMARY_PER_APP']}, Issues: {CONFIG['QC_ISSUES']}, Report: {CONFIG['QC_REPORT']}")

if __name__ == "__main__":
    main()