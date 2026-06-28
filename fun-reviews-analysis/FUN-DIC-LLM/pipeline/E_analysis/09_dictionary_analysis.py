#!/usr/bin/env python3

'''
Step 09: Dictionary Analysis - Analyzes review texts using FUN lexicon and computes metrics.

• Tokenizes and normalizes review texts
• Counts FUN lexicon hits (Flow, Utility, Nostalgia)
• Computes metrics at review and app level
• Outputs detailed metrics and summary statistics

Inputs: Reviews CSV, FUN dictionary CSV, configuration files.
Outputs: Metrics tables, summary statistics, logs.
'''


import csv
import sys
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

# Add project root to sys.path to allow absolute imports from 'pipeline'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.Z_utils.common import find_project_root, log as masked_log, ensure_dir
ROOT = find_project_root()
DATA = ROOT / "data"
INPUT = ROOT / "analysis_inputs"
import pandas as pd
import os

### Pipeline Step 09: Dictionary analysis
# All paths are relative to the repo root for reproducibility
# Helper functions are imported from utils/common.py
###
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
    # Inputs
    "REVIEWS_PER_APP_DIR": str(get_env("STEP09_REVIEWS_PER_APP_DIR", DATA / "reviews_per_app/")),
    "PAIRS_CSV": str(get_env("STEP09_PAIRS_CSV", DATA / "steamspy_top_bottom_pairs.csv")),
    "LEXICONS_DIR": str(get_env("STEP09_LEXICONS_DIR", INPUT / "dictionary/")),

    # Outputs
    "OUT_ROOT": str(get_env("STEP09_OUT_ROOT", DATA / "analysis/09/")),
    "TOKENS_PER_APP_DIR": str(get_env("STEP09_TOKENS_PER_APP_DIR", DATA / "analysis/09/tokens_per_app/")),
    "TOKENS_OVERALL_CSV": str(get_env("STEP09_TOKENS_OVERALL_CSV", DATA / "analysis/09/tokens_overall.csv")),
    "PER_APP_CATEGORY_CSV": str(get_env("STEP09_PER_APP_CATEGORY_CSV", DATA / "analysis/09/per_app_categories.csv")),
    "PER_PAIR_CATEGORY_CSV": str(get_env("STEP09_PER_PAIR_CATEGORY_CSV", DATA / "analysis/09/per_pair_deltas_categories.csv")),
    "PER_APP_SCORE_CSV": str(get_env("STEP09_PER_APP_SCORE_CSV", DATA / "analysis/09/per_app_scores.csv")),
    "PER_PAIR_SCORE_CSV": str(get_env("STEP09_PER_PAIR_SCORE_CSV", DATA / "analysis/09/per_pair_deltas_scores.csv")),
    "PER_REVIEW_FUN_CSV": str(get_env("STEP09_PER_REVIEW_FUN_CSV", DATA / "analysis/09/per_review_fun.csv")),
    "PER_APP_REVIEW_AGG_CSV": str(get_env("STEP09_PER_APP_REVIEW_AGG_CSV", DATA / "analysis/09/per_app_review_agg.csv")),
    "SUMMARY_TXT": str(get_env("STEP09_SUMMARY_TXT", DATA / "analysis/09/summary.txt")),

    # Tokenization settings (minimal normalization)
    "LOWERCASE": get_env("STEP09_LOWERCASE", True),
    "MIN_TOKEN_LEN": get_env("STEP09_MIN_TOKEN_LEN", 2),
    "KEEP_NUMERIC": get_env("STEP09_KEEP_NUMERIC", True),
    "APPLY_LIGHT_STEMMER": get_env("STEP09_APPLY_LIGHT_STEMMER", False),

    # Frequency normalization
    "PER_K_TOKENS": get_env("STEP09_PER_K_TOKENS", 1000.0),
    "PER_100_TOKENS": get_env("STEP09_PER_100_TOKENS", 100.0),
    # Output controls
    # When True, per-review and per-app aggregates will include per-100 token intensities
    # (e.g., Flow_per_100). When True, include normalized 4-d shares (Flow_share, Utility_share,
    # Nostalgia_share, None_share) that sum to 1.0 per review. Both can be enabled.
    "OUTPUT_PER100": get_env("STEP09_OUTPUT_PER100", True),
    "OUTPUT_SHARES": get_env("STEP09_OUTPUT_SHARES", True),

    # FUN canonical order
    "FUN_ORDER": ["Flow", "Utility", "Nostalgia"],

    # Logging
    "VERBOSE": get_env("STEP09_VERBOSE", True),
}

# Centralized log function
def log(level: str, msg: str):
    masked_log(f"[{level}] {msg}", verbose=CONFIG.get("VERBOSE", True))

# ----------------------- IO helpers --------------------

def ensure_dirs():
    for p in [CONFIG["OUT_ROOT"], CONFIG["TOKENS_PER_APP_DIR"]]:
        ensure_dir(p)

def list_review_csvs(root: str) -> List[Path]:
    d = Path(root)
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower()==".csv" and p.is_file()])

def read_pairs(path: str) -> Optional["pd.DataFrame"]:
    """
    Read Step 04 pairs and normalize dtypes so downstream merges won't fail.
    - appid -> str
    - pairIndex -> numeric
    - type -> str
    """
    if not Path(path).exists():
        log("INFO", f"Pairs CSV not found: {path} (pairIndex/type columns will be absent)")
        return None
    try:
        df = pd.read_csv(path)
        if "appid" in df.columns:
            df["appid"] = df["appid"].astype(str)
        if "pairIndex" in df.columns:
            df["pairIndex"] = pd.to_numeric(df["pairIndex"], errors="coerce")
        if "type" in df.columns:
            df["type"] = df["type"].astype(str)
        return df
    except Exception as e:
        log("WARN", f"Failed to read pairs CSV: {path} err={e}")
        return None

# ----------------------- Tokenization ------------------
TOKEN_RE = re.compile(r"[^a-z0-9]+")  # keep only a-z0-9, split others
_SUFFIXES = ("ing", "ed", "ly", "ies", "s")

def light_stem(token: str) -> str:
    # ultra-light, optional; disabled by default
    t = token
    for suf in _SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[:-len(suf)]
    return t

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = text
    if CONFIG["LOWERCASE"]:
        t = t.lower()
    t = TOKEN_RE.sub(" ", t)  # replace non a-z0-9 with spaces
    toks = [tok for tok in t.split() if len(tok) >= CONFIG["MIN_TOKEN_LEN"]]
    if not CONFIG["KEEP_NUMERIC"]:
        toks = [tok for tok in toks if not tok.isdigit()]
    if CONFIG["APPLY_LIGHT_STEMMER"]:
        toks = [light_stem(tok) for tok in toks]
    return toks

# ----------------------- Share utilities ----------------------
def _renormalize_shares(flow: float, util: float, nost: float, none_val: float) -> Tuple[float, float, float, float]:
        """Clamp to [0,1] then renormalize to sum to 1.0.

        Notes
        -----
    - Shares are initially computed as counts per token (count/total_tokens) for Flow/Utility/Nostalgia
      and soft None = 1 - sum(F,U,N). Because lexicon hits may overlap across categories, the sum can
      exceed 1.0 or produce a negative None. We clamp negatives to 0 and then re-scale the 4 values to
      sum exactly to 1.0. If all are zero (e.g., empty review), defaults to None=1.0.
        """
        vals = [max(0.0, min(1.0, float(x))) for x in (flow, util, nost, none_val)]
        s = sum(vals)
        if s <= 0:
                return 0.0, 0.0, 0.0, 1.0
        return tuple(v / s for v in vals)

# ----------------------- Lexicons ----------------------
def load_lexicons(dir_path: str):
    """
    Return two dicts:
      - score_lex: Dict[token] -> float
      - category_lex: Dict[token] -> Set[categories]
    Auto-detect schema by columns in each CSV. Skip invalid rows with WARN.
        Assumptions
        -----------
        - Lexicon sparsity: Many reviews will have zero hits; Step 09 reports pct_with_fun and zero_hit_pct.
        - Tokens are matched after light normalization (lowercase and punctuation stripping). Optional light
            stemming is available but disabled by default for reproducibility.
    """
    score_lex: Dict[str, float] = {}
    category_lex: Dict[str, Set[str]] = defaultdict(set)

    d = Path(dir_path)
    if not d.exists():
        log("INFO", f"Lexicons dir not found: {dir_path} (scoring will be limited to tokens)")
        return score_lex, category_lex

    files = sorted([p for p in d.iterdir() if p.suffix.lower()==".csv" and p.is_file()])
    if not files:
        log("INFO", f"No lexicon CSVs under {dir_path} (scoring will be limited to tokens)")
        return score_lex, category_lex

    for csv_path in files:
        try:
            with open(csv_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                cols = [c.strip().lower() for c in reader.fieldnames or []]
                is_score = ("score" in cols) or ("weight" in cols)
                is_cat = ("category" in cols)
                if not (is_score or is_cat) or ("token" not in cols):
                    log("WARN", f"Skipping lexicon (unknown schema): {csv_path.name}")
                    continue
                n = 0; bad = 0; dup = 0
                for row in reader:
                    tok = str(row.get("token", "")).strip().lower()
                    if not tok:
                        bad += 1; continue
                    if is_score:
                        sraw = row.get("score", row.get("weight"))
                        try:
                            sc = float(sraw)
                            if tok in score_lex:
                                dup += 1
                            score_lex[tok] = sc
                            n += 1
                        except Exception:
                            bad += 1
                    if is_cat:
                        cat = str(row.get("category", "")).strip()
                        if cat:
                            # Keep exact label casing; downstream we will enforce FUN order where relevant
                            category_lex[tok].add(cat)
                            n += 1
                t = "score" if is_score else "category"
                if is_score and is_cat:
                    t = "score+category"
                log("INFO", f"Loaded lexicon: {csv_path.name} rows_ok={n} dup_scores={dup} bad_rows={bad} type={t}")
        except Exception as e:
            log("WARN", f"Failed to read lexicon file {csv_path.name}: {e}")

    return score_lex, category_lex

# ----------------------- Per-app tokenization & scoring ----------------------
def counter_to_rows(cnt: Counter, per_k: float) -> List[Dict[str, object]]:
    total = sum(cnt.values()) or 1
    rows = []
    for tok, c in cnt.most_common():
        rows.append({"token": tok, "count": int(c), "freq_per_1k": (c / total) * per_k})
    return rows

def aggregate_categories(cnt: Counter, total_tokens: int, cat_lex: Dict[str, Set[str]], per_k: float) -> Dict[str, float]:
    # Sum counts per category; tokens can map to multiple categories
    cat_counts: Dict[str, int] = defaultdict(int)
    for tok, c in cnt.items():
        cats = cat_lex.get(tok, ())
        for cat in cats:
            cat_counts[cat] += c
    # Build outputs
    out = {}
    # FUN first
    for cat in CONFIG["FUN_ORDER"]:
        if cat in cat_counts:
            out[f"count::{cat}"] = cat_counts[cat]
            out[f"rate_per_1k::{cat}"] = (cat_counts[cat] / max(1, total_tokens)) * per_k
        else:
            out[f"count::{cat}"] = 0
            out[f"rate_per_1k::{cat}"] = 0.0
    # Then any other categories, alphabetically
    for cat in sorted([c for c in cat_counts.keys() if c not in CONFIG["FUN_ORDER"]]):
        out[f"count::{cat}"] = cat_counts[cat]
        out[f"rate_per_1k::{cat}"] = (cat_counts[cat] / max(1, total_tokens)) * per_k
    return out

def aggregate_scores(cnt: Counter, total_tokens: int, score_lex: Dict[str, float], per_k: float) -> Dict[str, float]:
    score_sum = 0.0
    hits = 0
    for tok, c in cnt.items():
        sc = score_lex.get(tok)
        if sc is not None:
            score_sum += sc * c
            hits += c
    return {
        "score_sum": score_sum,
        "score_per_1k": (score_sum / max(1, total_tokens)) * per_k,
        "score_hits": hits,
    }

# ----------------------- Review-level scoring ----------------------
def score_review_tokens(tokens: List[str], cat_lex: Dict[str, Set[str]]) -> Dict[str, int]:
    """Return dict with counts per category for a single review."""
    counts: Dict[str, int] = defaultdict(int)
    for tok in tokens:
        for cat in cat_lex.get(tok, ()):
            counts[cat] += 1
    return counts

def order_fun_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Put Flow/Utility/Nostalgia columns first in both counts and rates/props when present."""
    if df is None or df.empty:
        return df
    fun = CONFIG["FUN_ORDER"]
    # Determine column families
    cols = list(df.columns)
    fun_counts = [f"count::{c}" for c in fun if f"count::{c}" in cols]
    fun_rates  = [f"rate_per_1k::{c}" for c in fun if f"rate_per_1k::{c}" in cols]
    fun_props  = [f"{c}_prop" for c in fun if f"{c}_prop" in cols]
    fun_per100 = [f"{c}_per_100" for c in fun if f"{c}_per_100" in cols]
    fun_shares = [f"{c}_share" for c in fun if f"{c}_share" in cols]
    # Include None_* columns after FUN for shares only
    none_share = [c for c in ["None_share"] if c in cols]

    # Other count/rate/prop columns
    other_counts = [c for c in cols if c.startswith("count::") and c not in fun_counts]
    other_rates  = [c for c in cols if c.startswith("rate_per_1k::") and c not in fun_rates]
    other_props  = [c for c in cols if c.endswith("_prop") and c not in fun_props]
    other_per100 = [c for c in cols if c.endswith("_per_100") and c not in fun_per100]
    other_shares = [c for c in cols if c.endswith("_share") and c not in fun_shares + none_share]

    # Build ordered list; keep id/meta columns first
    id_like = [c for c in cols if c in ("appid","reviewId","pairIndex","type","total_tokens","unique_tokens","tokens_in_review","charLen","isPositive","zero_hit","n_reviews","n_with_fun","pct_with_fun")]
    # Then FUN, then others, then the rest
    ordered = id_like + \
              [c for c in cols if c not in id_like and not (c.startswith("count::") or c.startswith("rate_per_1k::") or c.endswith("_prop") or c.endswith("_per_100") or c.endswith("_share"))] + \
              fun_counts + other_counts + fun_rates + other_rates + fun_props + other_props + fun_per100 + other_per100 + fun_shares + none_share + other_shares

    # Drop duplicates but preserve order
    seen = set()
    final_cols = []
    for c in ordered:
        if c in cols and c not in seen:
            final_cols.append(c); seen.add(c)
    # Add any columns we missed
    for c in cols:
        if c not in seen:
            final_cols.append(c); seen.add(c)
    return df[final_cols]

# ----------------------- Per-pair deltas ----------------------
def per_pair_deltas(per_app_df: "pd.DataFrame", pairs_df: "pd.DataFrame", metric_cols: List[str]) -> "pd.DataFrame":
    """
    Join per-app metrics with pairs and compute top-minus-bottom deltas for metric_cols.
    Robust to dtype mismatches (normalizes appid as str and pairIndex numeric).
    """
    df_left = per_app_df.copy()
    df_right = pairs_df.copy()

    if "appid" in df_left.columns:
        df_left["appid"] = df_left["appid"].astype(str)
    if "appid" in df_right.columns:
        df_right["appid"] = df_right["appid"].astype(str)
    if "pairIndex" in df_right.columns:
        df_right["pairIndex"] = pd.to_numeric(df_right["pairIndex"], errors="coerce")

    p = df_right[[c for c in ["pairIndex","type","appid"] if c in df_right.columns]].drop_duplicates()
    m = df_left.merge(p, on="appid", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["pairIndex", "top_appid", "bottom_appid"] + metric_cols)

    top = m[m["type"]=="top"].set_index("pairIndex")
    bot = m[m["type"]=="bottom"].set_index("pairIndex")
    inter = sorted(set(top.index).intersection(bot.index))

    rows = []
    for idx in inter:
        t = top.loc[idx]
        b = bot.loc[idx]
        row = {"pairIndex": int(idx), "top_appid": str(t.get("appid")), "bottom_appid": str(b.get("appid"))}
        for col in metric_cols:
            tv = t.get(col)
            bv = b.get(col)
            row[f"{col}_top"] = tv
            row[f"{col}_bottom"] = bv
            try:
                row[f"{col}_diff_top_minus_bottom"] = (float(tv) - float(bv)) if pd.notna(tv) and pd.notna(bv) else None
            except Exception:
                row[f"{col}_diff_top_minus_bottom"] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("pairIndex")
    return out

# ----------------------- Main ---------------------------
def main():
    if pd is None:
        log("ERROR", "pandas is not available; cannot proceed.")
        sys.exit(2)

    ensure_dirs()

    # 1) Gather per-app review CSVs
    app_csvs = list_review_csvs(CONFIG["REVIEWS_PER_APP_DIR"])
    if not app_csvs:
        log("ERROR", f"No per-app review CSVs found in {CONFIG['REVIEWS_PER_APP_DIR']}")
        sys.exit(1)
    log("INFO", f"Found {len(app_csvs)} per-app review CSVs")

    # 2) Load lexicons
    score_lex, category_lex = load_lexicons(CONFIG["LEXICONS_DIR"])
    have_scores = len(score_lex) > 0
    have_cats   = any(category_lex.values())
    if not have_cats:
        log("WARN", "No category lexicons found. FUN metrics will be empty.")
    else:
        # Log lexicon coverage stats for FUN labels
        flow_lex_tokens = sum(1 for _tok, cats in category_lex.items() if "Flow" in cats)
        util_lex_tokens = sum(1 for _tok, cats in category_lex.items() if "Utility" in cats)
        nost_lex_tokens = sum(1 for _tok, cats in category_lex.items() if "Nostalgia" in cats)
        total_lex_tokens = len(category_lex)
        log("INFO", f"Lexicon stats: tokens_with_any_category={total_lex_tokens} Flow={flow_lex_tokens} Utility={util_lex_tokens} Nostalgia={nost_lex_tokens}")

    # 3) Optionally load pairs (used later for per-pair deltas and to attach pairIndex/type to review rows)
    pairs_df = read_pairs(CONFIG["PAIRS_CSV"])

    # Accumulators
    overall_tokens = Counter()
    per_app_token_counts: Dict[str, Counter] = {}
    per_app_total_tokens: Dict[str, int] = {}
    per_review_rows: List[Dict[str, object]] = []
    per_app_cat_metrics: List[Dict[str, object]] = []
    per_app_score_metrics: List[Dict[str, object]] = []

    # 4) Process each app file (single pass -> builds both app-level and review-level)

    for app_csv in app_csvs:
        try:
            df = pd.read_csv(app_csv)
        except Exception as e:
            log("WARN", f"Failed to read {getattr(app_csv, 'name', str(app_csv))}: {e}")
            continue

        # Columns we may use
        missing_cols = [c for c in ["appid","review"] if c not in df.columns]
        if missing_cols:
            log("WARN", f"Skipping {getattr(app_csv, 'name', str(app_csv))} (missing columns: {missing_cols})")
            continue

        # Normalize basics
        df["appid"] = df["appid"].astype(str)
        appid_vals = df["appid"].dropna().unique()
        if len(appid_vals) == 0:
            log("WARN", f"Skipping {getattr(app_csv, 'name', str(app_csv))} (empty appid)")
            continue
        appid = str(appid_vals[0])

        # Prepare sentiment/charLen/id if available
        has_rev_id   = "reviewId" in df.columns
        has_pos      = "isPositive" in df.columns
        has_char_len = "charLen" in df.columns

        # Running counters for this app
        app_counter = Counter()
        total_tokens_this_app = 0
        # Per-app coverage tracking
        zero_hit_count_this_app = 0
        fun_hit_reviews_this_app = 0

        # Attach pairIndex/type if available (constant per app)
        pairIndex = None; pairType = None
        if pairs_df is not None and "appid" in pairs_df.columns:
            m = pairs_df[pairs_df["appid"]==appid]
            if not m.empty:
                pairIndex = m["pairIndex"].iloc[0] if "pairIndex" in m.columns else None
                pairType  = m["type"].iloc[0] if "type" in m.columns else None

        # --- Review-level loop ---
        # Infinite loop prevention: check DataFrame shape and use vectorized ops if possible
        n_reviews = len(df)
        if n_reviews == 0:
            log("WARN", f"App {appid} has no reviews; skipping.")
            continue
        if n_reviews > 1000000:
            log("WARN", f"App {appid} has an unusually large number of reviews ({n_reviews}); check for data issues.")
            # Optionally, break or skip if too large
            continue

    # Vectorized tokenization: fallback to row-wise if needed. Tokenization is deterministic given CONFIG.
        for idx, row in df.iterrows():
            text = str(row.get("review", "") or "")
            tokens = tokenize(text)
            n_tok = len(tokens)
            total_tokens_this_app += n_tok
            app_counter.update(tokens)

            # Only compute FUN review-level if we have categories
            flow_count = util_count = nost_count = 0
            if have_cats and n_tok > 0:
                cat_counts = score_review_tokens(tokens, category_lex)
                flow_count = int(cat_counts.get("Flow", 0))
                util_count = int(cat_counts.get("Utility", 0))
                nost_count = int(cat_counts.get("Nostalgia", 0))

            total_fun_hits = flow_count + util_count + nost_count
            zero_hit = (total_fun_hits == 0)
            if zero_hit:
                zero_hit_count_this_app += 1
            else:
                fun_hit_reviews_this_app += 1

            # Proportions within FUN-only hits (exclude None dimension)
            flow_prop = util_prop = nost_prop = None
            if total_fun_hits > 0:
                flow_prop = flow_count / total_fun_hits
                util_prop = util_count / total_fun_hits
                nost_prop = nost_count / total_fun_hits

            # per-100 tokens intensities (counts normalized by review token length)
            base100 = CONFIG["PER_100_TOKENS"]
            denom = max(1, n_tok)
            flow_p100 = (flow_count / denom) * base100
            util_p100 = (util_count / denom) * base100
            nost_p100 = (nost_count / denom) * base100

            # Soft None share and normalized shares across 4-dimensions
            # Raw shares (pre-clamp): per-token rates for F/U/N; None = 1 - sum(F,U,N)
            f_share_raw = (flow_count / denom) if denom > 0 else 0.0
            u_share_raw = (util_count / denom) if denom > 0 else 0.0
            n_share_raw = (nost_count / denom) if denom > 0 else 0.0
            none_raw = 1.0 - (f_share_raw + u_share_raw + n_share_raw)
            f_share, u_share, nst_share, none_share = _renormalize_shares(f_share_raw, u_share_raw, n_share_raw, none_raw)

            # Gather row
            row_out = {
                "appid": appid,
                "reviewId": row.get("reviewId") if has_rev_id else None,
                "charLen": row.get("charLen") if has_char_len else (len(text) if text else None),
                "isPositive": row.get("isPositive") if has_pos else None,
                "tokens_in_review": n_tok,
                # Counts (FUN order)
                "Flow_count": flow_count,
                "Utility_count": util_count,
                "Nostalgia_count": nost_count,
                # Proportions within-FUN (exclude None; NaN for zero-hit)
                "Flow_prop": flow_prop,
                "Utility_prop": util_prop,
                "Nostalgia_prop": nost_prop,
                # Per-100 tokens (absolute intensity)
                "Flow_per_100": flow_p100,
                "Utility_per_100": util_p100,
                "Nostalgia_per_100": nost_p100,
                # Normalized 4-way shares that sum to 1.0 per review
                "Flow_share": f_share,
                "Utility_share": u_share,
                "Nostalgia_share": nst_share,
                "None_share": none_share,
                "zero_hit": bool(zero_hit),
                "pairIndex": pairIndex,
                "type": pairType,
            }
            # Optionally drop metrics per config to keep outputs light
            if not CONFIG.get("OUTPUT_PER100", True):
                for k in ("Flow_per_100","Utility_per_100","Nostalgia_per_100"):
                    row_out.pop(k, None)
            if not CONFIG.get("OUTPUT_SHARES", True):
                for k in ("Flow_share","Utility_share","Nostalgia_share","None_share"):
                    row_out.pop(k, None)
            per_review_rows.append(row_out)

        # Save per-app token frequencies
        per_app_token_counts[appid] = app_counter
        per_app_total_tokens[appid] = total_tokens_this_app
        overall_tokens.update(app_counter)

        # Write per-app token table
        token_rows = counter_to_rows(app_counter, CONFIG["PER_K_TOKENS"])
        out_path = Path(CONFIG["TOKENS_PER_APP_DIR"]) / f"{appid}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["token","count","freq_per_1k"])
            w.writeheader()
            for r in token_rows:
                w.writerow(r)
        # Log per-app coverage and lexicon hit summary
        fun_hit_pct = (fun_hit_reviews_this_app / n_reviews) if n_reviews else 0.0
        zhit_pct = (zero_hit_count_this_app / n_reviews) if n_reviews else 0.0
        # If category lexicons exist, summarize FUN category totals for the app
        fun_cat_summary = {}
        if have_cats and total_tokens_this_app > 0:
            cat_vals_tmp = aggregate_categories(app_counter, total_tokens_this_app, category_lex, CONFIG["PER_K_TOKENS"])
            for _c in CONFIG["FUN_ORDER"]:
                fun_cat_summary[_c] = int(cat_vals_tmp.get(f"count::{_c}", 0))
        log("INFO", (
            f"App {appid}: reviews={n_reviews} tokens={total_tokens_this_app} unique={len(app_counter)} "
            f"fun_hit_reviews={fun_hit_reviews_this_app} ({fun_hit_pct:.1%}) zero_hit={zero_hit_count_this_app} ({zhit_pct:.1%}) "
            + (f"FUN_counts={fun_cat_summary} " if fun_cat_summary else "")
            + f"-> {out_path}"
        ))

        # --- App-level dictionary metrics (categories & scores) ---
        base = {"appid": appid, "total_tokens": total_tokens_this_app, "unique_tokens": len(app_counter)}
        if have_cats:
            cat_vals = aggregate_categories(app_counter, total_tokens_this_app, category_lex, CONFIG["PER_K_TOKENS"])
            per_app_cat_metrics.append({**base, **cat_vals})
        if have_scores:
            score_vals = aggregate_scores(app_counter, total_tokens_this_app, score_lex, CONFIG["PER_K_TOKENS"])
            per_app_score_metrics.append({**base, **score_vals})

    # 5) Write overall tokens table
    overall_path = Path(CONFIG["TOKENS_OVERALL_CSV"])
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overall_path, "w", newline='', encoding='utf-8') as f:
        rows = counter_to_rows(overall_tokens, CONFIG["PER_K_TOKENS"])
        if rows:
            w = csv.DictWriter(f, fieldnames=["token","count","freq_per_1k"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log("INFO", f"Overall unique tokens={len(overall_tokens)} -> {CONFIG['TOKENS_OVERALL_CSV']}")

    # 6) Write app-level category metrics (FUN ordered)
    df_cat = None
    if per_app_cat_metrics:
        df_cat = pd.DataFrame(per_app_cat_metrics).fillna(0)
        # Enforce appid as str
        df_cat["appid"] = df_cat["appid"].astype(str)
        # Reorder columns to FUN-first
        df_cat = order_fun_columns(df_cat)
        df_cat.to_csv(CONFIG["PER_APP_CATEGORY_CSV"], index=False)
        log("INFO", f"Wrote per-app category metrics -> {CONFIG['PER_APP_CATEGORY_CSV']}")
    else:
        log("INFO", "No category lexicon metrics to write (no category lexicons found)")

    # 7) Write app-level score metrics (if any score lexicon provided)
    df_sc = None
    if per_app_score_metrics:
        df_sc = pd.DataFrame(per_app_score_metrics).fillna(0)
        df_sc["appid"] = df_sc["appid"].astype(str)
        df_sc.to_csv(CONFIG["PER_APP_SCORE_CSV"], index=False)
        log("INFO", f"Wrote per-app score metrics -> {CONFIG['PER_APP_SCORE_CSV']}")
    else:
        log("INFO", "No score lexicon metrics to write (no score lexicons found)")

    # 8) Per-pair deltas (app-level)
    if pairs_df is not None and df_cat is not None:
        metric_cols = [f"rate_per_1k::{c}" for c in CONFIG["FUN_ORDER"] if f"rate_per_1k::{c}" in df_cat.columns]
        if not metric_cols:
            metric_cols = [f"count::{c}" for c in CONFIG["FUN_ORDER"] if f"count::{c}" in df_cat.columns]
        df_cat_pairs = per_pair_deltas(df_cat, pairs_df, metric_cols)
        df_cat_pairs = order_fun_columns(df_cat_pairs)
        df_cat_pairs.to_csv(CONFIG["PER_PAIR_CATEGORY_CSV"], index=False)
        log("INFO", f"Wrote per-pair category deltas -> {CONFIG['PER_PAIR_CATEGORY_CSV']}")
    elif pairs_df is None:
        log("INFO", "Pairs CSV missing; skipping per-pair deltas for categories.")

    if pairs_df is not None and df_sc is not None:
        metric_cols_sc = [c for c in ["score_per_1k","score_sum","score_hits"] if c in df_sc.columns]
        if metric_cols_sc:
            df_sc_pairs = per_pair_deltas(df_sc, pairs_df, metric_cols_sc)
            df_sc_pairs.to_csv(CONFIG["PER_PAIR_SCORE_CSV"], index=False)
            log("INFO", f"Wrote per-pair score deltas -> {CONFIG['PER_PAIR_SCORE_CSV']}")

    # 9) Write review-level table
    df_reviews = pd.DataFrame(per_review_rows)
    if not df_reviews.empty:
        # Normalize dtypes
        df_reviews["appid"] = df_reviews["appid"].astype(str)
        # Reorder columns to FUN-first groupings
        df_reviews = order_fun_columns(df_reviews)
        df_reviews.to_csv(CONFIG["PER_REVIEW_FUN_CSV"], index=False)
        log("INFO", f"Wrote per-review FUN metrics -> {CONFIG['PER_REVIEW_FUN_CSV']}")
    else:
        log("WARN", "No per-review rows generated; review-level outputs skipped.")

    # 10) Per-app aggregation of review-level proportions (exclude zero-hit) and normalized shares/intensities
    df_app_agg = None
    if not df_reviews.empty:
        g = df_reviews.groupby("appid", dropna=False)

        def _agg_props(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                return pd.Series({"mean": None, "median": None})
            return pd.Series({"mean": float(s.mean()), "median": float(s.median())})

        rows = []
        for appid, sub in g:
            n_rev = len(sub)
            n_with_fun = int((sub["zero_hit"]==False).sum())
            pct_with_fun = (n_with_fun / n_rev) if n_rev else None
            zero_hit_pct = (1 - pct_with_fun) if pct_with_fun is not None else None

            # Means/medians of proportions EXCLUDING zero-hit
            nz = sub[sub["zero_hit"]==False]
            agg = {}
            for cat in CONFIG["FUN_ORDER"]:
                m = _agg_props(nz[f"{cat}_prop"]) if f"{cat}_prop" in nz.columns else pd.Series({"mean": None,"median":None})
                agg[f"{cat}_prop_mean"] = m["mean"]
                agg[f"{cat}_prop_median"] = m["median"]

            # Simple review-level averages (including zeros)
            tokens_mean = float(pd.to_numeric(sub["tokens_in_review"], errors="coerce").dropna().mean()) if "tokens_in_review" in sub.columns else None
            charlen_mean = float(pd.to_numeric(sub["charLen"], errors="coerce").dropna().mean()) if "charLen" in sub.columns else None

            # Per-100 tokens averages (including zeros)
            if CONFIG.get("OUTPUT_PER100", True):
                for cat in CONFIG["FUN_ORDER"]:
                    key = f"{cat}_per_100"
                    if key in sub.columns:
                        agg[f"{key}_mean"] = float(pd.to_numeric(sub[key], errors="coerce").dropna().mean()) if not sub[key].dropna().empty else None

            # Normalized share means (including None). Stored in 0..1 scale for scientific clarity.
            if CONFIG.get("OUTPUT_SHARES", True):
                share_cols = ["Flow_share","Utility_share","Nostalgia_share","None_share"]
                for c in share_cols:
                    if c in sub.columns:
                        mean_val = pd.to_numeric(sub[c], errors="coerce").dropna().mean()
                        agg[f"{c}_mean"] = float(mean_val) if pd.notna(mean_val) else None

            # Attach pair info (if constant per app in df_reviews)
            pairIndex = sub["pairIndex"].dropna().unique()
            pairIndex = pairIndex[0] if len(pairIndex)>0 else None
            pairType = sub["type"].dropna().unique()
            pairType = pairType[0] if len(pairType)>0 else None

            rows.append({
                "appid": str(appid),
                "n_reviews": int(n_rev),
                "n_with_fun": int(n_with_fun),
                "pct_with_fun": pct_with_fun,
                "zero_hit_pct": zero_hit_pct,
                "tokens_in_review_mean": tokens_mean,
                "charLen_mean": charlen_mean,
                "pairIndex": pairIndex,
                "type": pairType,
                **agg
            })

        df_app_agg = pd.DataFrame(rows)
        df_app_agg = order_fun_columns(df_app_agg)
        df_app_agg.to_csv(CONFIG["PER_APP_REVIEW_AGG_CSV"], index=False)
        log("INFO", f"Wrote per-app review-level aggregates -> {CONFIG['PER_APP_REVIEW_AGG_CSV']}")


    # 11) New output: per-app normalized FUN proportions and zero-hit proportion
    # -------------------------------------------------------------
    # For each appid:
    #   - f, u, n: normalized proportions of Flow, Utility, Nostalgia hits (among reviews with at least one FUN hit)
    #   - z: proportion of reviews with zero FUN hits
    # Output: CSV with columns: appid, f, u, n, z
    per_app_fun_norm_rows = []
    if not df_reviews.empty:
        grouped = df_reviews.groupby("appid", dropna=False)
        for appid, sub in grouped:
            n_reviews = len(sub)
            # Reviews with at least one FUN hit
            fun_hit_mask = (sub["zero_hit"] == False)
            fun_reviews = sub[fun_hit_mask]
            # Sum Flow/Utility/Nostalgia counts across these reviews
            f_sum = fun_reviews["Flow_count"].sum() if "Flow_count" in fun_reviews.columns else 0
            u_sum = fun_reviews["Utility_count"].sum() if "Utility_count" in fun_reviews.columns else 0
            n_sum = fun_reviews["Nostalgia_count"].sum() if "Nostalgia_count" in fun_reviews.columns else 0
            total_fun = f_sum + u_sum + n_sum
            # Normalized proportions (f+u+n=1 if total_fun>0, else all 0)
            if total_fun > 0:
                f_norm = f_sum / total_fun
                u_norm = u_sum / total_fun
                n_norm = n_sum / total_fun
            else:
                f_norm = u_norm = n_norm = 0.0
            # z: proportion of reviews with zero FUN hits
            z = float((sub["zero_hit"] == True).sum()) / n_reviews if n_reviews > 0 else None
            per_app_fun_norm_rows.append({
                "appid": str(appid),
                "flow": f_norm,
                "utility": u_norm,
                "nostalgia": n_norm,
                "z": z
            })
        # Write to CSV
        fun_norm_path = Path(CONFIG["OUT_ROOT"]) / "per_app_fun_normalized.csv"
        try:
            pd.DataFrame(per_app_fun_norm_rows).to_csv(fun_norm_path, index=False)
            log("INFO", f"Wrote per-app normalized FUN table -> {fun_norm_path}")
        except Exception as e:
            log("WARN", f"Failed to write per-app normalized FUN table: {e}")

    # 12) Summary TXT
    lines = []
    lines.append("Step 09 summary")
    # Token tables
    lines.append(f"Overall unique tokens: {len(overall_tokens)}")
    # App-level availability
    if df_cat is not None and not df_cat.empty:
        lines.append(f"Per-app category metrics: {CONFIG['PER_APP_CATEGORY_CSV']}")
    else:
        lines.append("Per-app category metrics: (none)")
    if df_sc is not None and not df_sc.empty:
        lines.append(f"Per-app score metrics: {CONFIG['PER_APP_SCORE_CSV']}")
    else:
        lines.append("Per-app score metrics: (none)")

    # Review-level availability
    if not df_reviews.empty:
        total_reviews = len(df_reviews)
        zero_hits = int((df_reviews["zero_hit"]==True).sum()) if "zero_hit" in df_reviews.columns else 0
        lines.append(f"Per-review FUN table: {CONFIG['PER_REVIEW_FUN_CSV']} (rows={total_reviews}, zero_hit={zero_hits})")
        if df_app_agg is not None and not df_app_agg.empty:
            lines.append(f"Per-app review aggregates: {CONFIG['PER_APP_REVIEW_AGG_CSV']}")
            # Quick headline: mean Flow_prop_mean across apps
            flow_means = pd.to_numeric(df_app_agg.get("Flow_prop_mean"), errors="coerce").dropna()
            if not flow_means.empty:
                lines.append(f"Mean of app Flow_prop_mean across apps: {flow_means.mean():.3f}")
        # Add note about new output
        lines.append(f"Per-app normalized FUN table: {fun_norm_path if 'fun_norm_path' in locals() else '(not written)'}")
    else:
        lines.append("Per-review FUN table: (none)")

    # Scientific notes and assumptions for reproducibility
    lines.append("")
    lines.append("Scientific notes")
    lines.append("- Lexicon sparsity: Many reviews have zero hits; zero_hit and pct_with_fun are reported.")
    lines.append("- Soft None share: None = 1 - (F + U + N) using per-token rates; we clamp to [0,1] and renormalize F/U/N/None to sum to 1.0.")
    lines.append("- Intensities vs. shares: per-100 intensities are absolute; normalized shares capture composition and include None.")
    lines.append("- Determinism: No randomness; results are deterministic for a given input and configuration.")

    try:
        with open(CONFIG["SUMMARY_TXT"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log("INFO", f"Wrote summary -> {CONFIG['SUMMARY_TXT']}")
    except Exception as e:
        log("WARN", f"Failed to write summary TXT: {e}")

    # Final console summary
    print("Summary:")
    print(f"  Token tables per app: {CONFIG['TOKENS_PER_APP_DIR']}")
    print(f"  Overall token table: {CONFIG['TOKENS_OVERALL_CSV']}")
    if df_cat is not None and not df_cat.empty:
        print(f"  Per-app categories: {CONFIG['PER_APP_CATEGORY_CSV']}")
    if pairs_df is not None and df_cat is not None and not df_cat.empty:
        print(f"  Per-pair category deltas: {CONFIG['PER_PAIR_CATEGORY_CSV']}")
    if not df_reviews.empty:
        print(f"  Per-review FUN: {CONFIG['PER_REVIEW_FUN_CSV']}")
        print(f"  Per-app review aggregates: {CONFIG['PER_APP_REVIEW_AGG_CSV']}")
        print(f"  Per-app normalized FUN: {fun_norm_path if 'fun_norm_path' in locals() else '(not written)'}")

if __name__ == "__main__":
    main()