#!/usr/bin/env python3

'''
Step 08: Descriptive Stats - Generates descriptive statistics and charts for the reviews corpus.

• Aggregates and summarizes review data
• Produces tables and PDF charts for analysis
• Handles multiple input sources and optional samples

Inputs: Step 04 pairs, Step 05 reviews (master or per-app), Step 06 samples (optional).
Outputs: Tables, charts (PDF), summary text under ../data/analysis/08/.
'''

import os
from ..utils.common import log, ensure_dir
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / "data"
import json
import math
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

### Pipeline Step 08: Descriptive statistics
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
    "STEP_NAME": get_env("STATS_STEP_NAME", "08"),
    "INPUTS": {
        "pairs_csv": str(get_env("STATS_PAIRS_CSV", DATA / "steamspy_top_bottom_pairs.csv")),
        "reviews_all": str(get_env("STATS_REVIEWS_ALL", DATA / "reviews_all.csv")),
        "reviews_per_app_dir": str(get_env("STATS_REVIEWS_PER_APP_DIR", DATA / "reviews_per_app")),
        "samples_all": str(get_env("STATS_SAMPLES_ALL", DATA / "reviews_sampled_all.csv")),
        "step02_scored": str(get_env("STATS_STEP02_SCORED", DATA / "steamspy_scored_sorted.csv")),
    },
    "OUT_DIRS": {
        "root": str(get_env("STATS_OUT_ROOT", DATA / "analysis/08")),
        "tables": str(get_env("STATS_OUT_TABLES", DATA / "analysis/08/tables")),
        "charts": str(get_env("STATS_OUT_CHARTS", DATA / "analysis/08/charts")),
        "debug": str(get_env("STATS_OUT_DEBUG", DATA / "analysis/08/debug")),
    },
    "CHART": {
        "format": get_env("STATS_CHART_FORMAT", "pdf"),
        "dpi": get_env("STATS_CHART_DPI", 120),
        "style": get_env("STATS_CHART_STYLE", "default"),
        "figsize": (8, 5),
        "bins_charlen": get_env("STATS_CHART_BINS_CHARLEN", 60),
        "bins_pos": get_env("STATS_CHART_BINS_POS", 40),
        "bins_acc": get_env("STATS_CHART_BINS_ACC", 60),
        "clip_charlen": get_env("STATS_CHART_CLIP_CHARLEN", 3000),
    },
}

# ----------------- Logging -----------------

def custom_log(level: str, msg: str):
    log(f"[{level}] {msg}")

def ensure_dirs():
    for p in CONFIG["OUT_DIRS"].values():
        try:
            ensure_dir(p)
        except Exception as e:
            custom_log("ERROR", f"Failed to create dir: {p}. Error: {e}")

def require_pandas():
    if pd is None:
        custom_log("ERROR", "pandas is not available. Cannot proceed.")
        sys.exit(2)

def require_matplotlib():
    if plt is None:
        custom_log("WARN", "matplotlib not available. Charts will be skipped.")

# ----------------- IO helpers -----------------

def read_csv(path: str) -> Optional["pd.DataFrame"]:
    if not os.path.isfile(path):
        custom_log("WARN", f"Missing input file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        custom_log("ERROR", f"Failed to read CSV: {path}. Error: {e}")
        return None

def concat_per_app_reviews(dir_path: str) -> Optional["pd.DataFrame"]:
    if not os.path.isdir(dir_path):
        custom_log("WARN", f"Per-app reviews dir not found: {dir_path}")
        return None
    frames = []
    for fn in sorted(os.listdir(dir_path)):
        if not fn.lower().endswith('.csv'):
            continue
        full = os.path.join(dir_path, fn)
        try:
            frames.append(pd.read_csv(full))
        except Exception as e:
            custom_log("WARN", f"Failed to read per-app file: {fn}. Error: {e}")
    if not frames:
        custom_log("WARN", "No per-app CSVs found.")
        return None
    return pd.concat(frames, ignore_index=True)

# ----------------- Core computations -----------------

def per_app_from_reviews(df: "pd.DataFrame") -> "pd.DataFrame":
    d = df.copy()
    # Coerce types we use
    for c in ["appid", "isPositive", "charLen", "timestamp_created", "totalEnglishReviews"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # Per-app aggregates
    grp = d.groupby("appid", dropna=False)

    agg = grp.agg({
        "reviewId": "count",
        "isPositive": "sum" if "isPositive" in d.columns else "size",
        "charLen": ["median", "mean"] if "charLen" in d.columns else "size",
        "timestamp_created": ["min", "max"] if "timestamp_created" in d.columns else "size",
        "totalEnglishReviews": "max" if "totalEnglishReviews" in d.columns else "size",
    })
    agg.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in agg.columns]
    out = agg.reset_index()
    # Rename
    rename = {
        "reviewId_count": "n_reviews_fetched",
        "isPositive_sum": "n_positive",
        "charLen_median": "charLen_median",
        "charLen_mean": "charLen_mean",
        "timestamp_created_min": "ts_earliest",
        "timestamp_created_max": "ts_latest",
        "totalEnglishReviews_max": "totalEnglishReviews",
    }
    out = out.rename(columns=rename)
    # Derived
    if "n_positive" in out.columns and "n_reviews_fetched" in out.columns:
        out["positivity_ratio"] = out["n_positive"] / out["n_reviews_fetched"].replace(0, pd.NA)
    if "ts_earliest" in out.columns and "ts_latest" in out.columns:
        out["days_span"] = (
            pd.to_datetime(out["ts_latest"], unit="s", errors="coerce")
            - pd.to_datetime(out["ts_earliest"], unit="s", errors="coerce")
        ).dt.total_seconds() / 86400.0
    if "totalEnglishReviews" in out.columns:
        out["fetched_vs_total_ratio"] = out["n_reviews_fetched"] / out["totalEnglishReviews"].replace(0, pd.NA)
    return out

def sample_length_shares(samples: Optional["pd.DataFrame"]) -> Optional["pd.DataFrame"]:
    if samples is None:
        return None
    if not all(c in samples.columns for c in ["appid", "lengthBucket"]):
        return None
    s = samples.copy()
    # Normalize labels
    s["lengthBucket"] = s["lengthBucket"].astype(str).str.lower()
    pivot = (s.assign(n=1)
               .pivot_table(index="appid", columns="lengthBucket", values="n", aggfunc="sum", fill_value=0))
    # Ensure columns exist
    for col in ["short", "medium", "long"]:
        if col not in pivot.columns:
            pivot[col] = 0
    tot = pivot[["short","medium","long"]].sum(axis=1).replace(0, pd.NA)
    shares = pd.DataFrame({
        "appid": pivot.index,
        "sample_size": pivot[["short","medium","long"]].sum(axis=1).astype(int),
        "sample_len_share_short": (pivot["short"] / tot),
        "sample_len_share_medium": (pivot["medium"] / tot),
        "sample_len_share_long": (pivot["long"] / tot),
    }).reset_index(drop=True)
    return shares

def full_length_shares_via_tertiles(reviews: "pd.DataFrame") -> "pd.DataFrame":
    # Compute overall tertiles per app on charLen (not per-sentiment; documented in summary)
    d = reviews.copy()
    if "charLen" not in d.columns:
        return pd.DataFrame({"appid": [], "full_len_share_short": [], "full_len_share_medium": [], "full_len_share_long": []})
    d["appid"] = pd.to_numeric(d["appid"], errors="coerce")
    d["charLen"] = pd.to_numeric(d["charLen"], errors="coerce")
    d = d.dropna(subset=["appid", "charLen"]).copy()

    rows = []
    for appid, sub in d.groupby("appid"):
        vals = sub["charLen"].sort_values().values
        n = len(vals)
        if n == 0:
            rows.append({"appid": int(appid), "full_len_share_short": None, "full_len_share_medium": None, "full_len_share_long": None})
            continue
        # nearest-rank like
        q1 = vals[int(math.floor(0.333 * (n-1)))] if n > 1 else vals[0]
        q2 = vals[int(math.floor(0.666 * (n-1)))] if n > 1 else vals[0]
        # classify
        s = (sub.assign(_bucket=sub["charLen"].apply(lambda x: "short" if x <= q1 else ("medium" if x <= q2 else "long")))
                 ._bucket.value_counts())
        short = int(s.get("short", 0)); med = int(s.get("medium", 0)); lng = int(s.get("long", 0))
        tot = short + med + lng
        rows.append({
            "appid": int(appid),
            "full_len_share_short": (short / tot) if tot else None,
            "full_len_share_medium": (med / tot) if tot else None,
            "full_len_share_long": (lng / tot) if tot else None,
        })
    return pd.DataFrame(rows)

def merge_with_pairs(per_app: "pd.DataFrame", pairs: "pd.DataFrame") -> "pd.DataFrame":
    # Only keep one row per app in pairs
    keep = [c for c in ["pairIndex","type","appid","name","acceptanceScore","rankAcceptance","rankFiltered"] if c in pairs.columns]
    p = pairs[keep].drop_duplicates(subset=["appid"]).copy()
    m = per_app.merge(p, on="appid", how="left")
    # Debug: mismatches
    missing_in_pairs = m[m["pairIndex"].isna()][["appid"]]
    if not missing_in_pairs.empty:
        dbg = os.path.join(CONFIG["OUT_DIRS"]["debug"], "reviews_appids_missing_in_pairs.csv")
        missing_in_pairs.to_csv(dbg, index=False)
        custom_log("WARN", f"AppIDs present in reviews but missing in pairs: {len(missing_in_pairs)}. Saved: {dbg}")
    missing_in_reviews = p[~p["appid"].isin(per_app["appid"])][["appid"]]
    if not missing_in_reviews.empty:
        dbg = os.path.join(CONFIG["OUT_DIRS"]["debug"], "pairs_appids_missing_in_reviews.csv")
        missing_in_reviews.to_csv(dbg, index=False)
        custom_log("WARN", f"AppIDs present in pairs but missing in reviews: {len(missing_in_reviews)}. Saved: {dbg}")
    return m

def add_sample_and_length_shares(merged: "pd.DataFrame", samples: Optional["pd.DataFrame"], reviews: "pd.DataFrame") -> "pd.DataFrame":
    # Attach sample size + sample length shares
    if samples is not None:
        shares = sample_length_shares(samples)
    else:
        shares = None
    full_len = full_length_shares_via_tertiles(reviews)

    out = merged.copy()
    if shares is not None:
        out = out.merge(shares, on="appid", how="left")
        out["sample_size"] = out["sample_size"].fillna(0).astype(int)
    else:
        out["sample_size"] = 0
        for c in ["sample_len_share_short","sample_len_share_medium","sample_len_share_long"]:
            out[c] = pd.NA

    out = out.merge(full_len, on="appid", how="left")
    return out

def save_per_app_table(df: "pd.DataFrame", path: str):
    cols = [
        "pairIndex","type","appid","name",
        "totalEnglishReviews","n_reviews_fetched","sample_size","fetched_vs_total_ratio",
        "positivity_ratio","n_positive",
        "charLen_mean","charLen_median",
        "ts_earliest","ts_latest","days_span",
        "sample_len_share_short","sample_len_share_medium","sample_len_share_long",
        "full_len_share_short","full_len_share_medium","full_len_share_long",
        "acceptanceScore","rankAcceptance","rankFiltered",
    ]
    keep = [c for c in cols if c in df.columns]
    df[keep].sort_values(["pairIndex","type","appid"], na_position="last").to_csv(path, index=False)
    custom_log("INFO", f"Saved per-app descriptive stats: {path}")

def pairwise_deltas(per_app: "pd.DataFrame") -> "pd.DataFrame":
    # For each pairIndex, compute top-minus-bottom deltas for selected metrics
    m = per_app.copy()
    need = ["pairIndex","type","appid","positivity_ratio","charLen_median","days_span","fetched_vs_total_ratio",
            "sample_len_share_short","sample_len_share_medium","sample_len_share_long",
            "full_len_share_short","full_len_share_medium","full_len_share_long"]
    have = [c for c in need if c in m.columns]
    m = m[have]

    top = m[m["type"] == "top"].set_index("pairIndex")
    bot = m[m["type"] == "bottom"].set_index("pairIndex")
    inter = top.index.intersection(bot.index)
    rows = []
    for idx in sorted(inter):
        t = top.loc[idx]; b = bot.loc[idx]
        row = {"pairIndex": int(idx), "top_appid": int(t.get("appid")), "bottom_appid": int(b.get("appid"))}
        for metric in [c for c in m.columns if c not in ["pairIndex","type","appid"]]:
            tv = t.get(metric); bv = b.get(metric)
            row[f"{metric}_top"] = tv
            row[f"{metric}_bottom"] = bv
            try:
                row[f"{metric}_diff_top_minus_bottom"] = (float(tv) - float(bv)) if pd.notna(tv) and pd.notna(bv) else None
            except Exception:
                row[f"{metric}_diff_top_minus_bottom"] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pairIndex")

# ----------------- Charts -----------------

def charts(reviews: "pd.DataFrame", per_app: "pd.DataFrame", pairs: Optional["pd.DataFrame"], step02: Optional["pd.DataFrame"], charts_dir: str, cfg: Dict):
    if plt is None:
        return
    plt.style.use(cfg.get("style","default"))
    dpi = cfg.get("dpi",120)
    figsize = cfg.get("figsize",(8,5))
    figfmt = cfg.get("format","pdf")

    # charLen distribution (all reviews, clipped)
    if "charLen" in reviews.columns:
        x = pd.to_numeric(reviews["charLen"], errors="coerce")
        x = x.dropna()
        if cfg.get("clip_charlen") is not None:
            x = x.clip(upper=cfg["clip_charlen"])
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(x, bins=cfg.get("bins_charlen",60), color="#4C72B0", edgecolor="white")
        ax.set_title("Distribution of review charLen (clipped)")
        ax.set_xlabel("charLen")
        ax.set_ylabel("count")
        fig.tight_layout(); fig.savefig(os.path.join(charts_dir, f"hist_charLen.{figfmt}"), dpi=dpi, bbox_inches="tight", format=figfmt); plt.close(fig)
    else:
        custom_log("INFO", "Skipping charLen histogram (column missing)")

    # Positivity ratio histogram (per-app)
    if "positivity_ratio" in per_app.columns:
        pr = pd.to_numeric(per_app["positivity_ratio"], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(pr, bins=cfg.get("bins_pos",40), color="#55A868", edgecolor="white")
        ax.set_title("Per-app positivity ratio")
        ax.set_xlabel("positivity ratio")
        ax.set_ylabel("apps")
        fig.tight_layout(); fig.savefig(os.path.join(charts_dir, f"hist_positivity_ratio_per_app.{figfmt}"), dpi=dpi, bbox_inches="tight", format=figfmt); plt.close(fig)
    else:
        custom_log("INFO", "Skipping positivity histogram (metric missing)")

    # AcceptanceScore histogram
    plotted = False
    if pairs is not None and "acceptanceScore" in pairs.columns:
        xs = pd.to_numeric(pairs.drop_duplicates("appid")["acceptanceScore"], errors="coerce").dropna()
        if len(xs) > 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(xs, bins=cfg.get("bins_acc",60), color="#8172B2", edgecolor="white")
            ax.set_title("AcceptanceScore (selected apps)")
            ax.set_xlabel("acceptanceScore")
            ax.set_ylabel("apps")
            fig.tight_layout(); fig.savefig(os.path.join(charts_dir, f"hist_acceptanceScore_selected.{figfmt}"), dpi=dpi, bbox_inches="tight", format=figfmt); plt.close(fig)
            plotted = True
    if not plotted and step02 is not None and "acceptanceScore" in step02.columns:
        xs = pd.to_numeric(step02["acceptanceScore"], errors="coerce").dropna()
        if len(xs) > 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(xs, bins=cfg.get("bins_acc",60), color="#8172B2", edgecolor="white")
            ax.set_title("AcceptanceScore (Step 02, all)")
            ax.set_xlabel("acceptanceScore")
            ax.set_ylabel("apps")
            fig.tight_layout(); fig.savefig(os.path.join(charts_dir, f"hist_acceptanceScore_all_step02.{figfmt}"), dpi=dpi, bbox_inches="tight", format=figfmt); plt.close(fig)
        else:
            custom_log("INFO", "No values to plot for acceptanceScore")

# ----------------- Summary -----------------

def write_txt_summary(path: str, per_app: "pd.DataFrame", per_pair: "pd.DataFrame"):
    lines = []
    n_apps = per_app["appid"].nunique() if "appid" in per_app.columns else 0
    n_reviews = int(per_app["n_reviews_fetched"].sum()) if "n_reviews_fetched" in per_app.columns else 0
    lines.append("Step 08 summary")
    lines.append(f"Apps: {n_apps}")
    lines.append(f"Total reviews (fetched): {n_reviews}")
    if "positivity_ratio" in per_app.columns:
        pr = per_app["positivity_ratio"].dropna()
        if len(pr) > 0:
            lines.append(f"Positivity ratio (per-app): min={pr.min():.3f} med={pr.median():.3f} max={pr.max():.3f}")
    if "charLen_median" in per_app.columns:
        cm = per_app["charLen_median"].dropna()
        if len(cm) > 0:
            lines.append(f"charLen_median (per-app): min={cm.min():.1f} med={cm.median():.1f} max={cm.max():.1f}")
    if "fetched_vs_total_ratio" in per_app.columns:
        fv = per_app["fetched_vs_total_ratio"].dropna()
        if len(fv) > 0:
            under = (
                per_app[(per_app["fetched_vs_total_ratio"] < 0.99) & per_app["fetched_vs_total_ratio"].notna()]
                [["appid","name","fetched_vs_total_ratio"]]
                .sort_values("fetched_vs_total_ratio")
            )
            lines.append(f"Fetched/Total ratio: mean={fv.mean():.3f} min={fv.min():.3f}")
            if not under.empty:
                lines.append("Apps with fetched<total:")
                for _, r in under.iterrows():
                    lines.append(f"  appid={int(r['appid'])} name={str(r.get('name') or '')} ratio={float(r['fetched_vs_total_ratio']):.3f}")
    # Pairwise headline
    if not per_pair.empty:
        col = "positivity_ratio_diff_top_minus_bottom"
        if col in per_pair.columns:
            top5 = per_pair.sort_values(col, ascending=False).head(3)
            lines.append("Top pairs by positivity delta (top-bottom):")
            for _, r in top5.iterrows():
                lines.append(f"  pair={int(r['pairIndex'])} delta={float(r[col]):.3f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    custom_log("INFO", f"Saved summary TXT: {path}")

# ----------------- Main -----------------

def main():
    require_pandas(); ensure_dirs(); require_matplotlib()

    inp = CONFIG["INPUTS"]; outd = CONFIG["OUT_DIRS"]

    # Load inputs
    pairs = read_csv(inp["pairs_csv"])
    if pairs is None or getattr(pairs, "empty", False):
        custom_log("ERROR", "Pairs CSV is required (Step 04). Aborting.")
        print("Summary: 0 outputs. Missing pairs CSV")
        return

    # SAFE fallback from reviews_all.csv -> per-app CSVs
    reviews = read_csv(inp["reviews_all"])
    if reviews is None or getattr(reviews, "empty", False):
        custom_log("INFO", "reviews_all.csv missing or empty; falling back to per-app CSVs...")
        reviews = concat_per_app_reviews(inp["reviews_per_app_dir"])
    if reviews is None or getattr(reviews, "empty", False):
        custom_log("ERROR", "No reviews found in either reviews_all.csv or data/reviews_per_app/*.csv. Aborting.")
        print("Summary: 0 outputs. Missing reviews.")
        return

    samples = read_csv(inp["samples_all"])   # optional
    step02  = read_csv(inp["step02_scored"]) # optional

    # Compute per-app stats
    per_app = per_app_from_reviews(reviews)
    merged  = merge_with_pairs(per_app, pairs)
    merged2 = add_sample_and_length_shares(merged, samples, reviews)

    # Save per-app table
    per_app_path = os.path.join(outd["tables"], "per_app_descriptive_stats.csv")
    save_per_app_table(merged2, per_app_path)

    # Per-pair deltas
    per_pair = pairwise_deltas(merged2)
    per_pair_path = os.path.join(outd["tables"], "per_pair_deltas.csv")
    per_pair.to_csv(per_pair_path, index=False)
    custom_log("INFO", f"Saved per-pair comparison: {per_pair_path}")

    # Charts
    charts(reviews, merged2, pairs, step02, outd["charts"], CONFIG["CHART"])

    # TXT summary
    summary_path = os.path.join(outd["root"], "summary.txt")
    write_txt_summary(summary_path, merged2, per_pair)

    # Final summary
    n_charts = 0
    try:
        n_charts = len([f for f in os.listdir(outd["charts"]) if f.lower().endswith('.pdf')])
    except Exception:
        pass
    print("Summary:")
    print(f"  Apps: {merged2['appid'].nunique() if 'appid' in merged2.columns else 0}")
    print(f"  Reviews rows: {len(reviews)}")
    print(f"  Tables: {outd['tables']}")
    print(f"  Charts: {outd['charts']} (pdf count: {n_charts})")


if __name__ == "__main__":
    main()