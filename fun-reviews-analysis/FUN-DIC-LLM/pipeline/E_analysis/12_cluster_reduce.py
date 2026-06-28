# -*- coding: utf-8 -*-

'''
Step 12: Cluster Reduce - Clusters review embeddings per app and selects representative reviews and summaries.

• Supports k-means and HDBSCAN clustering algorithms
• Selects medoid representatives and generates cluster summaries
• Writes assignments, representatives, and summaries to output files

Inputs: Embeddings parquet or JSONL file, configuration files.
Outputs: cluster_assignments.csv, cluster_reps.csv, cluster_summaries.csv (in data/analysis/12/).
'''
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.Z_utils.common import ensure_dir, log, find_project_root  # project-style logging

# Optional parquet engine fallback
try:
    import pyarrow  # noqa: F401
    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False

SEED = 7
np.random.seed(SEED)

# Always resolve data paths to root, regardless of working directory
ROOT = find_project_root()
DATA = ROOT / "data"
# ----------------------------------------------------------------------------
# Environment & defaults
# ----------------------------------------------------------------------------
@dataclass
class Env:
    ROOT: Path
    DATA: Path
    IN_EMB: Path
    OUT_DIR: Path
    OUT_ASSIGN: Path
    OUT_REPS: Path
    OUT_SUMM: Path


def get_env() -> Env:
    root = Path(__file__).parent.parent.resolve()
    data = root / "data"
    in_emb = data / "analysis" / "11" / "embeddings.parquet"
    out_dir = data / "analysis" / "12"
    return Env(
        ROOT=root,
        DATA=data,
        IN_EMB=in_emb,
        OUT_DIR=out_dir,
        OUT_ASSIGN=out_dir / "cluster_assignments.csv",
        OUT_REPS=out_dir / "cluster_reps.csv",
        OUT_SUMM=out_dir / "cluster_summaries.csv",
    )


DEFAULTS = {
    "algo": "kmeans",          # kmeans | hdbscan
    "min_k": 8,
    "max_k": 24,
    "summary_mode": "concat",   # concat | none
    "summary_max_reviews": 3,
    "summary_max_chars": 400,
}

# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def _read_embeddings_any(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix('.jsonl')
    if alt.exists():
        rows = []
        with alt.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(pd.read_json(line, typ='series'))
        if rows:
            return pd.DataFrame(rows)
    raise SystemExit(f"[12][fatal] Embeddings not found: {path.as_posix()} (and no JSONL fallback)")


# ----------------------------------------------------------------------------
# Clustering backends
# ----------------------------------------------------------------------------

def heuristic_k(n: int, kmin: int, kmax: int) -> int:
    # Simple, robust heuristic: k ≈ sqrt(n), clipped
    k = int(round(math.sqrt(max(n, 1))))
    k = max(kmin, min(k, kmax))
    k = min(k, n)  # cannot exceed n
    if k <= 0:
        k = 1
    return k


def fit_kmeans(X: np.ndarray, n_clusters: int, seed: int = SEED) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise SystemExit(
            "[12][fatal] scikit-learn is required for kmeans. Install with:\n"
            "    pip install -U scikit-learn\n"
            f"Original error: {e}"
        )
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(256, max(32, X.shape[0])),
        random_state=seed,
        n_init="auto",
    )
    labels = mbk.fit_predict(X)
    return labels.astype(int)


def fit_hdbscan(X: np.ndarray, min_samples: int = None, min_cluster_size: int = None, metric: str = 'euclidean') -> np.ndarray:
    try:
        from hdbscan import HDBSCAN
    except Exception as e:
        raise SystemExit(
            "[12][fatal] hdbscan backend requested but package missing. Install with:\n"
            "    pip install -U hdbscan\n"
            f"Original error: {e}"
        )
    n = X.shape[0]
    # min_cluster_size heuristic if not provided
    if min_cluster_size is None:
        mcs = max(8, int(round(math.sqrt(n) * 0.75)))
        mcs = min(mcs, max(25, int(0.25 * n)))
    else:
        mcs = min_cluster_size
    model = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = model.fit_predict(X)  # -1 == noise
    return labels.astype(int)


    # ----------------------------------------------------------------------------
    # Main execution block
    # ----------------------------------------------------------------------------
def main():
    log("[12][debug] Entered main()")
    env = get_env()
    log(f"[12][debug] Env: ROOT={env.ROOT}, DATA={env.DATA}, IN_EMB={env.IN_EMB}, OUT_DIR={env.OUT_DIR}")
    ensure_dir(env.OUT_DIR)
    log(f"[12] Start — clustering reviews from {env.IN_EMB}")
    df = _read_embeddings_any(env.IN_EMB)
    log(f"[12][debug] Embeddings DataFrame shape: {df.shape}")
    if df.empty:
        log("[12][fatal] No embeddings found. Exiting.")
        return
    log(f"[12][debug] Unique appids: {df['appid'].nunique()} — {df['appid'].unique()[:10]}")
    # Group by appid for per-app clustering
    results_assign = []
    results_reps = []
    # Use default params
    algo = DEFAULTS["algo"]
    min_k = DEFAULTS["min_k"]
    max_k = DEFAULTS["max_k"]
    results_summ = []
    summary_mode = DEFAULTS["summary_mode"]
    summary_max_reviews = DEFAULTS["summary_max_reviews"]
    summary_max_chars = DEFAULTS["summary_max_chars"]
    for appid, group in df.groupby("appid"):
        log(f"[12][debug] Processing appid: {appid} — group shape: {group.shape}")
        if group.empty:
            log(f"[12][debug] Skipping empty group for appid: {appid}")
            continue
        X = np.vstack(group["emb"].to_numpy())
        n = X.shape[0]
        log(f"[12][debug] Embedding matrix shape for appid {appid}: {X.shape}")
        if algo == "kmeans":
            k = heuristic_k(n, min_k, max_k)
            log(f"[12][debug] Using kmeans with k={k} for appid {appid}")
            labels = fit_kmeans(X, k)
        else:
            log(f"[12][debug] Using hdbscan for appid {appid}")
            labels = fit_hdbscan(X)
        group = group.copy()
        group["cluster_id"] = labels
        # Cluster assignments
        assign = group[["appid", "reviewId", "cluster_id"]].copy()
        assign["cluster_size"] = assign.groupby("cluster_id")["cluster_id"].transform("count")
        results_assign.append(assign)
        # Representatives (medoids: closest to centroid)
        reps = []
        for cid, cgroup in group.groupby("cluster_id"):
            log(f"[12][debug] Cluster {cid} for appid {appid} — size: {len(cgroup)}")
            if len(cgroup) == 0:
                log(f"[12][debug] Skipping empty cluster {cid} for appid {appid}")
                continue
            emb_mat = np.vstack(cgroup["emb"].to_numpy())
            centroid = emb_mat.mean(axis=0)
            dists = np.linalg.norm(emb_mat - centroid, axis=1)
            idx = np.argmin(dists)
            rep_reviewId = cgroup.iloc[idx]["reviewId"]
            reps.append({
                "appid": appid,
                "cluster_id": cid,
                "rep_reviewId": rep_reviewId,
                "cluster_size": len(cgroup),
            })
            # Cluster summaries
            if summary_mode == "concat":
                cgroup_sorted = cgroup.sort_values("charLen", ascending=False).head(summary_max_reviews)
                texts = cgroup_sorted["text"].tolist()
                summary_text = " ".join(texts)
                if len(summary_text) > summary_max_chars:
                    summary_text = summary_text[:summary_max_chars] + "..."
                results_summ.append({
                    "appid": appid,
                    "cluster_id": cid,
                    "summary_text": summary_text,
                    "summary_n_reviews": len(texts),
                    "source_reviewIds": ",".join(str(rid) for rid in cgroup_sorted["reviewId"]),
                })
        results_reps.extend(reps)
    # Write outputs
    log(f"[12][debug] Writing assignments: {env.OUT_ASSIGN} — {sum(len(a) for a in results_assign)} rows")
    log(f"[12][debug] Writing representatives: {env.OUT_REPS} — {len(results_reps)} rows")
    log(f"[12][debug] Writing summaries: {env.OUT_SUMM} — {len(results_summ)} rows")
    if results_assign:
        pd.concat(results_assign).to_csv(env.OUT_ASSIGN, index=False)
    if results_reps:
        pd.DataFrame(results_reps).to_csv(env.OUT_REPS, index=False)
    if results_summ:
        pd.DataFrame(results_summ).to_csv(env.OUT_SUMM, index=False)
    log(f"[12] Wrote assignments: {env.OUT_ASSIGN}")
    log(f"[12] Wrote representatives: {env.OUT_REPS}")
    log(f"[12] Wrote summaries: {env.OUT_SUMM}")

if __name__ == "__main__":
    main()


