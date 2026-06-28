#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Step 13a: Compare Methods - Compares FUN shares between dictionary and LLM methods and outputs statistical analyses.

• Compares results from dictionary and LLM scoring methods
• Outputs comparison tables, correlations, and divergences
• Calculates and reports statistical analyses

Inputs: per_app_review_agg.csv, per_app_llm_agg.csv (from previous steps).
Outputs: per_app_comparison.csv, statistical summaries, logs.
'''
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pipeline.Z_utils.common import ensure_dir, log


def get_paths(root: Path) -> Tuple[Path, Path, Path]:
    data = root / "data"
    per_app_dict = data / "analysis" / "09" / "per_app_review_agg.csv"
    per_app_llm = data / "analysis" / "13" / "tables" / "per_app_llm_agg.csv"
    out_csv = data / "analysis" / "13" / "tables" / "per_app_comparison.csv"
    return per_app_dict, per_app_llm, out_csv


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p||q) for discrete distributions represented as percent shares.
    Inputs in 0..100; converts to probabilities with smoothing.
    """
    p = np.clip(np.array(p, dtype=float) / 100.0, eps, 1.0)
    q = np.clip(np.array(q, dtype=float) / 100.0, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    root = Path(__file__).parent.parent.resolve()
    p_dict, p_llm, out_csv = get_paths(root)
    if not p_dict.exists() or not p_llm.exists():
        log("[13a][fatal] Missing inputs for comparison.")
        return

    df_d = pd.read_csv(p_dict, encoding="utf-8")
    df_l = pd.read_csv(p_llm, encoding="utf-8")
    df_d["appid"] = df_d["appid"].astype(str)
    df_l["appid"] = df_l["appid"].astype(str)

    # Select and rename required columns
    need_d = [
        "appid",
        "Flow_share_mean","Utility_share_mean","Nostalgia_share_mean","None_share_mean",
        "n_reviews","tokens_in_review_mean","pct_with_fun"
    ]
    miss = [c for c in need_d if c not in df_d.columns]
    if miss:
        log(f"[13a][fatal] Dictionary per-app agg missing columns: {miss}")
        return
    df_d2 = df_d[need_d].rename(columns={
        "Flow_share_mean":"Flow_share_dict",
        "Utility_share_mean":"Utility_share_dict",
        "Nostalgia_share_mean":"Nostalgia_share_dict",
        "None_share_mean":"None_share_dict",
    })

    need_l = ["appid","Flow","Utility","Nostalgia","None","confidence_mean"]
    miss = [c for c in need_l if c not in df_l.columns]
    if miss:
        log(f"[13a][fatal] LLM per-app agg missing columns: {miss}")
        return
    df_l2 = df_l[need_l].rename(columns={
        "Flow":"Flow_share_llm",
        "Utility":"Utility_share_llm",
        "Nostalgia":"Nostalgia_share_llm",
        "None":"None_share_llm",
    })

    merged = df_d2.merge(df_l2, on="appid", how="inner")
    ensure_dir(out_csv.parent)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    log(f"[13a] Wrote comparison CSV -> {out_csv.as_posix()}")

    # Correlations
    for dim in ["Flow","Utility","Nostalgia","None"]:
        dcol = f"{dim}_share_dict"; lcol = f"{dim}_share_llm"
        x = pd.to_numeric(merged[dcol], errors="coerce")
        y = pd.to_numeric(merged[lcol], errors="coerce")
        c = pd.concat([x, y], axis=1).dropna()
        if c.empty:
            log(f"[13a][info] No data for correlations on {dim}.")
            continue
        pearson = c.corr(method="pearson").iloc[0,1]
        spearman = c.corr(method="spearman").iloc[0,1]
        kld = kl_divergence(c.iloc[:,0].values, c.iloc[:,1].values)
        print(f"{dim}: pearson={pearson:.3f} spearman={spearman:.3f} KL(d||l)={kld:.3f}")


if __name__ == "__main__":
    main()
