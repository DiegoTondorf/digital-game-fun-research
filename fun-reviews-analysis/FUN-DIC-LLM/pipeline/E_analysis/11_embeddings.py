# pipeline/11_embeddings.py
# -*- coding: utf-8 -*-

'''
Step 11: Embeddings - Computes sentence embeddings for reviews and saves them for clustering and analysis.

• Supports multiple embedding providers and formats
• Saves embeddings in Parquet or JSONL for downstream steps
• Ensures reproducible paths and project-style logging

Inputs: Reviews CSV, configuration files.
Outputs: embeddings.parquet or embeddings.jsonl (in data/analysis/11/).
'''
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Project utils (consistent with other steps)
from pipeline.Z_utils.common import ensure_dir, log, find_project_root  # noqa: E402


# Always resolve data paths to root, regardless of working directory
ROOT = find_project_root()
DATA = ROOT / "data"

# Parquet (optional); we fallback to JSONL if not available
try:
    import pyarrow  # noqa: F401
    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False


# -----------------------------------------------------------------------------
# Defaults & environment
# -----------------------------------------------------------------------------
SEED = 7
np.random.seed(SEED)


@dataclass
class Env:
    ROOT: Path
    DATA: Path
    IN_SAMPLED: Path
    OUT_DIR: Path
    OUT_PATH: Path


def get_env() -> Env:
    root = Path(__file__).parent.parent.resolve()
    data = root / "data"
    in_sampled = data / "reviews_sampled_all.csv"
    out_dir = data / "analysis" / "11"
    out_path = out_dir / "embeddings.parquet"
    return Env(ROOT=root, DATA=data, IN_SAMPLED=in_sampled, OUT_DIR=out_dir, OUT_PATH=out_path)


# Defaults-first configuration (no args needed)
DEFAULTS = {
    # Backend: 'sbert' (sentence-transformers) or 'ollama'
    "provider": "sbert",
    # Models:
    #  - sbert : "sentence-transformers/all-MiniLM-L6-v2" (384-dim, fast CPU)
    #  - ollama: "nomic-embed-text" (768-dim; ensure `ollama pull nomic-embed-text`)
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    # Column name for input review text
    "text_col": "review",
    # Batching only used by 'sbert'
    "batch_size": 128,
    # Optional quick cap on rows (None for all)
    "max_rows": None,
    # Ollama host/timeouts
    "ollama_host": "http://localhost:11434",
    "ollama_timeout": 120,
}


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def to_parquet_or_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if _HAVE_PYARROW:
        df.to_parquet(out_path, index=False)
        log(f"[11] Wrote Parquet: {out_path.as_posix()}")
    else:
        alt = out_path.with_suffix(".jsonl")
        with alt.open("w", encoding="utf-8") as fh:
            for _, row in df.iterrows():
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        log(f"[11][warn] pyarrow not found; wrote JSONL fallback: {alt.as_posix()}")


def read_reviews(path: Path, text_col: str, max_rows: Optional[int]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[11][fatal] Input not found: {path.as_posix()}")

    df = pd.read_csv(path, encoding="utf-8")

    # appid
    if "appid" not in df.columns:
        alt = [c for c in df.columns if str(c).lower() in ("app", "app_id", "appid")]
        if alt:
            df.rename(columns={alt[0]: "appid"}, inplace=True)
        else:
            raise SystemExit("[11][fatal] Missing required column 'appid'")

    # reviewId
    if "reviewId" not in df.columns:
        alt = [c for c in df.columns if str(c).lower() in ("recommendationid", "review_id", "id")]
        if alt:
            df["reviewId"] = df[alt[0]]
        else:
            # Stable synthetic fallback
            df["reviewId"] = [f"idx:{i}" for i in range(len(df))]

    # review text
    if text_col not in df.columns:
        # Try to guess
        for c in ("review", "text", "body", "content"):
            if c in df.columns:
                text_col = c
                break
        else:
            raise SystemExit(f"[11][fatal] Text column '{text_col}' not found and no fallback detected.")

    # Normalize helper columns
    df["text"] = (
        df[text_col].astype(str).fillna("")
          .str.replace("\r\n", "\n")
          .str.replace("\r", "\n")
          .str.replace("\t", " ")
          .str.strip()
    )

    if "charLen" not in df.columns:
        df["charLen"] = df["text"].map(len)

    if "isPositive" not in df.columns:
        if "voted_up" in df.columns:
            df["isPositive"] = df["voted_up"].astype(int)
        else:
            df["isPositive"] = np.nan

    # Deterministic subset if requested
    if max_rows is not None:
        df = df.sort_values(["appid", "reviewId"]).head(int(max_rows)).copy()

    # Length buckets: short / medium / long
    def bucket(n: int) -> str:
        if n < 80:
            return "short"
        if n < 300:
            return "medium"
        return "long"

    df["lenBucket"] = df["charLen"].map(lambda n: bucket(int(n)))

    keep = ["appid", "reviewId", "text", "charLen", "isPositive", "lenBucket"]
    return df[keep].copy()


# -----------------------------------------------------------------------------
# Embedding providers
# -----------------------------------------------------------------------------

class SbertEmbedder:
    def __init__(self, model_name: str, batch_size: int = 128):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise SystemExit(
                "[11][fatal] sentence-transformers not installed. "
                "Install with:\n    pip install -U sentence-transformers pyarrow\n"
                f"Original error: {e}"
            )
        self.model = SentenceTransformer(model_name)
        self.batch_size = max(1, int(batch_size))

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                batch_vecs = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                vecs.extend(batch_vecs)
            except Exception as e:
                log(f"[11][error] Failed to embed batch {i}-{i+len(batch)}: {e}")
                # Fill with zeros for failed batch
                for _ in batch:
                    vecs.append(np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32))
            if (i // self.batch_size) % 5 == 0 or i + self.batch_size >= total:
                log(f"[11][progress] Embedded {min(i+self.batch_size, total)}/{total}")
        return np.asarray(vecs, dtype=np.float32)



class OllamaEmbedder:
    def __init__(self, model_name: str, host: str = "http://localhost:11434", timeout: int = 120):
        import requests  # lazy import
        self.requests = requests
        self.model = model_name
        self.host = host.rstrip("/")
        self.timeout = int(timeout)

    def _embed_one(self, text: str) -> List[float]:
        try:
            r = self.requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if emb is None:
                raise RuntimeError(f"Ollama returned no 'embedding' field: {data}")
            return emb
        except Exception as e:
            log(f"[11][error] Ollama embedding failed: {e}")
            return [0.0] * 768  # fallback dim, may need adjustment

    def encode(self, texts: List[str]) -> np.ndarray:
        # NOTE: Ollama API currently does not support batch embedding. If/when it does, refactor here.
        out: List[List[float]] = []
        for i, t in enumerate(texts, 1):
            if i % 50 == 0 or i == len(texts):
                log(f"[11][ollama] embedded {i}/{len(texts)}")
            out.append(self._embed_one(t))
        return np.asarray(out, dtype=np.float32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    env = get_env()

    ap = argparse.ArgumentParser(description="Step 11 — compute embeddings for sampled reviews", add_help=True)
    ap.add_argument("--provider", choices=["sbert", "ollama"], default=DEFAULTS["provider"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--in-csv", default=str(env.IN_SAMPLED))
    ap.add_argument("--out-path", default=str(env.OUT_PATH))
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--text-col", default=DEFAULTS["text_col"])
    ap.add_argument("--max-rows", type=int, default=DEFAULTS["max_rows"])
    ap.add_argument("--host", default=DEFAULTS["ollama_host"])
    ap.add_argument("--timeout", type=int, default=DEFAULTS["ollama_timeout"])
    args = ap.parse_args(argv)

    # Start
    log(f"[11] Start — provider={args.provider} model={args.model}")
    log(f"[11] Input : {args.in_csv}")
    log(f"[11] Output: {args.out_path}")

    df = read_reviews(Path(args.in_csv), text_col=args.text_col, max_rows=args.max_rows)
    n = len(df)
    log(f"[11] Loaded {n} reviews")

    texts = df["text"].tolist()

    if args.provider == "sbert":
        embedder = SbertEmbedder(args.model, batch_size=args.batch_size)
    else:
        embedder = OllamaEmbedder(args.model, host=args.host, timeout=args.timeout)

    t0 = time.time()
    vecs = embedder.encode(texts)
    dt = time.time() - t0

    if vecs.shape[0] != n:
        raise RuntimeError(f"[11][fatal] Embedding rows mismatch: expected {n}, got {vecs.shape[0]}")

    log(f"[11] Embeddings: shape={vecs.shape} in {dt:.1f}s (~{n/dt:.1f} items/s)")

    # Attach to DF (list[float] per row)
    df_out = df.copy()
    df_out["emb"] = [v.astype(float).tolist() for v in vecs]

    to_parquet_or_jsonl(df_out, Path(args.out_path))
    log("[11] Done.")


if __name__ == "__main__":
    main()
