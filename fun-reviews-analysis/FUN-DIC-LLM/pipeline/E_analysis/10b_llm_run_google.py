

'''
Step 10b: LLM Run (Google) - Sends prompts to Google AI Studio (Gemini) and manages structured response extraction.

• Sends prompts to Gemini and extracts/saves structured JSON responses
• Supports batching, error handling, and resumable runs
• Handles configuration, logging, and parallel dispatch

Inputs: Prompts JSONL file, configuration files.
Outputs: Responses JSONL file, debug logs.
'''
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from pipeline.Z_utils.common import find_project_root
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, confloat

from google import genai
from google.genai import types  # <-- for HttpOptions & GenerateContentConfig

try:
    from pipeline.Z_utils.common import ensure_dir, log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[10b {ts}] {msg}")

    def ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

@dataclass
class Env:
    ROOT: Path
    DATA: Path
    IN_CSV: Path
    OUT_DIR: Path
    OUT_RESPONSES: Path
    OUT_ERRORS: Path
    OUT_TABLES: Path
    OUT_PER_APP: Path
    OUT_PER_PAIR: Path
    META_PER_APP_DESC: Path

def get_env() -> Env:
    root = find_project_root()
    data = root / "data"
    in_csv = data / "reviews_sampled_all.csv"
    out_dir = data / "analysis" / "10"
    out_tables = out_dir / "tables"
    ensure_dir(out_tables)
    return Env(
        ROOT=root,
        DATA=data,
        IN_CSV=in_csv,
        OUT_DIR=out_dir,
        OUT_RESPONSES=out_dir / "responses_google.jsonl",
        OUT_ERRORS=out_dir / "errors_google.jsonl",
        OUT_TABLES=out_tables,
        OUT_PER_APP=out_tables / "per_app_llm_agg.csv",
        OUT_PER_PAIR=out_tables / "per_pair_deltas_llm.csv",
        META_PER_APP_DESC=data / "analysis" / "08" / "tables" / "per_app_descriptive_stats.csv",
    )

DEFAULTS = {
    "model": "gemini-2.5-flash",
    "workers": 4,
    "timeout": 60,
    "max_rows": None,
    "text_col": "review",
    "appid": None,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "rpm_soft_limit": 40,
}

def read_reviews(path: Path, text_col: str, max_rows: Optional[int], appid: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[10b][fatal] Input not found: {path.as_posix()}")
    df = pd.read_csv(path, encoding="utf-8")

    if "appid" not in df.columns:
        alts = [c for c in df.columns if str(c).lower() in ("app", "app_id", "appid")]
        if alts:
            df.rename(columns={alts[0]: "appid"}, inplace=True)
        else:
            raise SystemExit("[10b][fatal] Missing 'appid' column.")

    if "reviewId" not in df.columns:
        alts = [c for c in df.columns if str(c).lower() in ("recommendationid", "review_id", "id")]
        if alts:
            df["reviewId"] = df[alts[0]]
        else:
            df["reviewId"] = [f"idx:{i}" for i in range(len(df))]

    if text_col not in df.columns:
        for c in ("review", "text", "body", "content"):
            if c in df.columns:
                text_col = c
                break
        else:
            raise SystemExit(f"[10b][fatal] Text column '{text_col}' not found.")

    df["text"] = (
        df[text_col].astype(str).fillna("")
        .str.replace("\r\n", "\n")
        .str.replace("\r", "\n")
        .str.replace("\t", " ")
        .str.strip()
    )

    if appid is not None:
        df = df[df["appid"].astype(str) == str(appid)].copy()

    df = df[df["text"].map(lambda s: len(s) >= 3)].copy()

    df = df.sort_values(["appid", "reviewId"])
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    return df[["appid", "reviewId", "text"]].copy()

# --- simplified schema (scores+confidence only) ---
class FunScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Flow: float = Field(ge=0.0, le=1.0)
    Utility: float = Field(ge=0.0, le=1.0)
    Nostalgia: float = Field(ge=0.0, le=1.0)
    none_score: float = Field(alias="None", ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

SYSTEM_PROMPT = (
    "You are an expert rater for game reviews. Given a single review, assign four scores in [0,1]:\n"
    "- Flow: immersion/challenge-skill balance/control/absorption.\n"
    "- Utility: goals/progression/rewards/feedback/usefulness of mechanics.\n"
    "- Nostalgia: references to past games/eras, retro feelings, memories.\n"
    "- None: content unrelated to the above FUN dimensions.\n"
    "The four scores must sum to 1.0. Add a confidence in [0,1]. "
    "Return ONLY JSON, adhering to the provided schema."
)

def make_user_prompt(appid: str, review_id: str, text: str) -> str:
    header = f"APPID={appid} \n UNIT=review:{review_id} \n LEVEL=review\n"
    return header + "Review:\n" + text

def renormalize_from_model(model_obj: FunScores) -> Dict:
    d = model_obj.model_dump(by_alias=True)
    keys = ["Flow", "Utility", "Nostalgia", "None"]
    vals = [float(d.get(k, 0.0)) for k in keys]
    s = sum(vals)
    if s <= 0:
        d["Flow"] = d["Utility"] = d["Nostalgia"] = 0.0
        d["None"] = 1.0
    else:
        for k in keys:
            d[k] = float(d[k]) / s
    c = d.get("confidence", 0.0)
    try:
        c = float(c)
    except Exception:
        c = 0.0
    d["confidence"] = max(0.0, min(1.0, c))
    return d

def write_jsonl(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

class SoftRPMLimiter:
    def __init__(self, rpm: Optional[int]):
        self.rpm = max(0, int(rpm or 0))
        self.lock = threading.Lock()
        self.window = deque()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        now = time.time()
        with self.lock:
            while self.window and now - self.window[0] > 60.0:
                self.window.popleft()
            if len(self.window) >= self.rpm:
                wait = 60.0 - (now - self.window[0])
                if wait > 0:
                    time.sleep(min(wait, 1.0))
            self.window.append(time.time())

def score_reviews(
    df: pd.DataFrame,
    *,
    model_name: str,
    workers: int,
    timeout: int,
    rpm_soft_limit: int,
    temperature: float,
    top_p: float,
    top_k: int,
    out_jsonl: Path,
    out_errors: Path,
    skip_errors: bool = False,
) -> pd.DataFrame:
    # Set timeout via Client http_options (milliseconds)
    client = genai.Client(http_options=types.HttpOptions(timeout=int(timeout) * 1000))
    cfg = {
        "response_mime_type": "application/json",
        "response_schema": FunScores,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    limiter = SoftRPMLimiter(rpm_soft_limit)
    started = time.time()
    results: List[Dict] = []

    def _log_error(appid: str, rid: str, err: str, attempt: int, review_head: str) -> None:
        ensure_dir(out_errors.parent)
        payload = {
            "appid": appid,
            "unit_id": f"review:{rid}",
            "error": err,
            "attempt": attempt,
            "review_head": review_head,
        }
        with out_errors.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _call_one(row) -> Dict:
        appid = str(row["appid"])
        rid = str(row["reviewId"])
        text = row["text"]
        review_head = (text or "")[:180]
        user_content = make_user_prompt(appid, rid, text)
        contents = f"{SYSTEM_PROMPT}\n\n{user_content}"  # inline system text for compatibility

        delay = 1.0
        last_err = None

        for attempt in range(5):
            try:
                limiter.acquire()
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=cfg,
                )
                if not getattr(resp, "parsed", None):
                    raw = getattr(resp, "text", "") or ""
                    raise RuntimeError(f"Empty parsed content; raw text head: {raw[:120]!r}")

                parsed: FunScores = resp.parsed
                score_dict = renormalize_from_model(parsed)
                rec = {
                    "appid": appid,
                    "unit_id": f"review:{rid}",
                    "level": "review",
                    **score_dict,
                    "model": model_name,
                }
                write_jsonl(out_jsonl, rec)
                return {"ok": True, "rec": rec}

            except Exception as e:
                last_err = str(e)
                _log_error(appid, rid, last_err, attempt + 1, review_head)
                time.sleep(delay)
                delay = min(8.0, delay * 2.0)

        return {"ok": False, "appid": appid, "unit_id": f"review:{rid}", "error": last_err or "unknown"}

    n_total = len(df)
    log(f"[10b] Scoring {n_total} reviews with model={model_name}, workers={workers}")
    done = 0

    if workers > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(_call_one, r) for _, r in df.iterrows()]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                done += 1
                if done % 20 == 0 or done == n_total:
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_total - done) / rate / 60 if rate > 0 else float("inf")
                    log(f"[10b] progress {done}/{n_total} \n {rate:.2f} u/s \n ETA {eta:0.1f} min")
    else:
        for _, r in df.iterrows():
            res = _call_one(r)
            results.append(res)
            done += 1
            if done % 20 == 0 or done == n_total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate / 60 if rate > 0 else float("inf")
                log(f"[10b] progress {done}/{n_total} \n {rate:.2f} u/s \n ETA {eta:0.1f} min")

    oks = [r["rec"] for r in results if r.get("ok")]
    if not oks:
        msg = (
            "[10b][fatal] No successful responses. "
            f"Check {out_errors.as_posix()} for details (rate limits, schema issues, timeouts)."
        )
        if skip_errors:
            log(msg)
            return pd.DataFrame(columns=["appid", "unit_id", "level", "Flow", "Utility", "Nostalgia", "None", "confidence", "model"])
        raise SystemExit(msg)

    return pd.DataFrame(oks)

def per_app_aggregate(df_scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["Flow", "Utility", "Nostalgia", "None", "confidence"]
    if df_scores.empty:
        return pd.DataFrame(columns=["appid"] + cols)
    for c in cols:
        df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")
    agg = df_scores.groupby("appid", as_index=False)[cols].mean()
    for c in cols:
        agg[c] = agg[c] * 100.0
    return agg.sort_values("appid")

def per_pair_deltas(meta_csv: Path, per_app: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not meta_csv.exists() or per_app.empty:
        return None
    meta = pd.read_csv(meta_csv, encoding="utf-8")
    if not {"appid", "pairIndex", "type"}.issubset(meta.columns):
        return None
    meta["appid"] = meta["appid"].astype(str)
    per_app2 = per_app.copy()
    per_app2["appid"] = per_app2["appid"].astype(str)
    m = meta[["pairIndex", "type", "appid"]].merge(per_app2, on="appid", how="left")
    top = m[m["type"] == "top"].copy()
    bot = m[m["type"] == "bottom"].copy()
    k = top.merge(bot, on="pairIndex", suffixes=("_top", "_bottom"), how="inner")
    if k.empty:
        return None
    diffs = pd.DataFrame({
        "pairIndex": k["pairIndex"],
        "Flow_diff_top_minus_bottom": k["Flow_top"] - k["Flow_bottom"],
        "Utility_diff_top_minus_bottom": k["Utility_top"] - k["Utility_bottom"],
        "Nostalgia_diff_top_minus_bottom": k["Nostalgia_top"] - k["Nostalgia_bottom"],
        "None_diff_top_minus_bottom": k["None_top"] - k["None_bottom"],
        "confidence_diff_top_minus_bottom": k["confidence_top"] - k["confidence_bottom"],
        "appid_top": k["appid_top"],
        "appid_bottom": k["appid_bottom"],
    }).sort_values("pairIndex")
    return diffs

def _test_renorm():
    model = FunScores(Flow=0.3, Utility=0.3, Nostalgia=0.3, none_score=0.3, confidence=1.2)
    d = renormalize_from_model(model)
    s = sum(d[k] for k in ["Flow", "Utility", "Nostalgia", "None"])
    assert abs(s - 1.0) < 1e-9
    assert 0.0 <= d["confidence"] <= 1.0
    print("[10b][test] renormalize OK")

def main():
    env = get_env()

    ap = argparse.ArgumentParser(description="Step 10b — Gemini API per‑review scoring (structured JSON)")
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--timeout", type=int, default=DEFAULTS["timeout"])
    ap.add_argument("--rpm", type=int, default=DEFAULTS["rpm_soft_limit"])
    ap.add_argument("--max-rows", type=int, default=DEFAULTS["max_rows"])
    ap.add_argument("--text-col", default=DEFAULTS["text_col"])
    ap.add_argument("--appid", default=DEFAULTS["appid"], help="Optional: process only this appid")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts only; no API calls. Writes prompts_preview.jsonl")
    ap.add_argument("--head", type=int, default=None, help="Process only the first N rows after filtering/sorting")
    ap.add_argument("--skip-errors", action="store_true", help="Continue even if 0 successes; still write empty aggregates with warning")
    ap.add_argument("--test", action="store_true", help="Run minimal unit test and exit")
    args = ap.parse_args()

    if args.test:
        _test_renorm()
        return

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("[10b] No API key in environment; prompting securely.")
        key = getpass("Enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = key

    log(f"[10b] Start — model={args.model} workers={args.workers} rpm_soft={args.rpm}")
    log(f"[10b] Input : {env.IN_CSV.as_posix()}")
    log(f"[10b] Output: {env.OUT_RESPONSES.as_posix()} / {env.OUT_PER_APP.as_posix()}")

    df = read_reviews(env.IN_CSV, text_col=args.text_col, max_rows=args.max_rows, appid=args.appid)
    if args.head is not None:
        df = df.head(int(args.head)).copy()
    if df.empty:
        raise SystemExit("[10b][fatal] No reviews to process after filtering.")

    if args.dry_run:
        preview_path = env.OUT_DIR / "prompts_preview.jsonl"
        ensure_dir(preview_path.parent)
        with preview_path.open("w", encoding="utf-8") as fh:
            for _, r in df.iterrows():
                appid = str(r["appid"])
                rid = str(r["reviewId"])
                contents = f"{SYSTEM_PROMPT}\n\n{make_user_prompt(appid, rid, r['text'])}"
                fh.write(json.dumps({
                    "appid": appid,
                    "unit_id": f"review:{rid}",
                    "level": "review",
                    "model": args.model,
                    "contents": contents[:1200]
                }, ensure_ascii=False) + "\n")
        log(f"[10b] Dry-run complete. Wrote: {preview_path.as_posix()}")
        return

    t0 = time.time()
    df_scores = score_reviews(
        df,
        model_name=args.model,
        workers=int(args.workers),
        timeout=int(args.timeout),
        rpm_soft_limit=int(args.rpm),
        temperature=DEFAULTS["temperature"],
        top_p=DEFAULTS["top_p"],
        top_k=DEFAULTS["top_k"],
        out_jsonl=env.OUT_RESPONSES,
        out_errors=env.OUT_ERRORS,
        skip_errors=bool(args.skip_errors),
    )
    dt = time.time() - t0
    log(f"[10b] Collected {len(df_scores)} responses in {dt:.1f}s")

    per_app = per_app_aggregate(df_scores)
    ensure_dir(env.OUT_TABLES)
    per_app.to_csv(env.OUT_PER_APP, index=False, encoding="utf-8")
    log(f"[10b] Wrote per-app aggregates: {env.OUT_PER_APP.as_posix()}")

    deltas = per_pair_deltas(env.META_PER_APP_DESC, per_app)
    if deltas is not None:
        deltas.to_csv(env.OUT_PER_PAIR, index=False, encoding="utf-8")
        log(f"[10b] Wrote per-pair deltas: {env.OUT_PER_PAIR.as_posix()}")

    log("[10b] Done.")

if __name__ == "__main__":
    main()