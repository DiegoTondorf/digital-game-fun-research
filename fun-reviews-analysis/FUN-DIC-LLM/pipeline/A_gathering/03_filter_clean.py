# 03_filter_clean.py
# coding: utf-8

'''
Step 03: Filter & Clean - Filters, deduplicates, and ranks apps for downstream analysis.

• Filters and deduplicates app data
• Computes quartiles and local ranks
• Writes filtered and ranked app information

Inputs: steamspy_scored_sorted.csv (from Step 02)
Outputs: steamspy_filtered.csv (with ranking columns)
'''


import os
import csv
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from Z_utils.common import log, ensure_dir, find_project_root


ROOT = find_project_root()
DATA = ROOT / "data"

### Pipeline Step 03: Filter & Clean
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
	"INPUT_SORTED": str(get_env("FILTER_INPUT_SORTED", DATA / "steamspy_scored_sorted.csv")),
	"OUTPUT_FILTERED": str(get_env("FILTER_OUTPUT_FILTERED", DATA / "steamspy_filtered.csv")),
	"MIN_TOTAL_REVIEWS": get_env("FILTER_MIN_TOTAL_REVIEWS", 50),
	"QUARTILES_MODE": get_env("FILTER_QUARTILES_MODE", "keep_mid"),
	"ROUND_DECIMALS": get_env("FILTER_ROUND_DECIMALS", 6),
}

COLS_OUT = [
	"appid", "name",
	"posReview", "negReview", "totReview",
	"posRate", "popularity", "acceptanceScore",
	"rankAcceptance",     # global rank from Step 02 (or computed here if missing)
	"rankFiltered",       # rank within the filtered output
]

# ======================
# Helpers
# ======================
def to_int(x, default: int = 0) -> int:
	try:
		return int(float(x))
	except Exception:
		return default

def to_float(x, default: float = 0.0) -> float:
	try:
		return float(x)
	except Exception:
		return default

def read_all_rows(path: str) -> List[Dict[str, Any]]:
	"""Read all rows, coerce numeric fields, and sort DESC by acceptanceScore."""
	with open(path, newline="", encoding="utf-8-sig") as f:
		rows = list(csv.DictReader(f))

	out: List[Dict[str, Any]] = []
	for r in rows:
		out.append({
			"appid": str(r.get("appid", "")).strip(),
			"name": str(r.get("name", "")).strip(),
			"posReview": to_int(r.get("posReview", 0)),
			"negReview": to_int(r.get("negReview", 0)),
			"totReview": to_int(r.get("totReview", 0)),
			"posRate": to_float(r.get("posRate", 0.0)),
			"popularity": to_float(r.get("popularity", 0.0)),
			"acceptanceScore": to_float(r.get("acceptanceScore", 0.0)),
			# carry rankAcceptance if present; may be empty
			"rankAcceptance": to_int(r.get("rankAcceptance", 0), 0),
		})

	# Safety: enforce DESC by acceptanceScore
	out.sort(key=lambda r: r["acceptanceScore"], reverse=True)
	return out

def write_rows(path: str, rows: List[Dict[str, Any]], round_decimals: int = 6):
	ensure_dir(Path(path).parent)
	df = pd.DataFrame(rows)
	# Round columns
	for col in ["posRate", "popularity", "acceptanceScore"]:
		if col in df:
			df[col] = df[col].round(round_decimals)
	for col in ["rankAcceptance", "rankFiltered"]:
		if col in df:
			df[col] = df[col].astype(int)
	df.to_csv(path, columns=COLS_OUT, index=False, encoding="utf-8")

def quartile_indices(n: int):
	q1 = int(n * 0.25); q2 = int(n * 0.50); q3 = int(n * 0.75)
	return q1, q2, q3

def safe_score(rows: List[Dict[str, Any]], idx: int):
	return rows[idx]["acceptanceScore"] if 0 <= idx < len(rows) else None

# ======================
# Main
# ======================
def main():
	cfg = CONFIG
	inp = cfg["INPUT_SORTED"]
	out = cfg["OUTPUT_FILTERED"]
	nmin = cfg["MIN_TOTAL_REVIEWS"]
	mode = cfg["QUARTILES_MODE"]
	rnd = cfg["ROUND_DECIMALS"]

	if not Path(inp).exists():
		raise FileNotFoundError(f"Input not found: {inp}")

	# 0) Read ALL rows (DESC by acceptanceScore)
	df = pd.read_csv(inp, encoding="utf-8-sig")
	df = df.sort_values(by="acceptanceScore", ascending=False)
	n_all = len(df)
	log(f"[INFO] Read {n_all} rows from {inp} (DESC by acceptanceScore).")

	# 1) Ensure we have a global rankAcceptance; if missing/zero, compute now
	missing_rank = (df["rankAcceptance"] <= 0).any() if "rankAcceptance" in df else True
	if missing_rank:
		df["rankAcceptance"] = range(1, n_all + 1)
		log("[INFO] rankAcceptance was missing for some/all rows; computed global ranks based on acceptanceScore DESC.")
	else:
		log("[INFO] rankAcceptance found in input; preserving Step 02 global ranks.")

	# 2) Apply minimum total reviews
	df = df[df["totReview"] >= nmin]
	log(f"[INFO] After totReview >= {nmin}: {len(df)} rows.")

	# 3) De-duplicate by appid (keep first = highest acceptanceScore)
	df = df.drop_duplicates(subset=["appid"], keep="first")
	log(f"[INFO] After dedup by appid: {len(df)} rows.")

	# 4) Quartile slicing on acceptanceScore
	n = len(df)
	rows = df.to_dict(orient="records")
	if n >= 4 and mode != "keep_all":
		q1, q2, q3 = quartile_indices(n)
		borders = {
			"top[0]": safe_score(rows, 0),
			"q1-1": safe_score(rows, q1 - 1),
			"q1": safe_score(rows, q1),
			"q2-1": safe_score(rows, q2 - 1),
			"q2": safe_score(rows, q2),
			"q3-1": safe_score(rows, q3 - 1),
			"q3": safe_score(rows, q3),
			"last": safe_score(rows, n - 1),
		}
		log(f"[INFO] n={n} q1={q1}, q2={q2}, q3={q3}")
		log(f"[BORDERS] acceptanceScore: {borders}")

		if mode == "keep_mid":
			kept = rows[q1:q3]  # middle 50%
			lo = kept[-1]["acceptanceScore"] if kept else None
			hi = kept[0]["acceptanceScore"] if kept else None
			log(f"[INFO] keep_mid -> rows[{q1}:{q3}] (~ middle 50%). "
				  f"acceptanceScore range ~ {lo} .. {hi}")
			rows = kept

		elif mode == "keep_extremes":
			kept = rows[0:q1] + rows[q3:n]
			top = rows[0]["acceptanceScore"] if rows else None
			bot = rows[-1]["acceptanceScore"] if rows else None
			log(f"[INFO] keep_extremes -> rows[0:{q1}] U rows[{q3}:{n}] (Q4 + Q1). "
				  f"Top acceptanceScore ~ {top}; Bottom acceptanceScore ~ {bot}")
			rows = kept

		else:
			log(f"[WARN] Unknown QUARTILES_MODE='{mode}', keeping all.")
	else:
		if n < 4:
			log(f"[INFO] Quartiles not applied (n<{4}).")
		else:
			log(f"[INFO] Quartiles disabled (mode='{mode}').")

	# 5) Assign rankFiltered (1..n) in the current order (DESC by acceptanceScore)
	for idx, r in enumerate(rows, start=1):
		r["rankFiltered"] = idx

	# 6) Write output
	write_rows(out, rows, rnd)
	log(f"[OK] Generated: {out} (n={len(rows)}) mode={mode} min_total={nmin} "
		  f"with rankAcceptance (global) and rankFiltered (local).")

if __name__ == "__main__":
	main()
