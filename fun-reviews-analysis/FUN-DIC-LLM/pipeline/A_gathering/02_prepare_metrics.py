# 02_prepare_metrics.py
# coding: utf-8

'''
Step 02: Prepare Metrics - Computes metrics and ranks from SteamSpy CSV data.

• Calculates review counts, rates, popularity, and acceptance scores
• Sorts and ranks apps by acceptanceScore
• Outputs enriched and sorted app data for downstream steps

Inputs: steamspy_all_games.csv (from Step 01)
Outputs: steamspy_scored_sorted.csv (sorted by acceptanceScore DESC, with metrics columns)
'''

import math
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from Z_utils.common import log, ensure_dir, find_project_root


ROOT = find_project_root()
DATA = ROOT / "data"

### Pipeline Step 02: Prepare metrics
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
	"INPUT_CSV": str(get_env("METRICS_INPUT_CSV", DATA / "steamspy_all_games.csv")),
	"OUTPUT_SORTED": str(get_env("METRICS_OUTPUT_SORTED", DATA / "steamspy_scored_sorted.csv")),
	"ROUND_DECIMALS": get_env("METRICS_ROUND_DECIMALS", 6),
}

def ensure_parent_dir(path: str | Path):
	ensure_dir(Path(path).parent)

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

def compute_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
	appid = str(row.get("appid", "")).strip()
	name = str(row.get("name", "")).strip()

	pos = to_int(row.get("posReview", 0), 0)
	neg = to_int(row.get("negReview", 0), 0)
	tot = pos + neg

	if tot <= 0:
		pos_rate = 0.0
		popularity = 0.0
		acc = 0.0
	else:
		pos_rate = pos / tot
		popularity = math.log(tot)
		acc = pos_rate * popularity

	return {
		"appid": appid,
		"name": name,
		"posReview": pos,
		"negReview": neg,
		"totReview": tot,
		"posRate": pos_rate,
		"popularity": popularity,
		"acceptanceScore": acc,
	}

def read_rows(path: str) -> List[Dict[str, Any]]:
	# Use pandas for chunked reading and vectorized ops
	df = pd.read_csv(path, encoding="utf-8-sig")
	return df.to_dict(orient="records")

def write_rows(path: str, rows: List[Dict[str, Any]], round_decimals: int = 6):
	ensure_parent_dir(path)
	cols = [
		"appid", "name",
		"posReview", "negReview", "totReview",
		"posRate", "popularity", "acceptanceScore",
		"rankAcceptance",
	]
	df = pd.DataFrame(rows)
	# Round columns
	for col in ["posRate", "popularity", "acceptanceScore"]:
		if col in df:
			df[col] = df[col].round(round_decimals)
	df.to_csv(path, columns=cols, index=False, encoding="utf-8")

def assert_desc_by_acceptance(rows: List[Dict[str, Any]]):
	# Raise with a helpful message if not strictly nonincreasing
	for i in range(1, len(rows)):
		if rows[i]["acceptanceScore"] > rows[i-1]["acceptanceScore"]:
			raise AssertionError(
				f"Not sorted by acceptanceScore DESC at row {i}: "
				f"{rows[i-1]['acceptanceScore']} (prev) < {rows[i]['acceptanceScore']} (curr)"
			)

def main():
	inp = CONFIG["INPUT_CSV"]
	out = CONFIG["OUTPUT_SORTED"]
	rnd = CONFIG["ROUND_DECIMALS"]

	if not Path(inp).exists():
		raise FileNotFoundError(f"Input not found: {inp}")

	raw = read_rows(inp)
	log(f"[INFO] Read {len(raw)} rows from {inp}")

	# Vectorized metrics computation using pandas
	df = pd.DataFrame(raw)
	df["posReview"] = df["posReview"].apply(to_int)
	df["negReview"] = df["negReview"].apply(to_int)
	df["totReview"] = df["posReview"] + df["negReview"]
	df["posRate"] = df.apply(lambda r: r["posReview"] / r["totReview"] if r["totReview"] > 0 else 0.0, axis=1)
	df["popularity"] = df["totReview"].apply(lambda x: math.log(x) if x > 0 else 0.0)
	df["acceptanceScore"] = df["posRate"] * df["popularity"]

	# Sort by acceptanceScore DESC
	df = df.sort_values(by="acceptanceScore", ascending=False)
	df["rankAcceptance"] = range(1, len(df) + 1)

	# Sanity check: strictly DESC (nonincreasing)
	rows = df.to_dict(orient="records")
	assert_desc_by_acceptance(rows)

	# Quick debug print of top 5 heads (ASCII only)
	head = rows[:5]
	log("[DEBUG] Top 5 by acceptanceScore (acceptanceScore, posRate, popularity, totReview):")
	for h in head:
		log(f"        {h['acceptanceScore']:.6f}, {h['posRate']:.6f}, "
			  f"{h['popularity']:.6f}, {h['totReview']}")

	write_rows(out, rows, rnd)
	log(f"[OK] Generated: {out} (n={len(rows)}) sorted by acceptanceScore DESC.")

if __name__ == "__main__":
	main()
