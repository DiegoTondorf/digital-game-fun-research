# 01_steamspy_fetch.py
# coding: utf-8

'''
Step 01: SteamSpy Fetch - Fetches SteamSpy "all" pages and writes multiple output formats.

• Downloads and processes SteamSpy data pages
• Writes raw JSON pages, CSV, and NDJSON outputs
• Extracts key columns for downstream analysis

Inputs: SteamSpy API endpoints, configuration files.
Outputs: raw/steamspy/all/page_####.json, steamspy_all_games.csv, raw/steamspy/all_games.jsonl
'''

import csv
import json
import os
import time
from pathlib import Path

from Z_utils.common import log, ensure_dir, find_project_root



ROOT = find_project_root()
DATA = ROOT / "data"
from typing import Dict, Any, List, Optional

import requests
from requests.adapters import HTTPAdapter

### Pipeline Step 01: Fetch SteamSpy data
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
	# Outputs
	"RAW_DIR": str(get_env("STEAMSPY_RAW_DIR", DATA / "raw/steamspy/all/")),
	"RAW_JSONL": str(get_env("STEAMSPY_RAW_JSONL", DATA / "raw/steamspy/all_games.jsonl")),
	"OUT_CSV": str(get_env("STEAMSPY_OUT_CSV", DATA / "steamspy_all_games.csv")),

	# Pagination & checkpoint
	"START_PAGE": get_env("STEAMSPY_START_PAGE", 0),
	"MAX_PAGES": get_env("STEAMSPY_MAX_PAGES", None),
	"USE_CHECKPOINT": get_env("STEAMSPY_USE_CHECKPOINT", True),
	"CHECKPOINT": str(get_env("STEAMSPY_CHECKPOINT", DATA / "checkpoints/steamspy_all.page")),

	# Overwrite outputs at start?
	"RESET_OUTPUTS": get_env("STEAMSPY_RESET_OUTPUTS", False),

	# Networking
	"TIMEOUT": get_env("STEAMSPY_TIMEOUT", 30.0),
	"HTTP_POOL": get_env("STEAMSPY_HTTP_POOL", 8),
	"USER_AGENT": get_env("STEAMSPY_USER_AGENT",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
		"(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
	),

	# Throttling
	"PAGE_DELAY_SEC": get_env("STEAMSPY_PAGE_DELAY_SEC", 0.0),

	# Formatting
	"POS_RATE_DECIMALS": get_env("STEAMSPY_POS_RATE_DECIMALS", 6),
}

STEAMSPY_URL = "https://steamspy.com/api.php"

CSV_FIELDS = [
	"appid",
	"name",
	"posReview",
	"negReview",
	"totReview",
	"posRate",
]




def reset_outputs_if_needed():
	if CONFIG["RESET_OUTPUTS"]:
		for p in [CONFIG["OUT_CSV"], CONFIG["RAW_JSONL"]]:
			path = Path(p)
			if path.exists():
				path.unlink()
		# Do not delete RAW_DIR pages automatically (keep history)
		# Remove checkpoint so a fresh run starts from CONFIG["START_PAGE"]
		if CONFIG["USE_CHECKPOINT"]:
			cp = Path(CONFIG["CHECKPOINT"])
			if cp.exists():
				cp.unlink()


def read_checkpoint(path: str) -> Optional[int]:
	try:
		if Path(path).exists():
			return int(Path(path).read_text(encoding="utf-8").strip() or "0")
	except Exception:
		pass
	return None


def write_checkpoint(path: str, page: int):
	ensure_dir(Path(path).parent)
	Path(path).write_text(str(page), encoding="utf-8")


def build_session(pool_size: int) -> requests.Session:
	s = requests.Session()
	s.headers.update({
		"User-Agent": CONFIG["USER_AGENT"],
		"Accept": "application/json, text/plain, */*",
		"Accept-Language": "en-US,en;q=0.9",
		"Connection": "keep-alive",
	})
	adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
	s.mount("https://", adapter)
	s.mount("http://", adapter)
	return s


def fetch_page(session: requests.Session, page: int) -> Dict[str, Any] | None:
	try:
		r = session.get(
			STEAMSPY_URL,
			params={"request": "all", "page": page},
			timeout=CONFIG["TIMEOUT"],
		)
		if r.status_code != 200:
			return None
		data = r.json()
		return data if isinstance(data, dict) else None
	except Exception:
		return None


def normalize_item(v: Dict[str, Any]) -> Dict[str, Any] | None:
	# Expected SteamSpy fields in 'all' pages include: appid, name, positive, negative (and others)
	# We only take what we need for early steps.
	try:
		appid = int(v.get("appid"))
	except Exception:
		return None
	name = (v.get("name") or "").strip()

	def to_int(x) -> int:
		try:
			return int(x)
		except Exception:
			return 0

	pos = to_int(v.get("positive", 0))
	neg = to_int(v.get("negative", 0))
	tot = pos + neg
	pos_rate = round((pos / tot), CONFIG["POS_RATE_DECIMALS"]) if tot > 0 else 0.0

	return {
		"appid": appid,
		"name": name,
		"posReview": pos,
		"negReview": neg,
		"totReview": tot,
		"posRate": pos_rate,
	}


def append_csv(path: str, rows: List[Dict[str, Any]]):
	write_header = not Path(path).exists() or Path(path).stat().st_size == 0
	ensure_dir(Path(path).parent)
	with open(path, "a", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
		if write_header:
			w.writeheader()
		for r in rows:
			out = dict(r)
			out["posRate"] = f"{out['posRate']:.{CONFIG['POS_RATE_DECIMALS']}f}"
			w.writerow(out)


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
	ensure_dir(Path(path).parent)
	with open(path, "a", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_raw_page(raw_dir: str, page: int, payload: Dict[str, Any]):
	ensure_dir(raw_dir)
	raw_path = Path(raw_dir) / f"page_{page:04d}.json"
	raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
	import pandas as pd
	reset_outputs_if_needed()

	# Prepare outputs
	ensure_dir(Path(CONFIG["RAW_DIR"]))
	ensure_dir(Path(CONFIG["RAW_JSONL"]).parent)
	ensure_dir(Path(CONFIG["OUT_CSV"]).parent)

	# Determine start page
	page = CONFIG["START_PAGE"]
	if CONFIG["USE_CHECKPOINT"]:
		last = read_checkpoint(CONFIG["CHECKPOINT"])
		if last is not None:
			page = max(page, last + 1)

	session = build_session(CONFIG["HTTP_POOL"])
	pages = 0
	rows_total = 0

	# Use chunked lists for large data
	chunk_size = 10000
	all_rows = []

	while True:
		if CONFIG["MAX_PAGES"] is not None and pages >= CONFIG["MAX_PAGES"]:
			log(f"[stop] MAX_PAGES={CONFIG['MAX_PAGES']} reached.")
			break

		log(f"[fetch] page {page} ...")
		data = fetch_page(session, page)
		if not data:
			log("[stop] empty/error page. stopping.")
			break

		# Save raw
		save_raw_page(CONFIG["RAW_DIR"], page, data)

		# Normalize
		normalized = []
		for _, v in data.items():
			row = normalize_item(v or {})
			if row is not None:
				normalized.append(row)

		if not normalized:
			log("[stop] no rows after normalization. stopping.")
			break

		# Collect for chunked pandas write
		all_rows.extend(normalized)
		if len(all_rows) >= chunk_size:
			df = pd.DataFrame(all_rows)
			# Vectorized write to CSV (append, no header if file exists)
			write_header = not Path(CONFIG["OUT_CSV"]).exists() or Path(CONFIG["OUT_CSV"].stat().st_size == 0)
			df.to_csv(CONFIG["OUT_CSV"], mode="a", header=write_header, index=False, float_format=f"%.{CONFIG['POS_RATE_DECIMALS']}f")
			# Write jsonl chunk
			with open(CONFIG["RAW_JSONL"], "a", encoding="utf-8") as f:
				for r in all_rows:
					f.write(json.dumps(r, ensure_ascii=False) + "\n")
			all_rows = []

		pages += 1
		rows_total += len(normalized)
		log(f"[ok] page {page} -> {len(normalized)} rows (total={rows_total})")

		# Checkpoint
		if CONFIG["USE_CHECKPOINT"]:
			write_checkpoint(CONFIG["CHECKPOINT"], page)

		# SteamSpy 'all' full page has 1000 entries; a short page usually means we're done.
		if len(normalized) < 1000:
			log("[done] last partial page (< 1000). stopping.")
			break

		page += 1
		if CONFIG["PAGE_DELAY_SEC"] and CONFIG["PAGE_DELAY_SEC"] > 0:
			time.sleep(CONFIG["PAGE_DELAY_SEC"])

	# Write any remaining rows
	if all_rows:
		df = pd.DataFrame(all_rows)
		write_header = not Path(CONFIG["OUT_CSV"]).exists() or Path(CONFIG["OUT_CSV"].stat().st_size == 0)
		df.to_csv(CONFIG["OUT_CSV"], mode="a", header=write_header, index=False, float_format=f"%.{CONFIG['POS_RATE_DECIMALS']}f")
		with open(CONFIG["RAW_JSONL"], "a", encoding="utf-8") as f:
			for r in all_rows:
				f.write(json.dumps(r, ensure_ascii=False) + "\n")

	log("\n=== Summary ===")
	log(f"Pages fetched : {pages}")
	log(f"Rows saved    : {rows_total}")
	log(f"Raw pages dir : {CONFIG['RAW_DIR']}")
	log(f"NDJSON path   : {CONFIG['RAW_JSONL']}")
	log(f"CSV path      : {CONFIG['OUT_CSV']}")


if __name__ == "__main__":
	main()
# ...existing code from gathering/01_steamspy_fetch.py...
