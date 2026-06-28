# 05_fetch_reviews.py
# coding: utf-8

'''
Step 05: Fetch Reviews - Fetches English reviews for selected apps and writes multiple output formats.

• Downloads and processes review pages for each app
• Writes raw JSON pages, per-app CSVs, master CSV, and checkpoints
• Supports reproducible paths and helper utilities

Inputs: App list, API endpoints, configuration files.
Outputs: raw/reviews/{appid}/page_#####.json, reviews_per_app/{appid}.csv, reviews_all.csv, checkpoints/reviews/{appid}.ckpt.json
'''


import json
import os
import sys
import time
import random
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from pipeline.Z_utils.common import asc, log, ensure_dir, find_project_root


ROOT = find_project_root()
DATA = ROOT / "data"
import requests

### Pipeline Step 05: Fetch reviews
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
	# Inputs / Outputs
	"INPUT_PAIRS_CSV": str(get_env("REVIEWS_INPUT_PAIRS_CSV", DATA / "steamspy_top_bottom_pairs.csv")),

	"RAW_PER_APP_DIR": str(get_env("REVIEWS_RAW_PER_APP_DIR", DATA / "raw/reviews/")),
	"OUT_PER_APP_DIR": str(get_env("REVIEWS_OUT_PER_APP_DIR", DATA / "reviews_per_app/")),
	"OUT_MASTER_CSV":  str(get_env("REVIEWS_OUT_MASTER_CSV", DATA / "reviews_all.csv")),
	"CKPT_DIR":        str(get_env("REVIEWS_CKPT_DIR", DATA / "checkpoints/reviews/")),

	# Master CSV behavior
	"RESET_MASTER_OUTPUTS": get_env("REVIEWS_RESET_MASTER_OUTPUTS", False),

	# API request parameters (per Steam docs)
	"FILTER": get_env("REVIEWS_FILTER", "updated"),
	"LANGUAGE": get_env("REVIEWS_LANGUAGE", "english"),
	"REVIEW_TYPE": get_env("REVIEWS_REVIEW_TYPE", "all"),
	"PURCHASE_TYPE": get_env("REVIEWS_PURCHASE_TYPE", "all"),
	"FILTER_OFFTOPIC_ACTIVITY": get_env("REVIEWS_FILTER_OFFTOPIC_ACTIVITY", 1),
	"NUM_PER_PAGE": get_env("REVIEWS_NUM_PER_PAGE", 100),
	"REQUEST_TIMEOUT": get_env("REVIEWS_REQUEST_TIMEOUT", 15.0),

	# Limits / pacing
	"MAX_REVIEWS_PER_APP": get_env("REVIEWS_MAX_REVIEWS_PER_APP", None),
	"MAX_PAGES_PER_APP": get_env("REVIEWS_MAX_PAGES_PER_APP", None),
	"SLEEP_BETWEEN_PAGES_SEC": get_env("REVIEWS_SLEEP_BETWEEN_PAGES_SEC", 0.0),

	# Retry/backoff
	"MAX_ATTEMPTS": get_env("REVIEWS_MAX_ATTEMPTS", 6),
	"INITIAL_DELAY": get_env("REVIEWS_INITIAL_DELAY", 1.0),
	"MAX_DELAY": get_env("REVIEWS_MAX_DELAY", 20.0),
	"JITTER_FRAC": get_env("REVIEWS_JITTER_FRAC", 0.25),

	# Logging
	"USER_AGENT": get_env("REVIEWS_USER_AGENT", "Mozilla/5.0 (SteamReviews/1.0; +study)"),
	"VERBOSE": get_env("REVIEWS_VERBOSE", True),
	"DEBUG_TRACE_APPIDS": get_env("REVIEWS_DEBUG_TRACE_APPIDS", []),
	# Debug/test options
	"DEBUG_FETCH_ONE": True,  # Set to True to fetch only one review for testing
	"DEBUG_MAX_PAGES": 1,
	"DEBUG_MAX_REVIEWS": 1,
}

ENDPOINT = "https://store.steampowered.com/appreviews/{appid}?json=1"

CSV_FIELDS = [
	# From selection (Step 04)
	"pairIndex", "type", "appid", "name",
	"rankAcceptance", "rankFiltered",
	# For comparison
	"acceptanceScore",
	# Review fields
	"reviewId", "language", "review",
	"isPositive", "voted_up",
	"timestamp_created", "timestamp_updated",
	"authorSteamId", "playtime_forever", "playtime_last_two_weeks",
	"received_for_free", "written_during_early_access", "steam_purchase",
	"comment_count", "votes_up", "votes_funny", "weighted_vote_score",
	# Convenience
	"charLen",
	# From query_summary (for the configured language)
	"totalEnglishReviews",
]

# ======================
# ASCII-safe logging
# ======================

# ======================
# IO helpers
# ======================

def load_pairs(path: str) -> List[Dict[str, Any]]:
	df = pd.read_csv(path, encoding="utf-8-sig")
	df = df.drop_duplicates(subset=["appid"], keep="first")
	df = df.sort_values(by=["pairIndex", "type", "appid"])
	return df.to_dict(orient="records")

def write_master_header(path: str):
	ensure_dir(Path(path).parent)
	df = pd.DataFrame(columns=CSV_FIELDS)
	df.to_csv(path, index=False, encoding="utf-8")

def append_master_rows(path: str, rows: List[Dict[str, Any]]):
	ensure_dir(Path(path).parent)
	df = pd.DataFrame(rows)
	header = not Path(path).exists() or Path(path).stat().st_size == 0
	df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")

def append_app_rows(app_csv: str, rows: List[Dict[str, Any]]):
	ensure_dir(Path(app_csv).parent)
	df = pd.DataFrame(rows)
	header = not Path(app_csv).exists() or Path(app_csv).stat().st_size == 0
	df.to_csv(app_csv, mode="a", header=header, index=False, encoding="utf-8")

# ======================
# HTTP client with backoff
# ======================
class Http:
	def __init__(self):
		self.sess = requests.Session()
		self.sess.headers.update({"User-Agent": CONFIG["USER_AGENT"]})

	def _wait(self, attempt: int) -> float:
		base = CONFIG["INITIAL_DELAY"] * (2 ** (attempt - 1))
		base = min(base, CONFIG["MAX_DELAY"])
		jitter = base * CONFIG["JITTER_FRAC"]
		return max(0.0, base + random.uniform(-jitter, jitter))

	def get_json(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		max_attempts = CONFIG["MAX_ATTEMPTS"]
		for attempt in range(1, max_attempts + 1):
			try:
				r = self.sess.get(url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
				if r.status_code == 200:
					return r.json()
				if r.status_code in (429, 500, 502, 503, 504):
					wait_s = self._wait(attempt)
					log(f"[RETRY] {r.status_code} GET {url} attempt {attempt}/{max_attempts} wait={wait_s:.2f}s")
					time.sleep(wait_s)
					continue
				log(f"[ERROR] HTTP {r.status_code} GET {url} not retrying")
				return None
			except requests.RequestException as ex:
				wait_s = self._wait(attempt)
				log(f"[RETRY] Exception GET {url}: {asc(str(ex))} attempt {attempt}/{max_attempts} wait={wait_s:.2f}s")
				time.sleep(wait_s)
			except Exception as ex:
				log(f"[ERROR] Unexpected exception: {asc(str(ex))}")
				return None
		log("[ERROR] Exceeded max attempts")
		return None

# ======================
# Checkpoints & paths
# ======================
def ckpt_path_for(appid: str) -> Path:
	return Path(CONFIG["CKPT_DIR"]) / f"{appid}.ckpt.json"

def load_ckpt(appid: str) -> Dict[str, Any]:
	p = ckpt_path_for(appid)
	if not p.exists():
		return {"cursor": "*", "pages": 0, "reviews": 0}
	try:
		return json.loads(p.read_text(encoding="utf-8"))
	except Exception:
		return {"cursor": "*", "pages": 0, "reviews": 0}

def save_ckpt(appid: str, ck: Dict[str, Any]):
	ensure_dir(Path(CONFIG["CKPT_DIR"]))
	ckpt_path_for(appid).write_text(json.dumps(ck, ensure_ascii=False, indent=2), encoding="utf-8")

def raw_page_path(appid: str, page_idx: int) -> Path:
	return Path(CONFIG["RAW_PER_APP_DIR"]) / str(appid) / f"page_{page_idx:05d}.json"

# ======================
# Normalization
# ======================
def normalize_reviews(app_meta: Dict[str, Any], reviews: List[Dict[str, Any]], total_en: int) -> List[Dict[str, Any]]:
	rows = []
	for rv in reviews:
		rid = str(rv.get("recommendationid", "")).strip()
		txt = rv.get("review", "") or ""
		voted_up = bool(rv.get("voted_up", False))
		lang = str(rv.get("language", "")).strip()

		author = rv.get("author", {}) or {}
		rows.append({
			# selection context
			"pairIndex": app_meta["pairIndex"],
			"type": app_meta["type"],
			"appid": app_meta["appid"],
			"name": app_meta["name"],
			"rankAcceptance": app_meta["rankAcceptance"],
			"rankFiltered": app_meta["rankFiltered"],
			"acceptanceScore": app_meta["acceptanceScore"],

			# review payload
			"reviewId": rid,
			"language": lang,
			"review": txt,
			"isPositive": 1 if voted_up else 0,
			"voted_up": voted_up,
			"timestamp_created": int(rv.get("timestamp_created", 0) or 0),
			"timestamp_updated": int(rv.get("timestamp_updated", 0) or 0),
			"authorSteamId": str(author.get("steamid", "")).strip(),
			"playtime_forever": int(author.get("playtime_forever", 0) or 0),
			"playtime_last_two_weeks": int(author.get("playtime_last_two_weeks", 0) or 0),
			"received_for_free": bool(rv.get("received_for_free", False)),
			"written_during_early_access": bool(rv.get("written_during_early_access", False)),
			"steam_purchase": bool(rv.get("steam_purchase", False)),
			"comment_count": int(rv.get("comment_count", 0) or 0),
			"votes_up": int(rv.get("votes_up", 0) or 0),
			"votes_funny": int(rv.get("votes_funny", 0) or 0),
			"weighted_vote_score": str(rv.get("weighted_vote_score", "")),
			"charLen": len(txt),

			# constant for this app, given the configured params (language, filters)
			"totalEnglishReviews": int(total_en or 0),
		})
	return rows

# ======================
# Fetch per app
# ======================
def fetch_for_app(http: Http, app_meta: Dict[str, Any]):
	appid = app_meta["appid"]
	log(f"[APP] {appid} `{asc(app_meta['name'])}` pair={app_meta['pairIndex']} type={app_meta['type']}")
	ensure_dir(CONFIG["OUT_PER_APP_DIR"])
	ensure_dir(CONFIG["CKPT_DIR"])
	ensure_dir(Path(CONFIG["RAW_PER_APP_DIR"]) / str(appid))
	app_csv = str(Path(CONFIG["OUT_PER_APP_DIR"]) / f"{appid}.csv")
	ck = load_ckpt(appid)
	cursor = ck.get("cursor", "*") or "*"
	pages = int(ck.get("pages", 0) or 0)
	total_written = int(ck.get("reviews", 0) or 0)
	# DEBUG: Print initial checkpoint and raw file count
	raw_dir = Path(CONFIG["RAW_PER_APP_DIR"]) / str(appid)
	raw_files = list(raw_dir.glob("page_*.json")) if raw_dir.exists() else []
	log(f"[DEBUG] app={appid} START: checkpoint pages={pages}, raw files={len(raw_files)}")
	page_limit = CONFIG["DEBUG_MAX_PAGES"] if CONFIG.get("DEBUG_FETCH_ONE", False) else (CONFIG["MAX_PAGES_PER_APP"] or 10**9)
	cap_reviews = CONFIG["DEBUG_MAX_REVIEWS"] if CONFIG.get("DEBUG_FETCH_ONE", False) else (CONFIG["MAX_REVIEWS_PER_APP"] or 10**18)
	total_en_all = None
	seen_ids = set()
	url = ENDPOINT.format(appid=appid)
	while pages < page_limit and total_written < cap_reviews:
		params = {
			"json": 1,
			"filter": CONFIG["FILTER"],
			"language": CONFIG["LANGUAGE"],
			"review_type": CONFIG["REVIEW_TYPE"],
			"purchase_type": CONFIG["PURCHASE_TYPE"],
			"filter_offtopic_activity": CONFIG["FILTER_OFFTOPIC_ACTIVITY"],
			"num_per_page": CONFIG["NUM_PER_PAGE"],
			"cursor": cursor,
		}
		j = http.get_json(url, params=params)
		if not j or not isinstance(j, dict):
			log(f"[STOP] app={appid} no/invalid JSON; stopping.")
			break
		if total_en_all is None:
			q = j.get("query_summary", {}) or {}
			total_en_all = int(q.get("total_reviews", 0) or 0)
		reviews = j.get("reviews", []) or []
		next_cursor = j.get("cursor", cursor)
		# Save RAW page (always; you wanted raw preserved)
		pages += 1
		raw_path = raw_page_path(appid, pages)
		raw_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
		# DEBUG: Print after saving raw file
		raw_files = list(raw_dir.glob("page_*.json")) if raw_dir.exists() else []
		log(f"[DEBUG] app={appid} AFTER RAW: checkpoint pages={pages}, raw files={len(raw_files)}")
		if not reviews:
			log(f"[INFO] app={appid} page={pages} empty; stopping.")
			break
		# Normalize + dedupe
		rows = normalize_reviews(app_meta, reviews, total_en_all or 0)
		uniq = []
		for r in rows:
			rid = r["reviewId"]
			if rid and rid not in seen_ids:
				seen_ids.add(rid)
				uniq.append(r)
		if not uniq:
			log(f"[INFO] app={appid} page={pages} duplicates-only; stopping.")
			break
		# Apply cap if needed
		remaining = cap_reviews - total_written
		if remaining < len(uniq):
			uniq = uniq[:max(0, remaining)]
		# Write CSVs
		append_app_rows(app_csv, uniq)
		append_master_rows(CONFIG["OUT_MASTER_CSV"], uniq)
		total_written += len(uniq)
		log(f"[OK] app={appid} page={pages} wrote={len(uniq)} total={total_written}")
		# Checkpoint
		cursor = next_cursor or cursor
		ck = {"cursor": cursor, "pages": pages, "reviews": total_written}
		save_ckpt(appid, ck)
		# DEBUG: Print after saving checkpoint
		ck2 = load_ckpt(appid)
		raw_files = list(raw_dir.glob("page_*.json")) if raw_dir.exists() else []
		log(f"[DEBUG] app={appid} AFTER CKPT: checkpoint pages={ck2.get('pages', 0)}, raw files={len(raw_files)}")
		# End conditions
		if len(reviews) < CONFIG["NUM_PER_PAGE"]:
			log(f"[DONE] app={appid} last partial page ({len(reviews)}<{CONFIG['NUM_PER_PAGE']}).")
			break
		if CONFIG["SLEEP_BETWEEN_PAGES_SEC"] > 0:
			time.sleep(CONFIG["SLEEP_BETWEEN_PAGES_SEC"])
	log(f"[SUMMARY] app={appid} pages={pages} reviews_written={total_written} totalEnglishReviews={total_en_all}")

# ======================
# Main
# ======================
def main():
	# master init
	if CONFIG["RESET_MASTER_OUTPUTS"] and Path(CONFIG["OUT_MASTER_CSV"]).exists():
		Path(CONFIG["OUT_MASTER_CSV"]).unlink()
	if not Path(CONFIG["OUT_MASTER_CSV"]).exists():
		write_master_header(CONFIG["OUT_MASTER_CSV"])

	# read pairs
	pairs = load_pairs(CONFIG["INPUT_PAIRS_CSV"])
	if not pairs:
		raise FileNotFoundError(f"No appids found in {CONFIG['INPUT_PAIRS_CSV']}")
	log(f"[INFO] Loaded {len(pairs)} apps from {CONFIG['INPUT_PAIRS_CSV']}")

	http = Http()
	# Only fetch for appid 435480 for testing
	test_appid = "435480"
	test_meta = next((m for m in pairs if str(m.get("appid")) == test_appid), None)
	if test_meta:
		fetch_for_app(http, test_meta)
	else:
		log(f"[ERROR] Appid {test_appid} not found in pairs list.")

	log("[OK] Finished Step 05 – fetch reviews")

if __name__ == "__main__":
	main()
