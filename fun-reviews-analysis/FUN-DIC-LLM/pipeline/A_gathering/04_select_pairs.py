# 04_select_pairs.py (moved to pipeline/)
# coding: utf-8

'''
Step 04: Select Pairs - Selects top/bottom app pairs using genre rules and review thresholds.

• Applies genre and review count rules to filter apps
• Selects top and bottom pairs for analysis
• Ensures reproducibility and helper utility usage

Inputs: steamspy_filtered.csv
Outputs: steamspy_top_bottom_pairs.csv
'''


import os
import time
import random
import unicodedata
import re
import sys
import pandas as pd
from pipeline.Z_utils.common import asc, log, ensure_dir, find_project_root
from pathlib import Path


ROOT = find_project_root()
DATA = ROOT / "data"
from typing import Any, Dict, List, Optional, Tuple
import requests

### Pipeline Step 04: Select pairs
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
	# Files
	"INPUT_FILTERED": str(get_env("PAIRS_INPUT_FILTERED", DATA / "steamspy_filtered.csv")),
	"OUTPUT_PAIRS":   str(get_env("PAIRS_OUTPUT_PAIRS", DATA / "steamspy_top_bottom_pairs.csv")),

	# Selection
	"NUM_PAIRS": get_env("PAIRS_NUM_PAIRS", 5),
	"MIN_ENGLISH_REVIEWS": get_env("PAIRS_MIN_ENGLISH_REVIEWS", 50),

	# Networking
	"REQUEST_TIMEOUT": get_env("PAIRS_REQUEST_TIMEOUT", 15.0),
	"USER_AGENT": get_env("PAIRS_USER_AGENT", "Mozilla/5.0 (SteamPairs/1.0; +study)"),
	# Retry/backoff
	"MAX_ATTEMPTS": get_env("PAIRS_MAX_ATTEMPTS", 100),
	"INITIAL_DELAY": get_env("PAIRS_INITIAL_DELAY", 2.0),
	"MAX_DELAY": get_env("PAIRS_MAX_DELAY", 60.0),
	"JITTER_FRAC": get_env("PAIRS_JITTER_FRAC", 0.25),

	# Non-game / non-genre labels (store-facing labels that are not gameplay genres)
	"NON_GAME_LABELS": set(get_env("PAIRS_NON_GAME_LABELS",
		"utilities,software,web publishing,design and illustration,design & illustration,audio production,video production,photo editing,education,animation and modeling,animation & modeling,typing"
	).split(",")),

	# Maintain the previous jump rule labels as well
	"JUMP_LABELS": set(get_env("PAIRS_JUMP_LABELS", "free to play,early access").split(",")),

	# Debug: list of specific appids to trace (type, genres, english reviews)
	"DEBUG_TRACE_APPIDS": get_env("PAIRS_DEBUG_TRACE_APPIDS", []),

	# Logging
	"VERBOSE": get_env("PAIRS_VERBOSE", True),
}

# Endpoints
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={appid}&l=en"
APPREVIEWS_URL = (
	"https://store.steampowered.com/appreviews/{appid}"
	"?json=1&filter=all&language=english&purchase_type=all&num_per_page=1"
)

# Expected input columns (Step 03 output)
IN_COLS = [
	"appid", "name",
	"posReview", "negReview", "totReview",
	"posRate", "popularity", "acceptanceScore",
	"rankAcceptance", "rankFiltered",
]

# Output columns for the pairs CSV
OUT_COLS = [
	"pairIndex",      # 1..NUM_PAIRS
	"type",           # "top" or "bottom"
	"appid",
	"name",
	"firstGenre",
	"allGenres",      # as returned by appdetails, comma-separated
	"englishReviews", # integer (from appreviews)
	"posReview", "negReview", "totReview",
	"posRate", "popularity", "acceptanceScore",
	"rankAcceptance", "rankFiltered",
]


# ======================
# ASCII-safe logging utilities


# ======================
# CSV IO
# ======================
def read_filtered(path: str) -> List[Dict[str, Any]]:
	df = pd.read_csv(path, encoding="utf-8-sig")
	df = df.sort_values(by="acceptanceScore", ascending=False)
	return df.to_dict(orient="records")

def write_pairs(path: str, rows: List[Dict[str, Any]]):
	ensure_dir(Path(path).parent)
	df = pd.DataFrame(rows)
	df.to_csv(path, columns=OUT_COLS, index=False, encoding="utf-8")


# ======================
# Networking w/ backoff
# ======================
class HttpClient:
	def __init__(self):
		self.sess = requests.Session()
		self.sess.headers.update({"User-Agent": CONFIG["USER_AGENT"]})

	def _wait(self, attempt: int) -> float:
		base = CONFIG["INITIAL_DELAY"] * (2 ** (attempt - 1))
		base = min(base, CONFIG["MAX_DELAY"])
		jitter = base * CONFIG["JITTER_FRAC"]
		wait_s = max(0.0, base + random.uniform(-jitter, jitter))
		return wait_s

	def get_json(self, url: str) -> Optional[Any]:
		max_attempts = CONFIG["MAX_ATTEMPTS"]
		for attempt in range(1, max_attempts + 1):
			try:
				resp = self.sess.get(url, timeout=CONFIG["REQUEST_TIMEOUT"])
				status = resp.status_code
				if status == 200:
					return resp.json()
				# Retry for 429 / 5xx
				if status in (429, 500, 502, 503, 504):
					wait_s = self._wait(attempt)
					log(f"[RETRY] {status} GET {url} (attempt {attempt}/{max_attempts}) wait={wait_s:.2f}s")
					time.sleep(wait_s)
					continue
				# Non-retryable
				log(f"[ERROR] HTTP {status} GET {url} — not retrying")
				return None
			except requests.RequestException as ex:
				wait_s = self._wait(attempt)
				log(f"[RETRY] Exception GET {url}: {asc(str(ex))} (attempt {attempt}/{max_attempts}) wait={wait_s:.2f}s")
				time.sleep(wait_s)
			except Exception as ex:
				log(f"[ERROR] Unexpected exception GET {url}: {asc(str(ex))} — not retrying")
				return None
		log(f"[ERROR] Exceeded max attempts for {url}")
		return None


# ======================
# Steam store helpers
# ======================
def canon(s: str) -> str:
	"""Lowercase, strip accents/punct, collapse spaces."""
	if not s:
		return ""
	s = unicodedata.normalize("NFKD", s)
	s = "".join(ch for ch in s if not unicodedata.combining(ch))
	s = s.lower().strip()
	s = s.replace("&", " and ")
	s = re.sub(r"[\\/_+,-]+", " ", s)
	s = re.sub(r"\s+", " ", s)
	s = re.sub(r"[^a-z0-9 ]", "", s)
	return s.strip()

def first_valid_genre(genres: List[str]) -> Tuple[str, str]:
	"""
	Determine the first gameplay genre applying the 'jump' rule:
	  - Skip F2P/EA (JUMP_LABELS)
	  - Skip obvious non-game labels (NON_GAME_LABELS)
	Consider only the first three store 'genres' returned by appdetails.
	Returns (firstGenre, allGenresJoined); firstGenre == '' => no valid gameplay genre.
	"""
	cleaned = [g.strip() for g in (genres or []) if g and g.strip()]
	joined = ", ".join(cleaned)
	jump = CONFIG["JUMP_LABELS"]
	non_game = CONFIG["NON_GAME_LABELS"]
	for g in cleaned[:3]:  # only consider first three
		cg = canon(g)
		if cg and (cg not in jump) and (cg not in non_game):
			return g, joined
	return "", joined



class SteamStore:
	def __init__(self, http: HttpClient):
		self.http = http
		self.cache_details: Dict[str, Dict[str, Any]] = {}
		self.cache_en: Dict[str, int] = {}

	def appdetails(self, appid: str) -> Dict[str, Any]:
		if appid in self.cache_details:
			return self.cache_details[appid]
		url = APPDETAILS_URL.format(appid=appid)
		j = self.http.get_json(url)
		out = {"ok": False, "type": None, "genres": []}
		try:
			blk = j.get(str(appid), {}) if isinstance(j, dict) else {}
			if blk.get("success"):
				data = blk.get("data", {}) or {}
				typ = (data.get("type") or "").strip().lower()
				genres = [g.get("description", "").strip() for g in (data.get("genres") or []) if g.get("description")]
				out = {"ok": True, "type": typ, "genres": genres}
		except Exception:
			out = {"ok": False, "type": None, "genres": []}
		self.cache_details[appid] = out
		# Debug trace for specific appids
		if appid in CONFIG["DEBUG_TRACE_APPIDS"]:
			log(f"[DEBUG] appdetails appid={appid} type={out.get('type')} genres={asc(', '.join(out.get('genres', [])))}")
		return out

	def english_reviews(self, appid: str) -> int:
		if appid in self.cache_en:
			return self.cache_en[appid]
		if CONFIG["MIN_ENGLISH_REVIEWS"] <= 0:
			self.cache_en[appid] = 0
			return 0
		url = APPREVIEWS_URL.format(appid=appid)
		j = self.http.get_json(url)
		total = 0
		try:
			if isinstance(j, dict):
				q = j.get("query_summary", {}) or {}
				total = int(q.get("total_reviews", 0)) or (int(q.get("total_positive", 0)) + int(q.get("total_negative", 0)))
		except Exception:
			total = 0
		self.cache_en[appid] = max(0, total)
		# Debug trace
		if appid in CONFIG["DEBUG_TRACE_APPIDS"]:
			log(f"[DEBUG] english_reviews appid={appid} total_en={self.cache_en[appid]}")
		return self.cache_en[appid]


# ======================
# Selection logic
# ======================
def select_pairs(rows: List[Dict[str, Any]], k_pairs: int) -> List[Dict[str, Any]]:
	http = HttpClient()
	api = SteamStore(http)

	used_first_genres = set()
	picked: List[Dict[str, Any]] = []

	i, j = 0, len(rows) - 1
	pair_idx = 0

	log(f"[START] candidates={len(rows)} target_pairs={k_pairs} min_english={CONFIG['MIN_ENGLISH_REVIEWS']}")
	while i < j and pair_idx < k_pairs:
		# --------- pick A (top) ----------
		A = None; A_fg = ""; A_allg = ""; A_en = 0
		while i < j and A is None:
			ra = rows[i]
			appid = ra["appid"]
			det = api.appdetails(appid)
			if not (det.get("ok") and det.get("type") == "game"):
				log(f"[SKIP A] i={i} appid={appid} name=`{asc(ra['name'])}` reason=not_game_or_details_fail type={det.get('type')}")
				i += 1
				continue
			fg, allg = first_valid_genre(det.get("genres", []))
			if not fg:
				log(f"[SKIP A] i={i} appid={appid} name=`{asc(ra['name'])}` reason=no_valid_first_genre genres={asc(allg)}")
				i += 1
				continue
			if canon(fg) in used_first_genres:
				log(f"[SKIP A] i={i} appid={appid} name=`{asc(ra['name'])}` reason=first_genre_taken first_genre={asc(fg)}")
				i += 1
				continue
			en = api.english_reviews(appid)
			if CONFIG["MIN_ENGLISH_REVIEWS"] > 0 and en < CONFIG["MIN_ENGLISH_REVIEWS"]:
				log(f"[SKIP A] i={i} appid={appid} name=`{asc(ra['name'])}` reason=low_english_reviews en={en}")
				i += 1
				continue
			A, A_fg, A_allg, A_en = ra, fg, allg, en

		if A is None:
			log("[STOP] ran out of top candidates.")
			break

		# --------- pick B (bottom) ----------
		B = None; B_fg = ""; B_allg = ""; B_en = 0
		taken_for_B = set(used_first_genres); taken_for_B.add(canon(A_fg))
		while i < j and B is None:
			rb = rows[j]
			appid = rb["appid"]
			det = api.appdetails(appid)
			if not (det.get("ok") and det.get("type") == "game"):
				log(f"[SKIP B] j={j} appid={appid} name=`{asc(rb['name'])}` reason=not_game_or_details_fail type={det.get('type')}")
				j -= 1
				continue
			fg, allg = first_valid_genre(det.get("genres", []))
			if not fg:
				log(f"[SKIP B] j={j} appid={appid} name=`{asc(rb['name'])}` reason=no_valid_first_genre genres={asc(allg)}")
				j -= 1
				continue
			if canon(fg) in taken_for_B:
				log(f"[SKIP B] j={j} appid={appid} name=`{asc(rb['name'])}` reason=first_genre_taken first_genre={asc(fg)}")
				j -= 1
				continue
			en = api.english_reviews(appid)
			if CONFIG["MIN_ENGLISH_REVIEWS"] > 0 and en < CONFIG["MIN_ENGLISH_REVIEWS"]:
				log(f"[SKIP B] j={j} appid={appid} name=`{asc(rb['name'])}` reason=low_english_reviews en={en}")
				j -= 1
				continue
			B, B_fg, B_allg, B_en = rb, fg, allg, en

		if B is None:
			log("[STOP] ran out of bottom candidates.")
			break

		pair_idx += 1
		used_first_genres.add(canon(A_fg))
		used_first_genres.add(canon(B_fg))

		picked.append({
			"pairIndex": pair_idx, "type": "top",
			"appid": A["appid"], "name": A["name"],
			"firstGenre": A_fg, "allGenres": A_allg, "englishReviews": A_en,
			"posReview": A["posReview"], "negReview": A["negReview"], "totReview": A["totReview"],
			"posRate": A["posRate"], "popularity": A["popularity"], "acceptanceScore": A["acceptanceScore"],
			"rankAcceptance": A.get("rankAcceptance", 0),
			"rankFiltered":  A.get("rankFiltered", 0),
		})
		picked.append({
			"pairIndex": pair_idx, "type": "bottom",
			"appid": B["appid"], "name": B["name"],
			"firstGenre": B_fg, "allGenres": B_allg, "englishReviews": B_en,
			"posReview": B["posReview"], "negReview": B["negReview"], "totReview": B["totReview"],
			"posRate": B["posRate"], "popularity": B["popularity"], "acceptanceScore": B["acceptanceScore"],
			"rankAcceptance": B.get("rankAcceptance", 0),
			"rankFiltered":  B.get("rankFiltered", 0),
		})

		log(f"[PAIR #{pair_idx}] {asc(A['name'])} ({A['appid']}, {asc(A_fg)}) x "
			f"{asc(B['name'])} ({B['appid']}, {asc(B_fg)}) used_first_genres={sorted(list(used_first_genres))}")

		i += 1
		j -= 1

	log(f"[END] total_pairs={len(picked)//2}")
	return picked


# ======================
# Main
# ======================
def main():
	inp = CONFIG["INPUT_FILTERED"]
	out = CONFIG["OUTPUT_PAIRS"]

	if not Path(inp).exists():
		raise FileNotFoundError(f"Input not found: {inp}")

	rows = read_filtered(inp)
	log(f"[INFO] Loaded {len(rows)} candidates from {inp}")

	picked = select_pairs(rows, CONFIG["NUM_PAIRS"])
	write_pairs(out, picked)
	log(f"[OK] Wrote {len(picked)//2} pairs ({len(picked)} rows) to {out}")

if __name__ == "__main__":
	main()
