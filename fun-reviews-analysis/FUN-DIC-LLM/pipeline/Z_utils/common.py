'''
Common Utilities - Provides shared utility functions for logging, directory management, and project root detection.

• Project root detection for reproducible paths
• ASCII-safe logging with masking for emails and SteamIDs
• Directory creation utility for consistent file output

Inputs: Strings, paths, log messages.
Outputs: Utility functions for use in pipeline scripts.
'''
def find_project_root(marker="README.md"):
    """Walk up from current file to find the project root by marker file (e.g., README.md)."""
    p = Path(__file__).resolve().parent
    while not (p / marker).exists() and p != p.parent:
        p = p.parent
    return p
from pathlib import Path
import sys

import unicodedata
def asc(s: str) -> str:
    """Return an ASCII-safe representation for logs, stripping accents."""
    if not isinstance(s, str):
        s = str(s)
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join([c for c in nfkd if not unicodedata.combining(c) and ord(c) < 128])

def log(msg, verbose: bool = True):
    """Log a message, masking emails and SteamIDs. Accepts any input type."""
    if verbose:
        import re
        if not isinstance(msg, str):
            msg = str(msg)
        masked = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', msg)
        masked = re.sub(r'\b\d{17}\b', '[STEAMID]', masked)
        try:
            print(masked, flush=True)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            try:
                safe = masked.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
                print(safe, flush=True)
            except Exception:
                print(ascii(masked), flush=True)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)
