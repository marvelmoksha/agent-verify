"""Guard against committing secrets. Fails if a key-shaped string appears in any project file
(the real `.env` is git-ignored and is skipped)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".yaml", ".yml", ".json", ".sh", ".example", ".cfg", ""}
KEY_PATTERNS = [
    re.compile(r"sk-or-v1-[0-9a-fA-F]{20,}"),       # OpenRouter
    re.compile(r"freellmapi-[0-9a-fA-F]{20,}"),     # FreeLLMAPI unified key
    re.compile(r"sk-[A-Za-z0-9]{32,}"),             # generic provider keys
    re.compile(r"hf_[A-Za-z0-9]{30,}"),             # Hugging Face tokens
]


def _project_files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.name == ".env" or p.suffix not in TEXT_SUFFIXES:
            continue
        yield p


def test_no_key_shaped_strings_in_project_files():
    hits = []
    for p in _project_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(pat.search(text) for pat in KEY_PATTERNS):
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"Key-like strings found in: {hits}. Move them to .env and rotate the keys."


def test_env_example_has_empty_key_values():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith(("OPENROUTER_API_KEY=", "FREELLMAPI_KEY=")):
            assert line.split("=", 1)[1].strip() == "", f".env.example must not hold real keys: {line.split('=')[0]}"
