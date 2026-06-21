"""Local filesystem cache for distilled grader adapters.

The cache key is (dataset_id, grader_base_model). When a distilled grader
exists for a key and its audit fidelity meets the threshold, callers can use
it in place of the expensive frontier judge.

Layout:
    cache/
        index.json                        # registry of all cached entries
        {key}/
            traces.jsonl                  # accumulated grading traces
            adapter/                      # LoRA weights (copied or symlinked)
            audit.json                    # fidelity/grounding report
"""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CACHE_DIR = Path("cache")
MIN_PEARSON = 0.85  # minimum fidelity to trust the distilled grader


def _slug(dataset_id: str, grader_base: str) -> str:
    """Stable filesystem-safe key from dataset + base model."""
    raw = f"{dataset_id}--{grader_base}"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", raw)


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / "index.json"


def _read_index(cache_dir: Path) -> dict:
    p = _index_path(cache_dir)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _write_index(cache_dir: Path, index: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(_index_path(cache_dir), "w") as f:
        json.dump(index, f, indent=2)


@dataclass
class CachedGrader:
    dataset_id: str
    grader_base: str
    adapter_path: Path
    pearson: float
    trace_count: int
    registered_at: str

    def is_trustworthy(self, min_pearson: float = MIN_PEARSON) -> bool:
        return self.pearson >= min_pearson


def traces_path(dataset_id: str, grader_base: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Path where grading traces for this key should be accumulated."""
    return cache_dir / _slug(dataset_id, grader_base) / "traces.jsonl"


def lookup(
    dataset_id: str,
    grader_base: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_pearson: float = MIN_PEARSON,
) -> "CachedGrader | None":
    """Return the cached grader for this key if it exists and is trustworthy."""
    index = _read_index(cache_dir)
    key = _slug(dataset_id, grader_base)
    entry = index.get(key)
    if entry is None:
        return None

    adapter = cache_dir / key / "adapter"
    if not adapter.exists():
        return None

    grader = CachedGrader(
        dataset_id=dataset_id,
        grader_base=grader_base,
        adapter_path=adapter,
        pearson=entry.get("pearson", 0.0),
        trace_count=entry.get("trace_count", 0),
        registered_at=entry.get("registered_at", ""),
    )
    return grader if grader.is_trustworthy(min_pearson) else None


def register(
    dataset_id: str,
    grader_base: str,
    adapter_path: Path,
    audit_report: dict,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CachedGrader:
    """Register a freshly trained grader adapter into the cache.

    Copies the adapter weights into the cache directory so the entry is
    self-contained regardless of where the training run saved them.
    """
    key = _slug(dataset_id, grader_base)
    dest = cache_dir / key / "adapter"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(adapter_path, dest)

    audit_dest = cache_dir / key / "audit.json"
    with open(audit_dest, "w") as f:
        json.dump(audit_report, f, indent=2)

    trace_count = 0
    tp = traces_path(dataset_id, grader_base, cache_dir)
    if tp.exists():
        with open(tp) as f:
            trace_count = sum(1 for line in f if line.strip())

    index = _read_index(cache_dir)
    pearson = (
        audit_report.get("fidelity", {}).get("pearson")
        or audit_report.get("context", {}).get("grader_vs_teacher_pearson", 0.0)
    )
    index[key] = {
        "dataset_id": dataset_id,
        "grader_base": grader_base,
        "pearson": pearson,
        "trace_count": trace_count,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_index(cache_dir, index)

    return CachedGrader(
        dataset_id=dataset_id,
        grader_base=grader_base,
        adapter_path=dest,
        pearson=pearson,
        trace_count=trace_count,
        registered_at=index[key]["registered_at"],
    )
