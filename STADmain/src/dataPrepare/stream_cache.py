from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CACHE_SCHEMA_VERSION = 1
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


_LOCAL_RAG_SOURCE_FILES = {
    "KILTStreamSim": "KILT_all_domains_question_answer_label_gpt_lables.csv",
    "MultiHopRAGStreamSim": "MultiHopRAG_full_question_answer_label_gpt_lables_merged_rare.csv",
    "PopQAStreamSim": "PopQA_full_question_answer_label_gpt_lables.csv",
    "RAGBenchStreamSim": "RAGBench_full_question_answer_label.csv",
    "StateCacheQAStreamSim": "StateCacheQA_question_answer_label.csv",
}

_HF_SOURCE_NAMES = {
    "CLINCStreamSim": "clinc_oos/plus",
    "AGNewsStreamSim": "ag_news",
    "GaliboStreamSim": "galileo-ai/20_Newsgroups_Fixed",
    "MINDStreamSim": "microsoft/MIND-derived stream in text_stream.py",
    "BEIRStreamSim": "beir-derived stream in text_stream.py",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_file_meta(path: Path) -> Dict[str, Any]:
    sha = _sha256_file(path)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": sha,
    }


def _source_metadata(repo_root: Path, dataset_name: str) -> Dict[str, Any]:
    
    local_file = _LOCAL_RAG_SOURCE_FILES.get(dataset_name)
    if local_file:
        source_path = repo_root / "src" / "data" / "full_rag_csv" / "cleaned_question_answer_label" / local_file
        return {
            "type": "local_csv",
            "file_name": local_file,
            "file": _maybe_file_meta(source_path),
        }

    return {
        "type": "huggingface_or_generated",
        "name": _HF_SOURCE_NAMES.get(dataset_name, dataset_name),
        "note": "For external datasets, cache identity depends on the local HuggingFace dataset cache/version available when first built.",
    }


def _code_metadata(repo_root: Path) -> Dict[str, Any]:
    files = [
        repo_root / "src" / "dataPrepare" / "text_stream.py",
        repo_root / "src" / "dataPrepare" / "dataset.py",
    ]
    return {p.name: _maybe_file_meta(p) for p in files}


def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_summary(y: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return {"n": 0, "num_labels": 0, "counts": {}}
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return {
        "n": int(y.size),
        "num_labels": int(vals.size),
        "counts": {str(int(v)): int(c) for v, c in zip(vals, counts)},
    }


def _drift_summary(drift_labels: np.ndarray) -> Dict[str, Any]:
    D = np.asarray(drift_labels)
    if D.ndim != 2 or D.shape[1] < 2:
        return {"shape": list(D.shape), "range_sum": None, "cp_sum": None, "change_points": []}
    cps = np.where(D[:, 1].astype(int) == 1)[0].astype(int).tolist()
    return {
        "shape": [int(x) for x in D.shape],
        "range_sum": int(D[:, 0].sum()),
        "cp_sum": int(D[:, 1].sum()),
        "change_points_first_50": cps[:50],
        "num_change_points": int(len(cps)),
    }


def _build_identity_payload(
    repo_root: Path,
    dataset_name: str,
    win_len: int,
    dataset_kwargs: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "win_len": int(win_len),
        "dataset_kwargs": dataset_kwargs or {},
        "seed": int(seed),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "source": _source_metadata(repo_root, dataset_name),
        "code": _code_metadata(repo_root),
    }


def _load_cache_into_dataset(dataset_instance: Any, cache_dir: Path) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
    arrays_path = cache_dir / "arrays.npz"
    meta_path = cache_dir / "meta.json"
    text_path = cache_dir / "texts_answers.jsonl"

    if not arrays_path.exists() or not meta_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Incomplete stream cache directory: {cache_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = np.load(arrays_path, allow_pickle=False)
    x_train = arrays["x_train"]
    y_train = arrays["y_train"]
    x_test = arrays["x_test"]
    y_test = arrays["y_test"]
    drift_labels = arrays["drift_labels"]

    rows = _read_jsonl(text_path)
    texts = [str(r.get("text", "")) for r in rows]
    answers = [str(r.get("answer", "")) for r in rows]

    dataset_instance._data = (x_train, y_train, x_test, y_test, drift_labels)
    dataset_instance.original_texts = texts
    if hasattr(dataset_instance, "_answers"):
        dataset_instance._answers = answers
    dataset_instance.change_points = meta.get("drift_summary", {}).get("change_points_first_50", [])

    meta = dict(meta)
    meta["cache_status"] = "loaded"
    meta["cache_dir"] = str(cache_dir)
    return x_train, y_train, x_test, y_test, drift_labels, meta


def _save_cache(
    dataset_instance: Any,
    cache_dir: Path,
    identity_payload: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    drift_labels: np.ndarray,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    answers: List[str] = []

    if hasattr(dataset_instance, "get_original_texts"):
        texts = list(dataset_instance.get_original_texts() or [])
    if hasattr(dataset_instance, "get_ground_truth_answers"):
        answers = list(dataset_instance.get_ground_truth_answers() or [])

    if len(answers) < len(texts):
        answers = answers + [""] * (len(texts) - len(answers))
    if len(texts) < int(np.asarray(x_test).shape[0]):
        texts = texts + [""] * (int(np.asarray(x_test).shape[0]) - len(texts))
        answers = answers + [""] * (int(np.asarray(x_test).shape[0]) - len(answers))

    np.savez_compressed(
        cache_dir / "arrays.npz",
        x_train=np.asarray(x_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        x_test=np.asarray(x_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        drift_labels=np.asarray(drift_labels, dtype=np.int64),
    )

    _write_jsonl(
        cache_dir / "texts_answers.jsonl",
        (
            {"idx": i, "text": texts[i], "answer": answers[i] if i < len(answers) else ""}
            for i in range(len(texts))
        ),
    )

    meta = {
        **identity_payload,
        "cache_status": "built",
        "cache_dir": str(cache_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "array_shapes": {
            "x_train": [int(x) for x in np.asarray(x_train).shape],
            "y_train": [int(x) for x in np.asarray(y_train).shape],
            "x_test": [int(x) for x in np.asarray(x_test).shape],
            "y_test": [int(x) for x in np.asarray(y_test).shape],
            "drift_labels": [int(x) for x in np.asarray(drift_labels).shape],
        },
        "label_summary": _label_summary(y_test),
        "drift_summary": _drift_summary(drift_labels),
        "num_texts": int(len(texts)),
        "num_answers": int(sum(1 for a in answers if str(a).strip())),
    }

    with (cache_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=_json_default)

    return meta


def load_or_build_dataset(
    dataset_instance: Any,
    repo_root: Path,
    cache_root: Path,
    dataset_name: str,
    win_len: int,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    rebuild: bool = False,
    disable_cache: bool = False,
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
   
    repo_root = Path(repo_root).resolve()
    cache_root = Path(cache_root).resolve()
    dataset_kwargs = dataset_kwargs or {}

    identity_payload = _build_identity_payload(
        repo_root=repo_root,
        dataset_name=dataset_name,
        win_len=int(win_len),
        dataset_kwargs=dataset_kwargs,
        seed=int(seed),
    )
    key = _cache_key(identity_payload)
    cache_dir = cache_root / dataset_name / f"win{int(win_len)}_{key}"

    if disable_cache:
        data = dataset_instance.data()
        meta = {
            **identity_payload,
            "cache_status": "disabled",
            "cache_dir": str(cache_dir),
            "array_shapes": {
                "x_train": [int(x) for x in np.asarray(data[0]).shape],
                "y_train": [int(x) for x in np.asarray(data[1]).shape],
                "x_test": [int(x) for x in np.asarray(data[2]).shape],
                "y_test": [int(x) for x in np.asarray(data[3]).shape],
                "drift_labels": [int(x) for x in np.asarray(data[4]).shape],
            },
            "label_summary": _label_summary(data[3]),
            "drift_summary": _drift_summary(data[4]),
        }
        return (*data, meta)

    if cache_dir.exists() and not rebuild:
        print(f"[stream_cache] Loading frozen stream from: {cache_dir}")
        return _load_cache_into_dataset(dataset_instance, cache_dir)

    print(f"[stream_cache] Building frozen stream cache: {cache_dir}")
    data = dataset_instance.data()
    meta = _save_cache(dataset_instance, cache_dir, identity_payload, *data)
    return (*data, meta)
