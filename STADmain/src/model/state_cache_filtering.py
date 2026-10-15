from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import os
import random

import numpy as np


@dataclass(frozen=True)
class StateCacheFilterConfig:
    enabled: bool = True
    build_frac: float = 0.25
    max_build_queries: int = 25
    min_build_queries: int = 3
    query_topk: int = 3
    min_sim: float = 0.35
    min_docs: int = 1
    doc_emb_chars: int = 1200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def get_state_cache_filter_config() -> StateCacheFilterConfig:
   
    return StateCacheFilterConfig(
        enabled=_env_bool("DTSS_STATE_CACHE_FILTERING", True),
        build_frac=max(0.0, min(1.0, _env_float("DTSS_STATE_CACHE_BUILD_FRAC", 0.40))),
        max_build_queries=max(1, _env_int("DTSS_STATE_CACHE_MAX_BUILD_QUERIES", 25)),
        min_build_queries=max(1, _env_int("DTSS_STATE_CACHE_MIN_BUILD_QUERIES", 3)),
        query_topk=max(1, _env_int("DTSS_STATE_CACHE_QUERY_TOPK", 5)),
        min_sim=float(_env_float("DTSS_STATE_CACHE_MIN_SIM", 0.25)),
        min_docs=max(1, _env_int("DTSS_STATE_CACHE_MIN_DOCS", 1)),
        doc_emb_chars=max(100, _env_int("DTSS_STATE_CACHE_DOC_EMB_CHARS", 1200)),
    )


def select_cache_build_queries(
    primary_queries: Sequence[Dict[str, Any]],
    state_id: int,
    seed: int,
    config: Optional[StateCacheFilterConfig] = None,
) -> List[Dict[str, Any]]:
   
    primary_queries = list(primary_queries)
    if not primary_queries:
        return []

    cfg = config or get_state_cache_filter_config()

    if not cfg.enabled:
        return primary_queries

    n = int(math.ceil(len(primary_queries) * cfg.build_frac))
    n = max(cfg.min_build_queries, n)
    n = min(cfg.max_build_queries, n, len(primary_queries))

    rng = random.Random(int(seed) + int(state_id) + 1337)
    shuffled = list(primary_queries)
    rng.shuffle(shuffled)
    return shuffled[:n]


def _doc_to_embedding_text(doc: Dict[str, Any], max_chars: int) -> str:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title and text:
        return f"{title}. {text[:max_chars]}"
    if title:
        return title[:max_chars]
    return text[:max_chars]


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def build_state_cache_index(
    cached_docs: Sequence[Dict[str, Any]],
    encoder: Any,
    config: Optional[StateCacheFilterConfig] = None,
) -> Dict[str, Any]:
    
    cfg = config or get_state_cache_filter_config()
    valid_docs: List[Dict[str, Any]] = []
    doc_texts: List[str] = []

    for doc in cached_docs:
        if not isinstance(doc, dict):
            continue
        emb_text = _doc_to_embedding_text(doc, max_chars=cfg.doc_emb_chars)
        if not emb_text.strip():
            continue
        valid_docs.append(doc)
        doc_texts.append(emb_text)

    if not doc_texts:
        return {
            "docs": [],
            "doc_texts": [],
            "embeddings": np.zeros((0, 0), dtype=np.float32),
            "config": cfg.to_dict(),
        }

    embeddings = encoder.encode(
        doc_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype(np.float32)
    embeddings = _l2_normalize_rows(embeddings)

    return {
        "docs": valid_docs,
        "doc_texts": doc_texts,
        "embeddings": embeddings,
        "config": cfg.to_dict(),
    }


def select_relevant_cached_docs(
    query: str,
    cache_index: Dict[str, Any],
    encoder: Any,
    config: Optional[StateCacheFilterConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
   
    cfg = config or get_state_cache_filter_config()

    docs = list(cache_index.get("docs", []) or [])
    D = cache_index.get("embeddings")

    if not query or not docs or D is None or np.asarray(D).size == 0:
        return [], {
            "cache_sufficient": False,
            "num_selected": 0,
            "max_sim": 0.0,
            "mean_sim": 0.0,
            "min_sim_threshold": float(cfg.min_sim),
            "min_docs_required": int(cfg.min_docs),
            "query_topk": int(cfg.query_topk),
        }

    q = encoder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    q = _l2_normalize_rows(q)
    sims = np.asarray(D, dtype=np.float32).dot(q[0])

    if sims.size == 0:
        max_sim = 0.0
        mean_sim = 0.0
        order = []
    else:
        max_sim = float(np.max(sims))
        mean_sim = float(np.mean(sims))
        order = np.argsort(-sims)

    selected: List[Dict[str, Any]] = []
    for idx in order:
        sim = float(sims[idx])
        if sim < cfg.min_sim:
            continue
        doc = dict(docs[int(idx)])
        doc["state_cache_query_similarity"] = sim
        selected.append(doc)
        if len(selected) >= cfg.query_topk:
            break

    cache_sufficient = (len(selected) >= cfg.min_docs) and (max_sim >= cfg.min_sim)

    return selected, {
        "cache_sufficient": bool(cache_sufficient),
        "num_selected": int(len(selected)),
        "max_sim": float(max_sim),
        "mean_sim": float(mean_sim),
        "min_sim_threshold": float(cfg.min_sim),
        "min_docs_required": int(cfg.min_docs),
        "query_topk": int(cfg.query_topk),
    }
