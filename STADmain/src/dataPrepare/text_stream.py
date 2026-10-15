#make sure to chnge the dataset folder before running the code
from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from .dataset import Dataset


base_path = Path(__file__).resolve().parent.parent.parent
_st_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings



def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def _make_drift_matrix_from_labels(
    labels: np.ndarray,
    win_len: int,
    range_len: Optional[int] = None,
) -> np.ndarray:
   
    labels = np.asarray(labels, dtype=int).reshape(-1)
    N = len(labels)

    win_len = int(max(1, win_len))
    if range_len is None:
        range_len = win_len
    range_len = int(max(1, range_len))

    drift_cp = np.zeros(N, dtype=int)
    drift_range = np.zeros(N, dtype=int)

    if N > 1:
        drift_cp[1:] = (labels[1:] != labels[:-1]).astype(int)

    for b in np.where(drift_cp == 1)[0]:
        drift_range[b:min(N, b + range_len)] = 1

    return np.stack([drift_range, drift_cp], axis=1).astype(int)


def _change_points_from_drift_matrix(drift_mat: np.ndarray) -> List[int]:
    
    drift_mat = np.asarray(drift_mat, dtype=int)
    if drift_mat.ndim != 2 or drift_mat.shape[1] < 2:
        return []
    return np.where(drift_mat[:, 1] == 1)[0].astype(int).tolist()


def _pack_stream_data(
    X: np.ndarray,
    labels: np.ndarray,
    drift_mat: np.ndarray,
    train_frac: float = 0.10,
):
   
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    drift_mat = np.asarray(drift_mat, dtype=int)

    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X/labels length mismatch: X={X.shape[0]}, labels={labels.shape[0]}")

    if drift_mat.ndim != 2 or drift_mat.shape[1] < 2:
        raise ValueError(f"drift_mat must have shape (N, 2), got {drift_mat.shape}")

    if drift_mat.shape[0] != X.shape[0]:
        raise ValueError(
            f"X/drift_mat length mismatch: X={X.shape[0]}, drift_mat={drift_mat.shape[0]}"
        )

    N = X.shape[0]
    n_train = max(1, int(train_frac * N)) if N > 0 else 0
    n_train = min(n_train, N)

    return (
        X[:n_train],
        labels[:n_train].reshape(-1, 1),
        X,
        labels.reshape(-1, 1),
        drift_mat.astype(int),
    )


def _encode_texts(texts: List[str], log_name: str) -> np.ndarray:
    print(f"[{log_name}] Encoding texts with SentenceTransformer...")
    X = _st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return l2_normalize_rows(X)


def _simulate_reordered_stream(
    labels: np.ndarray,
    win_len: int,
    seed: int = 42,
    block_frac: float = 0.70,
    revisit_frac: float = 0.10,
    noise_between_frac: float = 0.05,
    label_order: str = "random",
    noise_burst_len: int = 6,
    revisit_rounds: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
   
    win_len = int(max(1, win_len))
    revisit_rounds = int(max(1, revisit_rounds))
    noise_burst_len = int(max(0, noise_burst_len))

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    N = labels.shape[0]

    if N == 0:
        return np.array([], dtype=int), np.zeros((0, 2), dtype=int)

    uniq = np.unique(labels)
    lab_list = uniq.tolist()

    if label_order == "random":
        rng.shuffle(lab_list)
    elif label_order == "sorted":
        lab_list = sorted(lab_list)
    else:
        raise ValueError(f"Unknown label_order={label_order}")

    idx_by_label: Dict[int, np.ndarray] = {}
    for lab in uniq:
        idx_by_label[int(lab)] = rng.permutation(np.where(labels == lab)[0])

    main_blocks: Dict[int, np.ndarray] = {}
    revisit_chunks: Dict[int, List[np.ndarray]] = {}
    noise_pool: List[int] = []

    for lab in lab_list:
        idx = idx_by_label[int(lab)]
        n = len(idx)

        n_main = int(np.floor(block_frac * n))
        n_revisit = int(np.floor(revisit_frac * n))
        n_main = max(0, min(n_main, n))
        n_revisit = max(0, min(n_revisit, n - n_main))

        main = idx[:n_main]
        revisit = idx[n_main:n_main + n_revisit]
        rest = idx[n_main + n_revisit:]

        main_blocks[int(lab)] = main
        revisit_chunks[int(lab)] = list(np.array_split(revisit, revisit_rounds))
        noise_pool.extend(rest.tolist())

    rng.shuffle(noise_pool)

    total_noise_injections = int(np.floor(noise_between_frac * N))
    total_noise_injections = max(0, min(total_noise_injections, len(noise_pool)))
    noise_ptr = 0

    def _take_noise_burst() -> List[int]:
        nonlocal noise_ptr
        if total_noise_injections <= 0 or noise_burst_len <= 0:
            return []
        if noise_ptr >= total_noise_injections:
            return []
        k = min(noise_burst_len, total_noise_injections - noise_ptr)
        out = noise_pool[noise_ptr:noise_ptr + k]
        noise_ptr += k
        return out

    order: List[int] = []

  
    for lab in lab_list:
        order.extend(main_blocks[int(lab)].tolist())
        order.extend(_take_noise_burst())

   
    for r in range(revisit_rounds):
        round_labels = lab_list.copy()
        rng.shuffle(round_labels)
        for lab in round_labels:
            chunks = revisit_chunks[int(lab)]
            if r < len(chunks) and len(chunks[r]) > 0:
                order.extend(chunks[r].tolist())
            order.extend(_take_noise_burst())

  
    if noise_ptr < total_noise_injections:
        order.extend(noise_pool[noise_ptr:total_noise_injections])

    seen = set(order)
    for lab in lab_list:
        for idx in idx_by_label[int(lab)].tolist():
            if idx not in seen:
                order.append(int(idx))
                seen.add(int(idx))


    final_order: List[int] = []
    seen_final = set()
    for idx in order:
        idx = int(idx)
        if 0 <= idx < N and idx not in seen_final:
            final_order.append(idx)
            seen_final.add(idx)

    order_idx = np.asarray(final_order, dtype=int)
    if len(order_idx) != N:
        raise ValueError(f"Stream simulation lost samples: kept {len(order_idx)} / {N}")

    labels_reordered = labels[order_idx]
    drift_mat = _make_drift_matrix_from_labels(labels_reordered, win_len=win_len)
    return order_idx, drift_mat




class CLINCStream(Dataset):
    def __init__(self, win_len):
        super().__init__(name="CLINCStream", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        HF_DATASET_ID = "clinc_oos"
        HF_CONFIG = "plus"
        print(f"[CLINCStream] Loading: {HF_DATASET_ID} ({HF_CONFIG})")

        ds_train = load_dataset(HF_DATASET_ID, HF_CONFIG, split="train")
        ds_val = load_dataset(HF_DATASET_ID, HF_CONFIG, split="validation")
        ds_test = load_dataset(HF_DATASET_ID, HF_CONFIG, split="test")

        texts_all = list(ds_train["text"]) + list(ds_val["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["intent"]) + list(ds_val["intent"]) + list(ds_test["intent"])

        texts: List[str] = []
        raw_labels: List[str] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            raw_labels.append(str(lab))

        unique_labels = sorted(set(raw_labels))
        label2id = {lab: i for i, lab in enumerate(unique_labels)}
        labels = np.array([label2id[lab] for lab in raw_labels], dtype=int)

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        print(
            f"[CLINCStream] Sample-level change points: "
            f"{len(self.change_points)} / {len(labels)} samples "
            f"({len(self.change_points) / max(len(labels), 1):.4%})"
        )

        X = _encode_texts(texts, "CLINCStream")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class CLINCStreamSim(Dataset):
    def __init__(self, win_len):
        super().__init__(name="CLINCStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        HF_DATASET_ID = "clinc_oos"
        HF_CONFIG = "plus"
        print(f"[CLINCStreamSim] Loading: {HF_DATASET_ID} ({HF_CONFIG})")

        ds_train = load_dataset(HF_DATASET_ID, HF_CONFIG, split="train")
        ds_val = load_dataset(HF_DATASET_ID, HF_CONFIG, split="validation")
        ds_test = load_dataset(HF_DATASET_ID, HF_CONFIG, split="test")

        texts_all = list(ds_train["text"]) + list(ds_val["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["intent"]) + list(ds_val["intent"]) + list(ds_test["intent"])

        texts: List[str] = []
        raw_labels: List[str] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            raw_labels.append(str(lab))

        unique_labels = sorted(set(raw_labels))
        label2id = {lab: i for i, lab in enumerate(unique_labels)}
        labels = np.array([label2id[lab] for lab in raw_labels], dtype=int)

        print(f"[CLINCStreamSim] Unique labels/intents: {len(unique_labels)}")

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.85,
            revisit_frac=0.15,
            noise_between_frac=0.00,
            label_order="random",
            noise_burst_len=0,
            revisit_rounds=1,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        labels = labels[order_idx]

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        print(
            f"[CLINCStreamSim] Sample-level change points: "
            f"{len(self.change_points)} / {len(labels)} samples "
            f"({len(self.change_points) / max(len(labels), 1):.4%})"
        )
        print(f"[CLINCStreamSim] First 30 change points: {self.change_points[:30]}")

        X = _encode_texts(texts, "CLINCStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers



class GaliboStream(Dataset):
    def __init__(self, win_len):
        super().__init__(name="GaliboStream", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        HF_DATASET_ID = "galileo-ai/20_Newsgroups_Fixed"
        print(f"[GaliboStream] Loading: {HF_DATASET_ID}")

        ds_train = load_dataset(HF_DATASET_ID, split="train")
        ds_test = load_dataset(HF_DATASET_ID, split="test")

        texts_all = list(ds_train["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["label"]) + list(ds_test["label"])

        texts: List[str] = []
        labels_str: List[str] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            labels_str.append(str(lab))

        print(f"[GaliboStream] Kept {len(texts)} examples after dropping empty labels/texts.")

        unique_labels = sorted(set(labels_str))
        label2id = {lab: i for i, lab in enumerate(unique_labels)}
        labels = np.array([label2id[lab] for lab in labels_str], dtype=int)

        print(f"[GaliboStream] Unique labels: {len(unique_labels)}")

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        print(
            f"[GaliboStream] Sample-level change points: "
            f"{len(self.change_points)} / {len(labels)} samples "
            f"({len(self.change_points) / max(len(labels), 1):.4%})"
        )

        X = _encode_texts(texts, "GaliboStream")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class GaliboStreamSim(Dataset):
    def __init__(self, win_len):
        super().__init__(name="GaliboStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        HF_DATASET_ID = "galileo-ai/20_Newsgroups_Fixed"
        print(f"[GaliboStreamSim] Loading: {HF_DATASET_ID}")

        ds_train = load_dataset(HF_DATASET_ID, split="train")
        ds_test = load_dataset(HF_DATASET_ID, split="test")

        texts_all = list(ds_train["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["label"]) + list(ds_test["label"])

        texts: List[str] = []
        labels_str: List[str] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            labels_str.append(str(lab))

        print(f"[GaliboStreamSim] Kept {len(texts)} examples after dropping empty labels/texts.")

        unique_labels = sorted(set(labels_str))
        label2id = {lab: i for i, lab in enumerate(unique_labels)}
        labels = np.array([label2id[lab] for lab in labels_str], dtype=int)
        print(f"[GaliboStreamSim] Unique labels: {len(unique_labels)}")

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.80,
            revisit_frac=0.20,
            noise_between_frac=0.00,
            label_order="random",
            noise_burst_len=0,
            revisit_rounds=1,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        labels = labels[order_idx]

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        print(
            f"[GaliboStreamSim] Sample-level change points: "
            f"{len(self.change_points)} / {len(labels)} samples "
            f"({len(self.change_points) / max(len(labels), 1):.4%})"
        )

        X = _encode_texts(texts, "GaliboStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers




class AGNewsStream(Dataset):
    def __init__(self, win_len):
        super().__init__(name="AGNewsStream", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        HF_DATASET_ID = "ag_news"
        print(f"[AGNewsStream] Loading: {HF_DATASET_ID}")

        ds_train = load_dataset(HF_DATASET_ID, split="train")
        ds_test = load_dataset(HF_DATASET_ID, split="test")

        texts_all = list(ds_train["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["label"]) + list(ds_test["label"])

        texts: List[str] = []
        labels_list: List[int] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            labels_list.append(int(lab))

        print(f"[AGNewsStream] Kept {len(texts)} examples after dropping empty labels/texts.")

        labels = np.array(labels_list, dtype=int)
        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        X = _encode_texts(texts, "AGNewsStream")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class AGNewsStreamSim(Dataset):
    def __init__(self, win_len):
        super().__init__(name="AGNewsStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    @staticmethod
    def _simulate_agnews_label_cycles(
        labels: np.ndarray,
        seed: int = 42,
        segments_per_label: int = 15,
    ) -> np.ndarray:
        """Build a controlled recurring stream for AG News."""
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels).reshape(-1)
        unique_labels = np.unique(labels)
        chunks_by_label = {}

        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            idx = rng.permutation(idx)
            chunks_by_label[int(lab)] = np.array_split(idx, segments_per_label)

        order: List[int] = []
        prev_lab = None

        for r in range(segments_per_label):
            labs_this_round = list(rng.permutation(unique_labels))
            if prev_lab is not None and len(labs_this_round) > 1 and labs_this_round[0] == prev_lab:
                labs_this_round[0], labs_this_round[1] = labs_this_round[1], labs_this_round[0]

            for lab in labs_this_round:
                chunk = chunks_by_label[int(lab)][r]
                if len(chunk) > 0:
                    order.extend(chunk.tolist())
                    prev_lab = int(lab)

        return np.asarray(order, dtype=int)

    def load(self):
        HF_DATASET_ID = "ag_news"
        print(f"[AGNewsStreamSim] Loading: {HF_DATASET_ID}")

        ds_train = load_dataset(HF_DATASET_ID, split="train")
        ds_test = load_dataset(HF_DATASET_ID, split="test")

        texts_all = list(ds_train["text"]) + list(ds_test["text"])
        labels_all = list(ds_train["label"]) + list(ds_test["label"])

        texts: List[str] = []
        labels_list: List[int] = []
        for t, lab in zip(texts_all, labels_all):
            if t is None or lab is None:
                continue
            text = str(t).strip()
            if not text:
                continue
            texts.append(text)
            labels_list.append(int(lab))

        print(f"[AGNewsStreamSim] Kept {len(texts)} examples after dropping empty labels/texts.")

        labels = np.array(labels_list, dtype=int)
        print(f"[AGNewsStreamSim] Unique labels: {len(np.unique(labels))}")

        order_idx = self._simulate_agnews_label_cycles(
            labels=labels,
            seed=42,
            segments_per_label=15,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        labels = labels[order_idx]

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        print(
            f"[AGNewsStreamSim] Sample-level change points: "
            f"{len(self.change_points)} / {len(labels)} samples "
            f"({len(self.change_points) / max(len(labels), 1):.4%})"
        )
        print(f"[AGNewsStreamSim] First 30 change points: {self.change_points[:30]}")

        X = _encode_texts(texts, "AGNewsStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers



_MIND_CATEGORIES = ["sports", "politics", "finance", "entertainment", "technology"]
_MIND_CAT_NORM = {
    "sports": "sports", "sport": "sports",
    "politics": "politics", "news": "politics",
    "usnews": "politics", "worldnews": "politics",
    "finance": "finance", "money": "finance", "business": "finance",
    "entertainment": "entertainment", "movies": "entertainment",
    "music": "entertainment", "tv": "entertainment",
    "celebrity": "entertainment",
    "technology": "technology", "tech": "technology", "science": "technology",
}
_MIND_CAT2ID = {cat: i for i, cat in enumerate(_MIND_CATEGORIES)}


_MIND_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "mind"
)


def _download_mind_small(dest_dir: str) -> str:
    import urllib.request
    import zipfile

    CANDIDATE_URLS = [
        (
            "MIND-small (~50 MB, 50k users)",
            "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "MINDsmall_train.zip",
            "MINDsmall_train",
        ),
        (
            "MIND-demo  (~5 MB,  5k users — smaller but same format)",
            "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip",
            "MINDdemo_train.zip",
            "MINDdemo_train",
        ),
    ]

    for _, _, _, subdir in CANDIDATE_URLS:
        cached_tsv = os.path.join(dest_dir, subdir, "news.tsv")
        if os.path.exists(cached_tsv):
            print(f"[MIND] Using cached news.tsv: {cached_tsv}")
            return os.path.join(dest_dir, subdir)

    os.makedirs(dest_dir, exist_ok=True)
    last_error = None

    for label, url, zip_filename, subdir in CANDIDATE_URLS:
        zip_path = os.path.join(dest_dir, zip_filename)
        extract_dir = os.path.join(dest_dir, subdir)
        news_tsv = os.path.join(extract_dir, "news.tsv")

        print(f"\n[MIND] Trying {label}")
        print(f"       URL: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("[MIND] Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(zip_path)
            if os.path.exists(news_tsv):
                print(f"[MIND] Ready — news.tsv at {news_tsv}")
                return extract_dir
            print("[MIND] Extracted but news.tsv not found, trying next URL...")
        except Exception as e:
            last_error = e
            print(f"[MIND] Failed: {e} — trying next URL...")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    small_tsv = os.path.join(dest_dir, "MINDsmall_train", "news.tsv")
    raise RuntimeError(
        f"\n[MIND] All automatic downloads failed (last error: {last_error})\n"
        f"Please download manually from https://msnews.github.io\n"
        f"Target path: {small_tsv}\n"
    )


def _load_mind_articles(articles_per_category: int, num_cycles: int) -> Tuple[List[str], List[int]]:
    extract_dir = _download_mind_small(_MIND_CACHE_DIR)
    news_tsv = os.path.join(extract_dir, "news.tsv")

    print(f"[MIND] Parsing {news_tsv} ...")
    grouped: Dict[str, List[str]] = {cat: [] for cat in _MIND_CATEGORIES}
    seen_ids: set = set()
    skipped = 0

    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            news_id = parts[0]
            raw_cat = parts[1].lower().strip()
            title = parts[3].strip()
            abstract = parts[4].strip()

            if news_id in seen_ids:
                continue
            seen_ids.add(news_id)

            cat = _MIND_CAT_NORM.get(raw_cat)
            if cat is None:
                skipped += 1
                continue
            if not title or not abstract or len(abstract.split()) < 8:
                skipped += 1
                continue

            grouped[cat].append(f"{title}. {abstract}")

    counts = {cat: len(arts) for cat, arts in grouped.items()}
    print(f"[MIND] Articles by category: {counts}  (skipped/unmapped: {skipped})")

    available = {cat: n for cat, n in counts.items() if n > 0}
    if not available:
        raise RuntimeError("[MIND] No articles loaded.")

    per_cat = min(articles_per_category, min(available.values()))
    texts: List[str] = []
    labels: List[int] = []

    for cycle in range(num_cycles):
        for cat in _MIND_CATEGORIES:
            arts = grouped.get(cat, [])
            if not arts:
                continue
            start = (cycle * per_cat) % len(arts)
            block = [arts[(start + i) % len(arts)] for i in range(per_cat)]
            texts.extend(block)
            labels.extend([_MIND_CAT2ID[cat]] * len(block))

    print(f"[MIND] Stream built: {len(texts)} articles total.")
    return texts, labels


class MINDStream(Dataset):
    def __init__(self, win_len: int, articles_per_category: int = 500, num_cycles: int = 3):
        super().__init__(name="MINDStream", file_name="")
        self.win_len = int(max(1, win_len))
        self.articles_per_category = articles_per_category
        self.num_cycles = num_cycles
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels_list = _load_mind_articles(self.articles_per_category, self.num_cycles)
        labels = np.array(labels_list, dtype=int)

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        X = _encode_texts(texts, "MINDStream")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class MINDStreamSim(Dataset):
    def __init__(self, win_len: int, articles_per_category: int = 500, num_cycles: int = 3):
        super().__init__(name="MINDStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.articles_per_category = articles_per_category
        self.num_cycles = num_cycles
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels_list = _load_mind_articles(self.articles_per_category, self.num_cycles)
        labels = np.array(labels_list, dtype=int)

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.70,
            revisit_frac=0.10,
            noise_between_frac=0.08,
            label_order="random",
            noise_burst_len=6,
            revisit_rounds=2,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        labels = labels[order_idx]

        self.original_texts = texts.copy()
        self._answers = [""] * len(texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        X = _encode_texts(texts, "MINDStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers



_BEIR_HF_DATASETS = [
    ("mteb/msmarco", "BeIR/msmarco", "web_search", 0, 8000),
    ("mteb/hotpotqa", "BeIR/hotpotqa", "multihop_qa", 1, 8000),
    ("mteb/quora", "BeIR/quora", "duplicate_qa", 2, 5000),
    ("mteb/fever", "BeIR/fever", "factcheck", 3, 4000),
    ("mteb/fiqa", "BeIR/fiqa", "finance", 4, None),
    ("BeIR/nfcorpus", None, "biomedical", 5, None),
    ("mteb/nq", "BeIR/nq", "nq_qa", 6, None),
    ("mteb/cqadupstack-english", "BeIR/cqadupstack", "community_qa", 7, 2200),
    ("mteb/climate-fever", "BeIR/climate-fever", "climate", 8, None),
    ("mteb/arguana", "BeIR/arguana", "argument", 9, None),
    ("mteb/scifact", "BeIR/scifact", "science", 10, None),
    ("mteb/scidocs", "BeIR/scidocs", "citation", 11, None),
]


def _extract_queries_from_ds(ds, hf_id: str) -> List[str]:
    for col in ("text", "query", "sentence1", "question"):
        if col in ds.column_names:
            queries = [str(row[col]).strip() for row in ds if row.get(col)]
            queries = [q for q in queries if len(q) > 5]
            if queries:
                return queries

    for col in ds.column_names:
        if col.lower() in ("_id", "id", "query-id", "corpus-id", "score"):
            continue
        queries = [str(row[col]).strip() for row in ds if row.get(col)]
        queries = [q for q in queries if len(q) > 5]
        if queries:
            return queries

    return []


def _load_beir_queries_hf(hf_id, fallback_id, domain_label, domain_idx, per_cycle_cap):
    cap_str = f"cap={per_cycle_cap}/cycle" if per_cycle_cap else "no cap"

    def _try_load(did):
        try:
            ds = load_dataset(did, "queries", split="queries")
            qs = _extract_queries_from_ds(ds, did)
            if qs:
                return qs
        except Exception:
            pass

        for split in ("test", "train", "validation"):
            try:
                ds = load_dataset(did, split=split)
                qs = _extract_queries_from_ds(ds, did)
                if qs:
                    return qs
            except Exception:
                pass

        return []

    queries = _try_load(hf_id)
    if not queries and fallback_id:
        queries = _try_load(fallback_id)

    if queries:
        print(f"   ✅ {hf_id:<45} {len(queries):>7} queries  ({cap_str})")
        return queries, [domain_idx] * len(queries)

    print(f"   ⚠️  {hf_id:<45} FAILED")
    return [], []


def _build_beir_stream(domain_queries, num_cycles):
    all_texts: List[str] = []
    all_labels: List[int] = []

    for cycle in range(num_cycles):
        for _, _, _, domain_idx, per_cycle_cap in _BEIR_HF_DATASETS:
            if domain_idx not in domain_queries:
                continue
            qs = domain_queries[domain_idx]
            cap = per_cycle_cap if per_cycle_cap else len(qs)
            offset = (cycle * cap) % len(qs)
            block = [qs[(offset + i) % len(qs)] for i in range(min(cap, len(qs)))]
            all_texts.extend(block)
            all_labels.extend([domain_idx] * len(block))

    return all_texts, all_labels


class BEIRStream(Dataset):
    def __init__(self, win_len: int, num_cycles: int = 3):
        super().__init__(name="BEIRStream", file_name="")
        self.win_len = int(max(1, win_len))
        self.num_cycles = num_cycles
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        print(f"[BEIRStream] Loading {len(_BEIR_HF_DATASETS)} BEIR domains...")
        domain_queries: Dict[int, List[str]] = {}
        for hf_id, fallback_id, domain_label, domain_idx, per_cycle_cap in _BEIR_HF_DATASETS:
            qs, _ = _load_beir_queries_hf(hf_id, fallback_id, domain_label, domain_idx, per_cycle_cap)
            if qs:
                domain_queries[domain_idx] = qs

        if not domain_queries:
            self._data = (np.zeros((0, 384)), np.zeros((0, 1)), np.zeros((0, 384)), np.zeros((0, 1)), np.zeros((0, 2)))
            return

        all_texts, all_labels = _build_beir_stream(domain_queries, self.num_cycles)
        labels = np.array(all_labels, dtype=int)
        self.original_texts = all_texts.copy()
        self._answers = [""] * len(all_texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        X = _encode_texts(all_texts, "BEIRStream")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class BEIRStreamSim(Dataset):
    def __init__(self, win_len: int, num_cycles: int = 2):
        super().__init__(name="BEIRStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.num_cycles = num_cycles
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        print(f"[BEIRStreamSim] Loading {len(_BEIR_HF_DATASETS)} BEIR domains...")
        domain_queries: Dict[int, List[str]] = {}
        for hf_id, fallback_id, domain_label, domain_idx, per_cycle_cap in _BEIR_HF_DATASETS:
            qs, _ = _load_beir_queries_hf(hf_id, fallback_id, domain_label, domain_idx, per_cycle_cap)
            if qs:
                domain_queries[domain_idx] = qs

        if not domain_queries:
            self._data = (np.zeros((0, 384)), np.zeros((0, 1)), np.zeros((0, 384)), np.zeros((0, 1)), np.zeros((0, 2)))
            return

        all_texts, all_labels = _build_beir_stream(domain_queries, self.num_cycles)
        labels = np.array(all_labels, dtype=int)

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.70,
            revisit_frac=0.10,
            noise_between_frac=0.08,
            label_order="random",
            noise_burst_len=6,
            revisit_rounds=2,
        )

        all_texts = [all_texts[i] for i in order_idx.tolist()]
        labels = labels[order_idx]

        self.original_texts = all_texts.copy()
        self._answers = [""] * len(all_texts)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)

        X = _encode_texts(all_texts, "BEIRStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers




_RAG_CLEANED_DIR = base_path / "src" / "data" / "full_rag_csv" / "cleaned_question_answer_label"


def _clean_string_value(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    x = str(x).strip()
    if x.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return None
    return x


def _read_rag_csv_robust(csv_path: Path, stream_name: str) -> pd.DataFrame:
    """Robust CSV reader for local RAG-style datasets."""
    print(f"[{stream_name}] Trying robust pandas CSV read...")

    try:
        df = pd.read_csv(
            csv_path,
            dtype=str,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="skip",
            engine="python",
            skipinitialspace=True,
        )
        df.columns = df.columns.str.strip()
        print(f"[{stream_name}] Pandas read successful.")
        print(f"[{stream_name}] Columns found: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"[{stream_name}] Pandas read failed with: {type(e).__name__}: {e}")
        print(f"[{stream_name}] Falling back to permissive line-based reader...")

    import csv

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"[{stream_name}] Empty CSV file: {csv_path}")

        header = [h.strip() for h in header_line.strip().split(",")]
        print(f"[{stream_name}] Raw header from fallback reader: {header}")
        has_dataset_subclass = "dataset_subclass" in header

        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue

            try:
                parsed = next(csv.reader([line], skipinitialspace=True))
            except Exception:
                parsed = line.split(",")

            parsed = [str(x).strip().strip('"') for x in parsed]

            if has_dataset_subclass:
                if len(parsed) < 4:
                    continue
                question = parsed[0]
                label = parsed[-1]
                dataset_subclass = parsed[-2]
                answer = ",".join(parsed[1:-2])
                rows.append({"question": question, "answer": answer, "dataset_subclass": dataset_subclass, "label": label})
            else:
                if len(parsed) < 3:
                    continue
                question = parsed[0]
                label = parsed[-1]
                answer = ",".join(parsed[1:-1])
                rows.append({"question": question, "answer": answer, "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"[{stream_name}] Fallback reader produced an empty dataframe: {csv_path}")

    df.columns = df.columns.str.strip()
    print(f"[{stream_name}] Fallback reader successful. Rows recovered: {len(df):,}")
    print(f"[{stream_name}] Columns found: {df.columns.tolist()}")
    return df


def _load_local_rag_qa_dataset(
    file_name: str,
    stream_name: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    csv_path = _RAG_CLEANED_DIR / file_name

    print(f"\n[{stream_name}] Loading cleaned file:")
    print(f"[{stream_name}] {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[{stream_name}] Cleaned dataset not found: {csv_path}\n"
            f"Expected cleaned file inside: {_RAG_CLEANED_DIR}"
        )

    df = _read_rag_csv_robust(csv_path, stream_name)
    df.columns = df.columns.str.strip()

    required_cols = ["question", "answer", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{stream_name}] Missing required columns: {missing}\n"
            f"Available columns are: {df.columns.tolist()}\n"
            f"File read from: {csv_path}"
        )

    df = df[required_cols].copy()
    for col in required_cols:
        df[col] = df[col].map(_clean_string_value)

    df = df.dropna(subset=["question", "label"]).reset_index(drop=True)
    df["answer"] = df["answer"].fillna("").astype(str)

    if len(df) == 0:
        raise ValueError(f"[{stream_name}] Dataset is empty after cleaning.")

    print("\n=== Label check before sampling/cap ===")
    print(f"[{stream_name}] Rows before cap: {len(df):,}")
    print(f"[{stream_name}] Unique labels before cap: {df['label'].nunique():,}")
    print(df["label"].value_counts())
    print("=======================================\n")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    print("\n=== Label check after sampling/cap ===")
    print(f"[{stream_name}] Rows kept: {len(df):,}")
    print(f"[{stream_name}] Unique labels after cap: {df['label'].nunique():,}")
    print(df["label"].value_counts())
    print("======================================\n")

    raw_labels = df["label"].astype(str).tolist()
    unique_labels = sorted(set(raw_labels))

    if len(unique_labels) < 2:
        raise ValueError(f"[{stream_name}] Needs at least 2 labels, found {len(unique_labels)}: {unique_labels}")

    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    labels = np.array([label2id[lab] for lab in raw_labels], dtype=int)

    texts = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    print(f"[{stream_name}] Encoded label mapping:")
    for lab_name, lab_id in label2id.items():
        print(f"  {lab_id}: {lab_name}")

    answered = sum(1 for a in answers if a.strip())
    print(f"[{stream_name}] Gold answers: {answered:,} / {len(answers):,}")

    return texts, labels, answers


def _print_gt_summary(stream_name: str, labels_int: np.ndarray, drift_mat: np.ndarray):
    N = len(labels_int)
    n_cp = int(drift_mat[:, 1].sum()) if drift_mat.ndim == 2 and drift_mat.shape[1] > 1 else 0
    n_pos = int(drift_mat[:, 0].sum()) if drift_mat.ndim == 2 else 0
    print(f"\n[{stream_name} GT]")
    print(f"  N={N}  unique_labels={len(np.unique(labels_int))}")
    print(f"  change points: {n_cp}")
    print(f"  drift_range positives: {n_pos}/{N} ({n_pos / max(1, N) * 100:.1f}%)")
    print(f"  drift_mat shape: {drift_mat.shape}")


def _group_blocks_with_optional_revisit(
    labels: np.ndarray,
    seed: int = 42,
    main_frac: float = 1.0,
    revisit: bool = False,
) -> np.ndarray:
    """Group local QA data by label; optionally split into main/revisit passes."""
    rng = random.Random(seed)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(np.asarray(labels, dtype=int).reshape(-1).tolist()):
        grouped[int(lab)].append(i)

    unique_labels = list(grouped.keys())
    rng.shuffle(unique_labels)
    for lab in unique_labels:
        rng.shuffle(grouped[lab])

    order: List[int] = []

    if not revisit:
        for lab in unique_labels:
            order.extend(grouped[lab])
    else:
        splits: Dict[int, int] = {}
        for lab in unique_labels:
            split = int(len(grouped[lab]) * main_frac)
            splits[lab] = split
            order.extend(grouped[lab][:split])

        revisit_labels = unique_labels.copy()
        rng.shuffle(revisit_labels)
        for lab in revisit_labels:
            split = splits[lab]
            order.extend(grouped[lab][split:])

    seen = set()
    clean: List[int] = []
    for i in order:
        if i not in seen:
            clean.append(i)
            seen.add(i)

    if len(clean) != len(labels):
        # Append any missing rows, grouped by label, to avoid data loss.
        for lab in unique_labels:
            for i in grouped[lab]:
                if i not in seen:
                    clean.append(i)
                    seen.add(i)

    return np.asarray(clean, dtype=int)


class KILTStreamSim(Dataset):
    def __init__(self, win_len, max_samples: Optional[int] = None):
        super().__init__(name="KILTStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.max_samples = max_samples
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels, answers = _load_local_rag_qa_dataset(
            file_name="KILT_all_domains_question_answer_lable.csv",
            stream_name="KILTStreamSim",
            max_samples=self.max_samples,
            seed=42,
        )

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.90,
            revisit_frac=0.08,
            noise_between_frac=0.00,
            label_order="random",
            noise_burst_len=1,
            revisit_rounds=1,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        answers = [answers[i] for i in order_idx.tolist()]
        labels = labels[order_idx].astype(int).reshape(-1)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)
        _print_gt_summary("KILTStreamSim", labels, drift_mat)

        self.original_texts = texts.copy()
        self._answers = answers.copy()

        X = _encode_texts(texts, "KILTStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class MultiHopRAGStreamSim(Dataset):
    def __init__(self, win_len, max_samples: Optional[int] = None):
        super().__init__(name="MultiHopRAGStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.max_samples = max_samples
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels, answers = _load_local_rag_qa_dataset(
            file_name="MultiHopRAG_full_question_answer_lable.csv",
            stream_name="MultiHopRAGStreamSim",
            max_samples=self.max_samples,
            seed=42,
        )

        order = _group_blocks_with_optional_revisit(labels, seed=42, main_frac=0.80, revisit=True)
        texts = [texts[i] for i in order.tolist()]
        answers = [answers[i] for i in order.tolist()]
        labels = labels[order].astype(int).reshape(-1)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)
        _print_gt_summary("MultiHopRAGStreamSim", labels, drift_mat)

        self.original_texts = texts.copy()
        self._answers = answers.copy()

        X = _encode_texts(texts, "MultiHopRAGStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class PopQAStreamSim(Dataset):
    def __init__(self, win_len, max_samples: Optional[int] = None):
        super().__init__(name="PopQAStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.max_samples = max_samples
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels, answers = _load_local_rag_qa_dataset(
            file_name="PopQA_full_question_answer_lable.csv",
            stream_name="PopQAStreamSim",
            max_samples=self.max_samples,
            seed=42,
        )

        order = _group_blocks_with_optional_revisit(labels, seed=42, revisit=False)
        texts = [texts[i] for i in order.tolist()]
        answers = [answers[i] for i in order.tolist()]
        labels = labels[order].astype(int).reshape(-1)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)
        _print_gt_summary("PopQAStreamSim", labels, drift_mat)

        self.original_texts = texts.copy()
        self._answers = answers.copy()

        X = _encode_texts(texts, "PopQAStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers


class RAGBenchStreamSim(Dataset):
    def __init__(self, win_len, max_samples: Optional[int] = None):
        super().__init__(name="RAGBenchStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.max_samples = max_samples
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels, answers = _load_local_rag_qa_dataset(
            file_name="RAGBench_full_question_answer_lable.csv",
            stream_name="RAGBenchStreamSim",
            max_samples=self.max_samples,
            seed=42,
        )

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.90,
            revisit_frac=0.08,
            noise_between_frac=0.00,
            label_order="random",
            noise_burst_len=1,
            revisit_rounds=1,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        answers = [answers[i] for i in order_idx.tolist()]
        labels = labels[order_idx].astype(int).reshape(-1)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)
        _print_gt_summary("RAGBenchStreamSim", labels, drift_mat)

        self.original_texts = texts.copy()
        self._answers = answers.copy()

        X = _encode_texts(texts, "RAGBenchStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers

class StateCacheQAStreamSim(Dataset):
    

    def __init__(self, win_len, max_samples: Optional[int] = None):
        super().__init__(name="StateCacheQAStreamSim", file_name="")
        self.win_len = int(max(1, win_len))
        self.max_samples = max_samples
        self.original_texts: List[str] = []
        self._answers: List[str] = []
        self.change_points: List[int] = []

    def load(self):
        texts, labels, answers = _load_local_rag_qa_dataset(
            file_name="StateCacheQA_question_answer_label.csv",
            stream_name="StateCacheQAStreamSim",
            max_samples=self.max_samples,
            seed=42,
        )

        order_idx, _ = _simulate_reordered_stream(
            labels=labels,
            win_len=self.win_len,
            seed=42,
            block_frac=0.80,
            revisit_frac=0.20,
            noise_between_frac=0.00,
            label_order="random",
            noise_burst_len=0,
            revisit_rounds=1,
        )

        texts = [texts[i] for i in order_idx.tolist()]
        answers = [answers[i] for i in order_idx.tolist()]
        labels = labels[order_idx].astype(int).reshape(-1)

        drift_mat = _make_drift_matrix_from_labels(labels, self.win_len)
        self.change_points = _change_points_from_drift_matrix(drift_mat)
        _print_gt_summary("StateCacheQAStreamSim", labels, drift_mat)

        self.original_texts = texts.copy()
        self._answers = answers.copy()

        X = _encode_texts(texts, "StateCacheQAStreamSim")
        self._data = _pack_stream_data(X, labels, drift_mat)

    def get_original_texts(self) -> List[str]:
        return self.original_texts

    def get_ground_truth_answers(self) -> List[str]:
        return self._answers
