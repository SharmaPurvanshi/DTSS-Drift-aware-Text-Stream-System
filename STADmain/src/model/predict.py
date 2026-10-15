from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import os
import csv
import json
import re
import string
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import v_measure_score
from tqdm.auto import tqdm
from .drift_detection import KLDivergence, AlibiKSTest, MMDTest, EnergyTest


_STATE_CACHE_DEFAULT_ENV = {
    "DTSS_STATE_CACHE_FILTERING": "1",
    "DTSS_STATE_CACHE_BUILD_FRAC": "0.40",
    "DTSS_STATE_CACHE_MAX_BUILD_QUERIES": "25",
    "DTSS_STATE_CACHE_MIN_BUILD_QUERIES": "3",
    "DTSS_STATE_CAG_RETRIEVE_TOPK": "5",
    "DTSS_STATE_CAG_MAX_DOCS": "20",
    "DTSS_STATE_CACHE_QUERY_TOPK": "5",
    "DTSS_STATE_CACHE_MIN_SIM": "0.25",
    "DTSS_STATE_CACHE_MIN_DOCS": "1",
    "DTSS_STATE_CACHE_DOC_EMB_CHARS": "1200",
    "DTSS_USE_TRUE_KV": "0",
    "DTSS_USE_STATE_TOPIC_CACHE": "1",
    "DTSS_STATE_TOPIC_SOURCE": "primary",
    "DTSS_STATE_TOPIC_MAX_QUERIES": "3",
    "DTSS_STATE_TOPIC_RETRIEVE_TOPK": "5",
    "DTSS_STATE_TOPIC_TOP_TERMS": "10",
    "DTSS_STATE_TOPIC_REPRESENTATIVE_K": "3",
    "DTSS_FRESH_RAG_TOPK": "5",
    "DTSS_MIN_RAG_STATE_QUERIES": "5",
}
for _k, _v in _STATE_CACHE_DEFAULT_ENV.items():
    os.environ.setdefault(_k, _v)


class Predict:
    def __init__(
        self,
        experiment_dir,
        hidden_dim,
        batch_size,
        dropout_rate,
        epochs,
        early_stopping_patience,
        learning_rate,
        weight_decay,
        window_size,
        steps,
        input_dim,
        l_hist_size,
        l_new_size,
        ks_alpha,
        ks_hist_min_size,
        gpu=None,
        drift_mode: str = "mmd",
        lambda_mmd: float = 0.005,
        lambda_energy: Optional[float] = None,
        homo_threshold: float = 0.8,
        fuse_mode: str = "or",
        state_sim_threshold: float = 0.90,
        state_update_alpha: float = 0.05,
        force_new_state_on_alarm: bool = False,
        merge_every: int = 200,
        merge_threshold: float = 0.95,
        min_state_windows: int = 10,
        log_every: int = 10,
        max_log_lines: int = 200000,
        normalize_window: bool = True,
        normalize_for_detectors: bool = False,
        event_tolerance_k: int = 3,
        event_window_W: int = 10,
        event_fp_penalty: float = 0.11,
        state_penalty_lambda: float = 1.0,
        state_logic: str = "cosine",
        max_state_num: int = 50,
        enable_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        num_test_queries: int = 10,
        rag_random_seed: int = 42,
        strict_knowledge_base: bool = True,
        filter_question_like_kb: bool = False,
        pointwise_tolerance_k: int = 1,
        alarm_cooldown_k: int = 0,
    ):
        self.experiment_dir = experiment_dir
        self.hidden_dim = hidden_dim
        self.batch_size = int(batch_size)
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = int(window_size)
        self.steps = int(steps)
        self.input_dim = int(input_dim)
        self.l_hist_size = int(l_hist_size)
        self.l_new_size = int(l_new_size)
       
        self.detector_alpha = float(ks_alpha)
        self.ks_alpha = self.detector_alpha
        self.ks_hist_min_size = int(ks_hist_min_size)
        self.gpu = gpu
        self.drift_mode = str(drift_mode).lower().strip()
        self.lambda_mmd = float(lambda_mmd)
        self.lambda_energy = float(lambda_mmd if lambda_energy is None else lambda_energy)
        self.homo_threshold = float(homo_threshold)
        self.fuse_mode = str(fuse_mode).lower().strip()
        self.state_sim_threshold = float(state_sim_threshold)
        self.state_update_alpha = float(state_update_alpha)
        self.force_new_state_on_alarm = bool(force_new_state_on_alarm)
        self.merge_every = int(max(1, merge_every))
        self.merge_threshold = float(merge_threshold)
        self.min_state_windows = int(max(1, min_state_windows))
        self.log_every = int(max(1, log_every))
        self.max_log_lines = int(max_log_lines)
        self.normalize_window = bool(normalize_window)
        self.normalize_for_detectors = bool(normalize_for_detectors)
        self.event_tolerance_k = int(max(0, event_tolerance_k))
        pointwise_env = os.environ.get("DTSS_POINTWISE_TOLERANCE_K")
        if pointwise_env is not None and str(pointwise_env).strip() != "":
            pointwise_tolerance_k = int(float(pointwise_env))
        self.pointwise_tolerance_k = int(max(0, pointwise_tolerance_k))
        cooldown_env = os.environ.get("DTSS_ALARM_COOLDOWN_K")
        if cooldown_env is not None and str(cooldown_env).strip() != "":
            alarm_cooldown_k = int(float(cooldown_env))
        self.alarm_cooldown_k = int(max(0, alarm_cooldown_k))

        self.event_window_W = int(max(1, event_window_W))
        self.event_fp_penalty = float(event_fp_penalty)
        self.state_penalty_lambda = float(state_penalty_lambda)
        self.state_logic = str(state_logic)
        self.max_state_num = int(max_state_num)
        self.enable_rag = enable_rag
        self.num_test_queries = int(num_test_queries)
        self.rag_random_seed = int(rag_random_seed)
        self.strict_knowledge_base = bool(strict_knowledge_base)
        self.filter_question_like_kb = bool(filter_question_like_kb)
        self.original_texts = None
        self.rag_answers = []
        self.retriever = None
        self.fresh_retriever = None
        self.knowledge_cache = None
        self.generator = None
        self.answer_gen = None
        self.comparator = None

        if self.enable_rag:
            try:
                from .rag_components import (
                    KnowledgeRetriever, StateKnowledgeCache,
                    HuggingFaceGenerator,
                )

                try:
                    from .answer_generator import AnswerGenerator
                except Exception as e:
                    AnswerGenerator = None
                    print(f" AnswerGenerator import failed: {e}. Using generator fallback.")

                try:
                    from .rag_evaluation import AnswerComparator
                    self.comparator = AnswerComparator()
                except Exception as e:
                    self.comparator = None
                    print(f" AnswerComparator import failed: {e}. ROUGE aggregate may be skipped.")

                self.retriever = KnowledgeRetriever(
                    knowledge_base_path=knowledge_base_path or "data/knowledge_base.json",
                    top_k=5,
                    strict=self.strict_knowledge_base,
                    filter_question_like=self.filter_question_like_kb,
                )

                # WebRetriever not available here — fresh RAG uses the same FAISS retriever.
                self.fresh_retriever = None
                print("  Fresh RAG: using KnowledgeRetriever/FAISS fallback")

                self.knowledge_cache = StateKnowledgeCache()

                # If gpu is provided, respect that exact CUDA device (e.g. gpu=1 -> cuda:1).
                # If gpu is None, let the generator auto-select CUDA when available.
                device = f"cuda:{int(self.gpu)}" if self.gpu is not None else None

                # Main generator used for normal generation and KV-cache generation.
                self.generator = HuggingFaceGenerator(
                    model_name="microsoft/Phi-3.5-mini-instruct",
                    device=device,
                    max_new_tokens=100,
                )

                # Reuse the already-loaded generator to avoid loading Phi twice.
                self.answer_gen = None
                if AnswerGenerator is not None:
                    try:
                        try:
                            self.answer_gen = AnswerGenerator(generator=self.generator)
                        except TypeError:
                            # Backward compatibility with older AnswerGenerator signature.
                            self.answer_gen = AnswerGenerator(
                                model_name="microsoft/Phi-3.5-mini-instruct",
                                device=device,
                                max_new_tokens=100,
                            )
                    except Exception as e:
                        print(f" AnswerGenerator init failed: {e}. Using generator fallback.")
                        self.answer_gen = None

                print("RAG pipeline enabled:")
                print("   Primary retrieval : KnowledgeRetriever (FAISS)")
                print("   Fresh RAG         : KnowledgeRetriever (FAISS fallback)")
                print("   Generation        : Phi-3.5-mini-instruct")
                print("   Cached RAG        : state cached context by default")
                print("                       query-filtered state cache by default; set DTSS_STATE_CACHE_FILTERING=0 for legacy full-cache ablation")

            except Exception as e:
                print(f"RAG init failed: {e}. Disabling RAG.")
                import traceback
                traceback.print_exc()
                self.enable_rag = False

        os.makedirs(self.experiment_dir, exist_ok=True)

    def set_original_texts(self, texts: List[str]):
        self.original_texts = texts
        print(f"Set {len(texts)} original texts for RAG")

    def set_ground_truth_answers(self, answers: List[str]):
        self.ground_truth_answers = answers
        print(f"Set {len(answers)} ground truth answers for RAG")

    @staticmethod
    def _l2_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x / (np.linalg.norm(x) + eps)

    @staticmethod
    def _l2_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    @staticmethod
    def _safe_auc(y_true_arr, scores_arr):
        scores_arr = np.asarray(scores_arr, dtype=float)
        y_true_arr = np.asarray(y_true_arr, dtype=int)
        mask = np.isfinite(scores_arr)
        if mask.sum() < 2:
            return None
        y_f, s_f = y_true_arr[mask], scores_arr[mask]
        if np.unique(y_f).size < 2:
            return None
        try:
            return float(metrics.roc_auc_score(y_f, s_f))
        except Exception:
            return None

    @staticmethod
    def _safe_pr_auc(y_true_arr, scores_arr):
        scores_arr = np.asarray(scores_arr, dtype=float)
        y_true_arr = np.asarray(y_true_arr, dtype=int)
        mask = np.isfinite(scores_arr)
        if mask.sum() < 2:
            return None
        y_f, s_f = y_true_arr[mask], scores_arr[mask]
        if np.unique(y_f).size < 2:
            return None
        try:
            return float(metrics.average_precision_score(y_f, s_f))
        except Exception:
            return None

    @staticmethod
    def _softmax_hist(X: np.ndarray):
        if X is None or len(X) == 0:
            return []
        return softmax(pd.DataFrame(X), axis=1).mean(axis=0).tolist()

    @staticmethod
    def _safe_cov(X: np.ndarray, eps: float = 1e-6):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        if X.shape[0] < 2:
            return eps * np.eye(d)
        S = np.cov(X, rowvar=False)
        if np.ndim(S) == 0:
            return eps * np.eye(d)
        return S + eps * np.eye(d)

    @staticmethod
    def _append_samples(buf: dict, sid: int, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        buf[sid] = X if sid not in buf else np.concatenate([buf[sid], X], axis=0)

    @staticmethod
    def _is_binary_drift_labels(drift_labels: np.ndarray) -> bool:
        uniq = np.unique(np.asarray(drift_labels).astype(int))
        return uniq.size <= 2 and np.all((uniq == 0) | (uniq == 1))

    def _build_windows(self, X: np.ndarray, y: np.ndarray, drift: np.ndarray):
        win = self.window_size
        st  = self.steps
        XY  = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        seqs  = [XY[i:i + win]    for i in range(0, XY.shape[0]    - win + 1, st)]
        drift = np.asarray(drift, dtype=int)
        if drift.ndim == 1:
            dseqs = [drift[i:i + win]    for i in range(0, drift.shape[0] - win + 1, st)]
        else:
            dseqs = [drift[i:i + win, :] for i in range(0, drift.shape[0] - win + 1, st)]
        return seqs, dseqs

    @staticmethod
    def _change_points_from_drift_labels(drift: np.ndarray) -> np.ndarray:
        drift = np.asarray(drift, dtype=int).reshape(-1)
        if drift.size == 0:
            return np.array([], dtype=int)
        idx = np.where(drift == 1)[0]
        if idx.size == 0:
            return np.array([], dtype=int)
        keep = [int(idx[0])]
        for i in range(1, idx.size):
            if idx[i] != idx[i - 1] + 1:
                keep.append(int(idx[i]))
        return np.asarray(keep, dtype=int)

    @staticmethod
    def _cp_samples_to_cp_windows(cp_samples: np.ndarray, win: int, step: int, N: int) -> np.ndarray:
        cp_samples = np.asarray(cp_samples, dtype=int).reshape(-1)
        if cp_samples.size == 0 or N < win:
            return np.array([], dtype=int)
        num_w = int(np.floor((N - win) / step) + 1)
        w = np.ceil((cp_samples - (win - 1)) / float(step)).astype(int)
        w = np.maximum(w, 0)
        w = w[w < num_w]
        return np.unique(w)

    def _build_window_gt_from_cp(
        self,
        cp_windows: np.ndarray,
        num_w: int,
        tolerance_k: int,
        shift_by_lnew: bool = False,
    ) -> np.ndarray:
        
        gt = np.zeros(num_w, dtype=int)
        tolerance_k = int(max(0, tolerance_k))

        for cp in cp_windows:
            center = int(cp)
            if shift_by_lnew:
                center += int(self.l_new_size)

            lo = max(0, center - tolerance_k)
            hi = min(num_w - 1, center + tolerance_k)
            gt[lo:hi + 1] = 1

        return gt

    @staticmethod
    def _event_eval(alarm: np.ndarray, event_w: np.ndarray, k: int, W: int, fp_penalty: float) -> dict:
        alarm   = np.asarray(alarm,   dtype=int).reshape(-1)
        event_w = np.asarray(event_w, dtype=int).reshape(-1)
        num_alarms = int(alarm.sum())
        alarm_idx  = np.where(alarm == 1)[0]

        if event_w.size == 0:
            return {
                "num_events": 0, "num_alarms": num_alarms,
                "event_precision": 0.0 if num_alarms > 0 else None,
                "event_recall": None, "event_f1": None,
                "mean_delay": None, "median_delay": None,
                "nab_score": None, "false_alarms": int(num_alarms),
            }

        matched_event = np.zeros(event_w.size, dtype=bool)
        delays = []
        for j, t in enumerate(event_w):
            in_tol = (alarm_idx >= int(t)) & (alarm_idx <= int(t + k))
            if np.any(in_tol):
                matched_event[j] = True
                delays.append(int(alarm_idx[in_tol].min()) - int(t))

        TP_events      = int(matched_event.sum())
        event_recall   = TP_events / float(event_w.size)
        alarm_matched  = np.zeros(alarm_idx.size, dtype=bool)
        for i, a in enumerate(alarm_idx):
            for t in event_w:
                if int(t) <= int(a) <= int(t + k):
                    alarm_matched[i] = True
                    break
        matched_alarms  = int(alarm_matched.sum())
        event_precision = (matched_alarms / float(num_alarms)) if num_alarms > 0 else 0.0
        event_f1 = (
            2.0 * event_precision * event_recall / (event_precision + event_recall)
            if (event_precision + event_recall) > 0 else 0.0
        )
        nab = 0.0
        for t in event_w:
            t = int(t)
            idx = alarm_idx[(alarm_idx >= t) & (alarm_idx <= t + W)]
            if idx.size > 0:
                nab += max(0.0, 1.0 - (int(idx.min()) - t) / float(W))
        fp = sum(
            1 for a in alarm_idx
            if not any(int(t) <= int(a) <= int(t + W) for t in event_w)
        )
        nab      -= float(fp_penalty) * float(fp)
        nab_score = float(nab / max(1, event_w.size))

        return {
            "num_events": int(event_w.size), "num_alarms": int(num_alarms),
            "event_precision": float(event_precision), "event_recall": float(event_recall),
            "event_f1": float(event_f1),
            "mean_delay": float(np.mean(delays)) if delays else None,
            "median_delay": float(np.median(delays)) if delays else None,
            "nab_score": nab_score, "false_alarms": int(fp),
        }

    @staticmethod
    def _uf_find(parent: Dict[int, int], x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return parent.get(x, x)

    @staticmethod
    def _uf_union(parent: Dict[int, int], a: int, b: int) -> int:
        ra, rb = Predict._uf_find(parent, a), Predict._uf_find(parent, b)
        if ra == rb:
            return ra
        parent[rb] = ra
        return ra

    def _handle_drift_state_transition(self, z, state_protos, state_counts, uf_parent):
        
        if len(state_protos) == 0:
            state_protos[0] = z.copy()
            state_counts[0] = 0
            uf_parent[0] = 0
            return 0, 1.0, "first_state"

        if self.force_new_state_on_alarm and len(state_protos) < self.max_state_num:
            new_sid = int(max(state_protos.keys()) + 1)
            state_protos[new_sid] = z.copy()
            state_counts[new_sid] = 0
            uf_parent[new_sid] = new_sid
            return new_sid, -1.0, "force_create_new"

        sids, protos = [], []
        for sid in list(state_protos.keys()):
            canon = self._uf_find(uf_parent, sid)
            if canon != sid:
                continue
            sids.append(canon)
            protos.append(state_protos[canon])

        if not sids:
            state_protos.clear()
            state_counts.clear()
            uf_parent.clear()
            state_protos[0] = z.copy()
            state_counts[0] = 0
            uf_parent[0] = 0
            return 0, 1.0, "first_state"

        protos = np.stack(protos, axis=0)
        z_n = self._l2_norm(z)
        sims = self._l2_norm_rows(protos) @ z_n
        best_i = int(np.argmax(sims))
        best_sid = int(sids[best_i])
        best_sim = float(sims[best_i])

        if getattr(self, "log_every", 10) <= 10:
            print(
                f"[COSINE STATE DEBUG] best_sim={best_sim:.6f}, "
                f"threshold={self.state_sim_threshold}, "
                f"num_states={len(state_protos)}"
            )

        if best_sim >= self.state_sim_threshold:
            return best_sid, best_sim, "revisit_old"

        if len(state_protos) >= self.max_state_num:
            return best_sid, best_sim, "max_state_reuse"

        new_sid = int(max(state_protos.keys()) + 1)
        state_protos[new_sid] = z.copy()
        state_counts[new_sid] = 0
        uf_parent[new_sid] = new_sid
        return new_sid, best_sim, "create_new"

    def _handle_no_drift_state_assignment(self, z, current_state, state_protos, state_counts, uf_parent):
        """Cosine-based assignment when there is no drift alarm."""
        if current_state in state_protos:
            canon = self._uf_find(uf_parent, current_state)
            if canon in state_protos:
                z_n = self._l2_norm(z)
                sim = float(np.dot(z_n, self._l2_norm(state_protos[canon])))
                return canon, sim, "stay_current"

        if not state_protos:
            state_protos[0] = z.copy()
            state_counts[0] = 0
            uf_parent[0] = 0
            return 0, 1.0, "fallback_new"

        sids, protos = [], []
        for sid in state_protos.keys():
            canon = self._uf_find(uf_parent, sid)
            if canon != sid:
                continue
            sids.append(canon)
            protos.append(state_protos[canon])

        if not sids:
            state_protos[0] = z.copy()
            state_counts[0] = 0
            uf_parent[0] = 0
            return 0, 1.0, "fallback_new"

        protos = np.stack(protos, axis=0)
        z_n = self._l2_norm(z)
        sims = self._l2_norm_rows(protos) @ z_n
        best_i = int(np.argmax(sims))
        return int(sids[best_i]), float(sims[best_i]), "fallback_match"

    # ══════════════════════════════════════════════════════════════════
    # KLD-BASED STATE ASSIGNMENT  (original STAD — Pi + KL divergence)
    # ══════════════════════════════════════════════════════════════════

    def _handle_drift_state_transition_kld(
        self, Xw, state_Pi_kld, state_counts_kld, uf_parent_kld, kld_epsilon
    ):
       
        pi_new = self._stad_density_estimation(Xw)

        if len(state_Pi_kld) == 0:
            state_Pi_kld[0] = pi_new.copy()
            state_counts_kld[0] = 0
            uf_parent_kld[0] = 0
            return 0, 0.0, "first_state"

        kld_scores = {}
        for sid, pi_stored in state_Pi_kld.items():
            canon = self._uf_find(uf_parent_kld, sid)
            if canon != sid:
                continue
            kld_scores[canon] = self._symmetric_kld_stad(pi_stored, pi_new)

        if not kld_scores:
            new_sid = 0
            state_Pi_kld[new_sid] = pi_new.copy()
            state_counts_kld[new_sid] = 0
            uf_parent_kld[new_sid] = new_sid
            return new_sid, 0.0, "first_state"

        best_sid = min(kld_scores, key=kld_scores.get)
        best_kld = float(kld_scores[best_sid])

        if best_kld < float(kld_epsilon):
            return int(best_sid), best_kld, "revisit_old"

       
        if len(state_Pi_kld) >= self.max_state_num:
            return int(best_sid), best_kld, "max_state_reuse"

        new_sid = int(max(state_Pi_kld.keys()) + 1)
        state_Pi_kld[new_sid] = pi_new.copy()
        state_counts_kld[new_sid] = 0
        uf_parent_kld[new_sid] = new_sid
        return new_sid, best_kld, "create_new"

    def _handle_no_drift_state_assignment_kld(
        self, Xw, current_state_kld, state_Pi_kld, state_counts_kld, uf_parent_kld
    ):
     
        if current_state_kld in state_Pi_kld:
            canon = self._uf_find(uf_parent_kld, current_state_kld)
            if canon in state_Pi_kld:
                pi_new = self._stad_density_estimation(Xw)
                kld = self._symmetric_kld_stad(state_Pi_kld[canon], pi_new)
                return canon, kld, "stay_current"

        if not state_Pi_kld:
            pi_new = self._stad_density_estimation(Xw)
            state_Pi_kld[0] = pi_new.copy()
            state_counts_kld[0] = 0
            uf_parent_kld[0] = 0
            return 0, 0.0, "fallback_new"

        pi_new = self._stad_density_estimation(Xw)
        kld_scores = {}
        for sid in state_Pi_kld:
            canon = self._uf_find(uf_parent_kld, sid)
            if canon != sid:
                continue
            kld_scores[canon] = self._symmetric_kld_stad(state_Pi_kld[canon], pi_new)

        if not kld_scores:
            return current_state_kld, 0.0, "fallback_match"

        best_sid = min(kld_scores, key=kld_scores.get)
        return int(best_sid), float(kld_scores[best_sid]), "fallback_match"
    @staticmethod
    def _stad_density_estimation(z: np.ndarray) -> np.ndarray:
       
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z[None, :]
        # Apply row-wise softmax then column mean — exact original STAD formula
        # Build DataFrame first so softmax operates correctly row-wise,
        # then extract numpy array safely regardless of pandas version
        df   = pd.DataFrame(z)
        soft = softmax(df.values, axis=1)   # scipy softmax on numpy → always numpy
        pi   = soft.mean(axis=0)
        return np.asarray(pi, dtype=float)

    @staticmethod
    def _symmetric_kld_stad(p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
        q = np.clip(np.asarray(q, dtype=float), 1e-12, None)
        return float(np.sum(p * np.log(p / q) + q * np.log(q / p)))

    def _update_state_proto(self, sid: int, z: np.ndarray, state_protos: Dict[int, np.ndarray]):
        
        if sid not in state_protos:
            state_protos[sid] = z.copy()

    def _merge_states(self, state_protos, state_counts, uf_parent, log_lines) -> int:
        sids, protos = [], []
        for sid in sorted(state_protos.keys()):
            canon = self._uf_find(uf_parent, sid)
            if canon != sid or int(state_counts.get(canon, 0)) < self.min_state_windows:
                continue
            sids.append(canon); protos.append(state_protos[canon])

        if len(sids) < 2:
            return 0

        P    = self._l2_norm_rows(np.stack(protos, axis=0))
        S    = P @ P.T
        cand = [(float(S[i, j]), sids[i], sids[j])
                for i in range(len(sids)) for j in range(i + 1, len(sids))
                if S[i, j] >= self.merge_threshold]
        if not cand:
            return 0

        cand.sort(reverse=True, key=lambda x: x[0])
        merges = 0
        for sim, a, b in cand:
            ra, rb = self._uf_find(uf_parent, a), self._uf_find(uf_parent, b)
            if ra == rb:
                continue
            root  = self._uf_union(uf_parent, ra, rb)
            other = rb if root == ra else ra
            na, nb = int(state_counts.get(root, 0)) or 1, int(state_counts.get(other, 0)) or 1
            state_protos[root]  = (na * state_protos[root] + nb * state_protos[other]) / float(na + nb)
            state_counts[root]  = na + nb
            state_protos.pop(other, None); state_counts.pop(other, None)
            merges += 1

        if merges:
            log_lines.append(f"[MERGE] merges={merges} remaining_states={len(state_protos)}\n")
        return merges

    def _get_query_text(self, window_idx: int, window_data: np.ndarray) -> str:
      
        if self.original_texts is not None:
            text_idx = window_idx * self.steps
            if text_idx < len(self.original_texts):
                return self.original_texts[text_idx]
        return f"Query from window {window_idx}"

    def _get_query_records_for_window(self, window_idx: int) -> list:
       
        if self.original_texts is None:
            return [{
                "window_idx": int(window_idx),
                "original_idx": -1,
                "query_text": f"Query from window {window_idx}",
            }]

        start = int(window_idx) * self.steps
        
        step_len = max(1, int(self.steps))
        end = min(start + step_len, len(self.original_texts))
        seen = set()
        records = []

        for original_idx in range(start, end):
            text = self.original_texts[original_idx]
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue

            
            dedupe_key = text
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            records.append({
                "window_idx": int(window_idx),
                "original_idx": int(original_idx),
                "query_text": text,
            })

        if not records:
            records.append({
                "window_idx": int(window_idx),
                "original_idx": -1,
                "query_text": f"Query from window {window_idx}",
            })
        return records

    def _get_all_queries_for_window(self, window_idx: int) -> list:
     
        return [r["query_text"] for r in self._get_query_records_for_window(window_idx)]

    def _get_gold_answer_for_query_record(self, query_record: dict) -> str:
       
        if not hasattr(self, "ground_truth_answers") or not self.ground_truth_answers:
            return ""

        original_idx = int(query_record.get("original_idx", -1))
        if 0 <= original_idx < len(self.ground_truth_answers):
            return self.ground_truth_answers[original_idx] or ""

        if self.original_texts is not None:
            query = str(query_record.get("query_text", "")).strip()
            start = int(query_record.get("window_idx", 0)) * self.steps
            end = min(start + self.window_size, len(self.original_texts))
            for idx in range(start, end):
                if str(self.original_texts[idx]).strip() == query:
                    if idx < len(self.ground_truth_answers):
                        return self.ground_truth_answers[idx] or ""

        return ""

    @staticmethod
    def _normalize_for_match(text: str) -> str:
       
        text = str(text or "").lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _doc_key_for_metrics(doc: dict):
      
        title = str(doc.get("title", "")).strip().lower()
        if title:
            return ("title", title)
        doc_id = doc.get("id")
        if doc_id is not None:
            return ("id", str(doc_id))
        text = str(doc.get("text", "")).strip().lower()
        return ("text", text[:300])

    @classmethod
    def _doc_key_set(cls, docs: list) -> set:
        return {cls._doc_key_for_metrics(d) for d in (docs or []) if isinstance(d, dict)}

    @classmethod
    def _fresh_cache_coverage(cls, fresh_docs: list, cached_docs: list, k: int = 5) -> dict:
        
        fresh_top = list(fresh_docs or [])[:int(k)]
        fresh_keys = [cls._doc_key_for_metrics(d) for d in fresh_top if isinstance(d, dict)]
        cache_keys = cls._doc_key_set(cached_docs)
        overlap = sum(1 for key in fresh_keys if key in cache_keys)
        denom = max(1, len(fresh_keys))
        return {
            "coverage_at_k": float(overlap / denom),
            "overlap_count": int(overlap),
            "fresh_k": int(len(fresh_keys)),
        }

    @classmethod
    def _answer_in_doc_pool(cls, gold_answer: str, docs: list) -> int:
       
        gold_n = cls._normalize_for_match(gold_answer)
        if not gold_n:
            return 0
        for doc in docs or []:
            text_n = cls._normalize_for_match(doc.get("text", ""))
            if gold_n and gold_n in text_n:
                return 1
        return 0

    @staticmethod
    def _state_cache_doc_stats(docs: list) -> dict:
        docs = list(docs or [])
        hits = [int(d.get("state_cache_retrieval_hits", 1) or 1) for d in docs]
        chars = [len(str(d.get("text", ""))) for d in docs]
        return {
            "unique_docs": int(len(docs)),
            "total_doc_chars": int(sum(chars)),
            "mean_doc_chars": float(np.mean(chars)) if chars else 0.0,
            "mean_retrieval_hits_per_doc": float(np.mean(hits)) if hits else 0.0,
            "max_retrieval_hits_per_doc": int(max(hits)) if hits else 0,
            "reused_doc_fraction": float(np.mean([h > 1 for h in hits])) if hits else 0.0,
        }

    def _rag_answer(self, query: str, state_id: int, is_alarm: bool, reason: str) -> str:
        if not self.enable_rag:
            return ""
        if self.knowledge_cache is None or self.retriever is None or self.generator is None:
            return ""
        cached_knowledge = self.knowledge_cache.get(state_id)
        if cached_knowledge is not None:
            knowledge = cached_knowledge
        else:
            knowledge = self.retriever.retrieve(query)
            self.knowledge_cache.set(state_id, knowledge)
        return self.generator.generate(query, knowledge)

    def _map_query_to_state_simple(self, query_text: str) -> int:
        from sentence_transformers import SentenceTransformer
        encoder   = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = self._l2_norm(encoder.encode([query_text])[0])
        best_state, best_sim = 0, -np.inf
        for state_id, prototype in self.state_protos.items():
            sim = np.dot(query_emb, self._l2_norm(prototype))
            if sim > best_sim:
                best_sim = sim; best_state = state_id
        return best_state

    def _run_baseline_rag_comparison(self):
        try:
            from .query_selector import QuerySelector
            print("\n" + "=" * 70)
            print(f"🔬 BASELINE VS RAG COMPARISON ({self.num_test_queries} queries)")
            print("=" * 70)
            selector = QuerySelector('data/agnews/test.csv')
            queries  = selector.select_per_category(
                n_per_category=max(1, self.num_test_queries // 4)
            )[:self.num_test_queries]
            results = []
            for i, q in enumerate(queries, 1):
                qt       = q['text']
                state_id = self._map_query_to_state_simple(qt)
                knowledge = self.knowledge_cache.get(state_id)
                if knowledge is None:
                    knowledge = self.retriever.retrieve(qt)
                    self.knowledge_cache.set(state_id, knowledge)
                results.append({
                    'query_id': q['query_id'], 'query_text': qt,
                    'true_label': q['true_label'], 'state_id': state_id,
                    'baseline_answer': self.answer_gen.generate_baseline(qt),
                    'rag_answer': self.answer_gen.generate_with_rag(qt, knowledge),
                    'knowledge': knowledge,
                })
            comparisons = [
                self.comparator.compare_answers(
                    query=r['query_text'],
                    baseline_answer=r['baseline_answer'],
                    cached_rag_answer=r['rag_answer'],
                    fresh_rag_answer=r['rag_answer'],
                    gold_answer="",
                ) for r in results
            ]
            aggregate = self.comparator.aggregate_comparison(comparisons)
            self.comparator.print_comparison_summary(aggregate)
            f = os.path.join(self.experiment_dir, "baseline_vs_rag.csv")
            with open(f, 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=['query_id','query_text','state_id','true_label','baseline_answer','rag_answer'])
                w.writeheader()
                for r in results:
                    w.writerow({k: r[k] for k in ['query_id','query_text','state_id','true_label','baseline_answer','rag_answer']})
            with open(os.path.join(self.experiment_dir, "comparison_metrics.json"), 'w') as fh:
                json.dump({'comparisons': comparisons, 'aggregate': aggregate}, fh, indent=2)
        except Exception as e:
            print(f"\n⚠️  Baseline vs RAG comparison failed: {e}")
            import traceback; traceback.print_exc()


    def _build_state_topic_queries(
        self,
        primary_queries: list,
        top_n_terms: int = 12,
        max_topic_queries: int = 6,
        representative_k: int = 3,
    ) -> list:
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as _np
        except Exception as e:
            print(f"   Topic-query extraction unavailable: {e}")
            return []

        texts = []
        for q in primary_queries:
            if isinstance(q, dict):
                t = str(q.get("query_text", "")).strip()
            else:
                t = str(q).strip()
            if t:
                texts.append(t)

        if not texts:
            return []

        if len(texts) < 3:
            return texts[:max_topic_queries]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=300,
        )

        try:
            X = vectorizer.fit_transform(texts)
            terms = vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"  TF-IDF topic extraction failed: {e}")
            return texts[:max_topic_queries]

        if X.shape[1] == 0:
            return texts[:max_topic_queries]

        scores = _np.asarray(X.mean(axis=0)).reshape(-1)
        top_idx = scores.argsort()[::-1][:top_n_terms]
        top_terms = [terms[i] for i in top_idx if scores[i] > 0]

        topic_queries = []

        # Broad state-topic query.
        if top_terms:
            topic_queries.append(" ".join(top_terms[:5]))

        # Smaller focused topic queries.
        for i in range(0, min(len(top_terms), top_n_terms), 3):
            q = " ".join(top_terms[i:i + 3]).strip()
            if q:
                topic_queries.append(q)

        # Representative original primary queries closest to the TF-IDF centroid.
        try:
            centroid = X.mean(axis=0)
            sims = cosine_similarity(X, centroid).reshape(-1)
            rep_idx = sims.argsort()[::-1][:representative_k]
            for idx in rep_idx:
                topic_queries.append(texts[int(idx)])
        except Exception:
            topic_queries.extend(texts[:representative_k])

        final_queries = []
        seen = set()
        for q in topic_queries:
            q = str(q).strip()
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                final_queries.append(q)

        return final_queries[:max_topic_queries]

    def _compute_topic_query_support_counts(
        self,
        state_id: int,
        topic_queries: list,
        primary_queries: list,
        topic_retrieval_topk: int = 0,
    ) -> list:
       
        def _get_text(q):
            if isinstance(q, dict):
                return str(q.get("query_text", "")).strip()
            return str(q).strip()

        def _norm(text):
            text = str(text).lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        texts = [_get_text(q) for q in primary_queries]
        texts = [t for t in texts if t]
        norm_texts = [_norm(t) for t in texts]
        norm_text_set = set(norm_texts)

        rows = []

        for rank, tq in enumerate(topic_queries, start=1):
            topic_query = str(tq).strip()
            topic_norm = _norm(topic_query)

            if not topic_norm:
                continue

            topic_terms = [term for term in topic_norm.split() if len(term) > 2]

            exact_phrase_count = sum(
                1 for txt in norm_texts
                if topic_norm and topic_norm in txt
            )

            support_any_count = sum(
                1 for txt in norm_texts
                if topic_terms and any(term in txt.split() for term in topic_terms)
            )

            support_all_count = sum(
                1 for txt in norm_texts
                if topic_terms and all(term in txt.split() for term in topic_terms)
            )

            is_representative_query = topic_norm in norm_text_set

            rows.append({
                "state_id": int(state_id),
                "topic_rank": int(rank),
                "topic_query": topic_query,
                "topic_query_norm": topic_norm,
                "topic_terms": " ".join(topic_terms),
                "num_primary_queries_in_state": int(len(texts)),
                "support_any_count": int(support_any_count),
                "support_all_count": int(support_all_count),
                "exact_phrase_count": int(exact_phrase_count),
                "support_any_rate": float(support_any_count / max(1, len(texts))),
                "support_all_rate": float(support_all_count / max(1, len(texts))),
                "exact_phrase_rate": float(exact_phrase_count / max(1, len(texts))),
                "topic_retrieval_topk": int(topic_retrieval_topk),
                "topic_retrieval_count": 1,
                "topic_type": (
                    "representative_primary_query"
                    if is_representative_query
                    else "tfidf_topic_query"
                ),
            })

        return rows

    def _save_all_state_topic_summaries(self, state_queries: dict, all_topic_usage_rows: Optional[list] = None) -> dict:
        
        if all_topic_usage_rows is None:
            all_topic_usage_rows = []

        all_state_topic_rows = []
        all_state_topic_usage_rows = []

        topic_topk = int(os.environ.get("DTSS_STATE_TOPIC_RETRIEVE_TOPK", "5"))
        top_n_terms = int(os.environ.get("DTSS_STATE_TOPIC_TOP_TERMS", "10"))
        max_topic_queries = int(os.environ.get("DTSS_STATE_TOPIC_MAX_QUERIES", "3"))
        representative_k = int(os.environ.get("DTSS_STATE_TOPIC_REPRESENTATIVE_K", "3"))
        min_topic_state_queries = int(os.environ.get("DTSS_MIN_TOPIC_STATE_QUERIES", "2"))

        for topic_state_id, topic_queries_for_state in sorted(state_queries.items()):
            if len(topic_queries_for_state) < min_topic_state_queries:
                continue

            extracted_topics = self._build_state_topic_queries(
                primary_queries=topic_queries_for_state,
                top_n_terms=top_n_terms,
                max_topic_queries=max_topic_queries,
                representative_k=representative_k,
            )

            topic_stats = self._compute_topic_query_support_counts(
                state_id=int(topic_state_id),
                topic_queries=extracted_topics,
                primary_queries=topic_queries_for_state,
                topic_retrieval_topk=int(topic_topk),
            )

            all_state_topic_usage_rows.extend(topic_stats)

            top_topic_strings = [
                str(x.get("topic_query", "")).strip()
                for x in topic_stats
                if str(x.get("topic_query", "")).strip()
            ]

            all_state_topic_rows.append({
                "state_id": int(topic_state_id),
                "num_queries": int(len(topic_queries_for_state)),
                "num_unique_topics": int(len(set(top_topic_strings))),
                "num_topic_queries": int(len(top_topic_strings)),
                "top_topics": " | ".join(top_topic_strings[:10]),
                "top_support_any_counts": " | ".join(
                    str(int(x.get("support_any_count", 0))) for x in topic_stats[:10]
                ),
                "top_support_all_counts": " | ".join(
                    str(int(x.get("support_all_count", 0))) for x in topic_stats[:10]
                ),
                "mean_support_any_rate": float(
                    np.mean([float(x.get("support_any_rate", 0.0)) for x in topic_stats])
                ) if topic_stats else 0.0,
                "mean_support_all_rate": float(
                    np.mean([float(x.get("support_all_rate", 0.0)) for x in topic_stats])
                ) if topic_stats else 0.0,
            })

            per_state_topic_json = os.path.join(
                self.experiment_dir,
                f"state_{int(topic_state_id)}_topic_summary.json",
            )
            with open(per_state_topic_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state_id": int(topic_state_id),
                        "num_queries": int(len(topic_queries_for_state)),
                        "num_unique_topics": int(len(set(top_topic_strings))),
                        "topic_queries": extracted_topics,
                        "topic_query_stats": topic_stats,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        state_topic_summary_df = pd.DataFrame(all_state_topic_rows)
        state_topic_usage_df = pd.DataFrame(all_state_topic_usage_rows)

        state_topic_summary_csv = os.path.join(self.experiment_dir, "state_topic_summary.csv")
        state_topic_summary_json = os.path.join(self.experiment_dir, "state_topic_summary.json")
        state_topic_summary_txt = os.path.join(self.experiment_dir, "state_topic_summary.txt")
        state_topic_usage_csv = os.path.join(self.experiment_dir, "state_topic_usage_counts.csv")
        global_topic_usage_csv = os.path.join(self.experiment_dir, "global_topic_usage_summary.csv")
        global_topic_usage_json = os.path.join(self.experiment_dir, "global_topic_usage_summary.json")

        state_topic_summary_df.to_csv(state_topic_summary_csv, index=False)
        state_topic_usage_df.to_csv(state_topic_usage_csv, index=False)

        if not state_topic_usage_df.empty and "topic_query_norm" in state_topic_usage_df.columns:
            global_topic_df = (
                state_topic_usage_df
                .groupby("topic_query_norm", as_index=False)
                .agg(
                    topic_query=("topic_query", "first"),
                    states_using_topic=("state_id", "nunique"),
                    total_support_any_count=("support_any_count", "sum"),
                    total_support_all_count=("support_all_count", "sum"),
                    total_exact_phrase_count=("exact_phrase_count", "sum"),
                    mean_support_any_rate=("support_any_rate", "mean"),
                    mean_support_all_rate=("support_all_rate", "mean"),
                    mean_exact_phrase_rate=("exact_phrase_rate", "mean"),
                    total_topic_retrieval_count=("topic_retrieval_count", "sum"),
                )
                .sort_values(["states_using_topic", "total_support_any_count"], ascending=[False, False])
            )
        else:
            global_topic_df = pd.DataFrame(columns=[
                "topic_query_norm",
                "topic_query",
                "states_using_topic",
                "total_support_any_count",
                "total_support_all_count",
                "total_exact_phrase_count",
                "mean_support_any_rate",
                "mean_support_all_rate",
                "mean_exact_phrase_rate",
                "total_topic_retrieval_count",
            ])

        global_topic_df.to_csv(global_topic_usage_csv, index=False)

        state_payload = {
            "num_states_with_topic_summary": int(len(all_state_topic_rows)),
            "states": all_state_topic_rows,
            "note": (
                "This file summarizes unique topic queries per discovered state. "
                "It is generated before RAG holdout evaluation, so it is not affected by the holdout cap."
            ),
        }
        with open(state_topic_summary_json, "w", encoding="utf-8") as f:
            json.dump(state_payload, f, indent=2, ensure_ascii=False)

        with open(state_topic_summary_txt, "w", encoding="utf-8") as f:
            f.write("State-topic summary\n")
            f.write("=" * 80 + "\n\n")
            for row in all_state_topic_rows:
                f.write(
                    f"State {row['state_id']}: "
                    f"{row['num_queries']} queries, "
                    f"{row['num_unique_topics']} unique topics\n"
                )
                f.write(f"Top topics: {row['top_topics']}\n\n")

        global_payload = {
            "num_unique_topic_queries": int(len(global_topic_df)),
            "top_topics": global_topic_df.head(30).to_dict(orient="records"),
            "files": {
                "state_topic_summary_csv": "state_topic_summary.csv",
                "state_topic_usage_counts_csv": "state_topic_usage_counts.csv",
                "global_topic_usage_summary_csv": "global_topic_usage_summary.csv",
            },
            "explanation": (
                "states_using_topic counts how many discovered states use a topic. "
                "total_support_any_count counts how many state queries contain at least one term from the topic query. "
                "These statistics support state interpretation, dataset construction, and recommendation-style extensions."
            ),
        }
        with open(global_topic_usage_json, "w", encoding="utf-8") as f:
            json.dump(global_payload, f, indent=2, ensure_ascii=False)

        print(f"   {state_topic_summary_csv}")
        print(f"   {state_topic_summary_json}")
        print(f"    {state_topic_summary_txt}")
        print(f"    {state_topic_usage_csv}")
        print(f"    {global_topic_usage_csv}")
        print(f"    {global_topic_usage_json}")

        # Make the later RAG-topic summary include all state topics, not only
        # whichever state happens to enter the holdout-answering branch.
        all_topic_usage_rows.extend(all_state_topic_usage_rows)
        self._all_state_topic_summary_generated = True

        return {
            "num_states_with_topic_summary": int(len(all_state_topic_rows)),
            "num_unique_topic_queries": int(len(global_topic_df)),
            "top_topics": global_topic_df.head(10).to_dict(orient="records"),
            "files": {
                "state_topic_summary_csv": "state_topic_summary.csv",
                "state_topic_usage_counts_csv": "state_topic_usage_counts.csv",
                "global_topic_usage_summary_csv": "global_topic_usage_summary.csv",
                "state_topic_summary_json": "state_topic_summary.json",
                "state_topic_summary_txt": "state_topic_summary.txt",
                "global_topic_usage_summary_json": "global_topic_usage_summary.json",
            },
        }


    def _run_simplified_efficiency_analysis(self):
        """
        Per-state 80/20 RAG comparison:
          - For every discovered state, collect all unique queries assigned to that state.
          - Use 80% of that state's queries as primary queries to retrieve documents
            and build the state-level cached knowledge pool.
          - Use the remaining 20% as holdout queries for the 3-way comparison:
                baseline   : no retrieval
                cached_rag : state-level cached knowledge context by default; optional KV cache
                fresh_rag  : per-query retrieval

        Uses self.state_assignments, which comes from the active state logic
        selected by self.state_logic: cosine or kld.
        """
        try:
            print("\n" + "=" * 70)
            print("🔬 PER-STATE 80/20 RAG EFFICIENCY-QUALITY ANALYSIS")
            print("   For each state: 80% primary candidates / 20% holdout answering")
            print("   Cache is built from a smaller primary subset and filtered per query")
            print("   baseline / filtered_cached_rag / fresh_rag")
            print("=" * 70)

            import time
            from datetime import datetime, timedelta

            analysis_start_time = time.time()

            from .state_cache_filtering import (
                get_state_cache_filter_config,
                select_cache_build_queries,
                build_state_cache_index,
                select_relevant_cached_docs,
            )

            cache_filter_cfg = get_state_cache_filter_config()
            print("   State-cache filtering config:", cache_filter_cfg.to_dict())
            if cache_filter_cfg.enabled:
                print("   Cached RAG mode: query-filtered state cache with insufficient-cache fallback")
                print("   Note: DTSS_USE_TRUE_KV=0 is set in code by default for filtered mode.")
            else:
                print("   Cached RAG mode: legacy full state-cache context")

            def _fmt_seconds(seconds: float) -> str:
                seconds = max(0, int(seconds))
                h = seconds // 3600
                m = (seconds % 3600) // 60
                sec = seconds % 60
                if h > 0:
                    return f"{h}h {m}m {sec}s"
                if m > 0:
                    return f"{m}m {sec}s"
                return f"{sec}s"

            def _eta_message(prefix: str, done: int, total: int, start_time: float) -> str:
                elapsed = time.time() - start_time
                avg = elapsed / max(1, done)
                remaining = max(0, total - done)
                eta_seconds = remaining * avg
                finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                return (
                    f"{prefix} {done}/{total} | "
                    f"elapsed={_fmt_seconds(elapsed)} | "
                    f"avg={avg:.2f}s/query | "
                    f"remaining={remaining} | "
                    f"ETA={_fmt_seconds(eta_seconds)} | "
                    f"finish≈{finish_time.strftime('%H:%M:%S')}"
                )

            if self.original_texts is None or len(self.original_texts) == 0:
                print(" No original_texts found. Cannot run meaningful RAG analysis.")
                return

            def _generate_baseline(query: str) -> str:
                if self.answer_gen is not None and hasattr(self.answer_gen, "generate_baseline"):
                    return self.answer_gen.generate_baseline(query)
                if self.generator is not None and hasattr(self.generator, "generate_fresh"):
                    return self.generator.generate_fresh(query, [])
                if self.generator is not None and hasattr(self.generator, "generate"):
                    return self.generator.generate(query, [])
                return ""

            def _generate_with_rag(query: str, docs: list) -> str:
                if self.answer_gen is not None and hasattr(self.answer_gen, "generate_with_rag"):
                    return self.answer_gen.generate_with_rag(query, docs)
                if self.generator is not None and hasattr(self.generator, "generate_fresh"):
                    return self.generator.generate_fresh(query, docs)
                if self.generator is not None and hasattr(self.generator, "generate"):
                    return self.generator.generate(query, docs)
                return ""

            def _generate_with_kv(query: str, kv_cache, kv_len: int, fallback_docs: list) -> str:
                if kv_cache is None or kv_len <= 0:
                    return _generate_with_rag(query, fallback_docs)
                if self.answer_gen is not None and hasattr(self.answer_gen, "generate_with_kv_cache"):
                    return self.answer_gen.generate_with_kv_cache(query, kv_cache, kv_len)
                if self.generator is not None and hasattr(self.generator, "generate_with_kv_cache"):
                    return self.generator.generate_with_kv_cache(query, kv_cache, kv_len)
                return _generate_with_rag(query, fallback_docs)

            state_queries = {}
            globally_seen = set()
            for idx in range(len(self.state_assignments)):
                sid = int(self.state_assignments[idx])
                query_records = self._get_query_records_for_window(idx)
                for qr in query_records:
                    qt = str(qr.get("query_text", "")).strip()
                    if not qt:
                        continue

                    original_idx = int(qr.get("original_idx", -1))
                    # Prevent leakage caused by overlapping sliding windows:
                    # the same original sample must not be used in both the
                    # primary cache-building set and the holdout set.
                    global_key = ("idx", original_idx) if original_idx >= 0 else ("text", qt)
                    if global_key in globally_seen:
                        continue
                    globally_seen.add(global_key)

                    state_queries.setdefault(sid, {})
                    if global_key not in state_queries[sid]:
                        state_queries[sid][global_key] = {
                            "window_idx": int(qr.get("window_idx", idx)),
                            "original_idx": original_idx,
                            "query_text": qt,
                            "state_id": sid,
                        }

            state_queries = {
                sid: list(qdict.values())
                for sid, qdict in state_queries.items()
            }

            total_unique = sum(len(v) for v in state_queries.values())
            print(f"   Unique queries across all states: {total_unique}")
            for sid, qs in sorted(state_queries.items()):
                print(f"   State {sid}: {len(qs)} unique queries")

            self.topic_retrieval_count_total = 0
            all_primary_results = []
            all_holdout_results = []
            all_topic_usage_rows = []

            # Save state-topic summaries for all discovered states before the RAG
            # diagnostic loop starts. This is independent of the holdout cap.
            all_state_topic_summary = self._save_all_state_topic_summaries(
                state_queries=state_queries,
                all_topic_usage_rows=all_topic_usage_rows,
            )

            # Maximum number of holdout questions to answer across the full run.
            # Each holdout question still receives three generated answers:
            # baseline, cached_rag, and fresh_rag.
            max_total_holdout_questions = int(
                os.environ.get(
                    "DTSS_MAX_TOTAL_HOLDOUT_QUERIES",
                    str(self.num_test_queries if int(getattr(self, "num_test_queries", 0)) > 0 else 50),
                )
            )
            min_rag_state_queries = int(os.environ.get("DTSS_MIN_RAG_STATE_QUERIES", "5"))
            fresh_topk = int(os.environ.get("DTSS_FRESH_RAG_TOPK", "5"))
            holdout_questions_selected_so_far = 0

            print(
                f"   Holdout answer cap: "
                f"{max_total_holdout_questions if max_total_holdout_questions > 0 else 'no cap'} questions"
            )

            for state_id, queries in sorted(state_queries.items()):
                if len(queries) < min_rag_state_queries:
                    print(
                        f"\n⚠️ State {state_id}: {len(queries)} queries — "
                        f"skipping because need at least {min_rag_state_queries} for RAG evaluation"
                    )
                    continue

                print(f"\n{'=' * 70}")
                print(f"📊 STATE {state_id}: {len(queries)} queries")

                import random
                shuffled = queries.copy()
                rng = random.Random(self.rag_random_seed + int(state_id))
                rng.shuffle(shuffled)

                split_point = max(1, int(len(shuffled) * 0.8))
                if split_point >= len(shuffled):
                    split_point = len(shuffled) - 1

                primary_queries = shuffled[:split_point]
                holdout_queries = shuffled[split_point:]

                # Thesis RAG evaluation should use real dataset gold answers.
                # If gold answers are available, remove holdout records without
                # a gold answer so that cached/fresh/baseline are evaluated
                # against ground truth, not against a proxy reference.
                if hasattr(self, "ground_truth_answers") and self.ground_truth_answers:
                    before_gold_filter = len(holdout_queries)
                    holdout_queries = [
                        q for q in holdout_queries
                        if str(self._get_gold_answer_for_query_record(q) or "").strip()
                    ]
                    if len(holdout_queries) < before_gold_filter:
                        print(
                            f"   Gold-answer filter: {before_gold_filter} → "
                            f"{len(holdout_queries)} holdout queries"
                        )

              
                original_holdout_count = len(holdout_queries)

                if max_total_holdout_questions > 0:
                    remaining_holdout_budget = (
                        max_total_holdout_questions - holdout_questions_selected_so_far
                    )

                    if remaining_holdout_budget <= 0:
                        print(
                            f"   ⚠️ Holdout cap already reached "
                            f"({max_total_holdout_questions}); skipping remaining states."
                        )
                        break

                    holdout_queries = holdout_queries[:remaining_holdout_budget]
                    holdout_questions_selected_so_far += len(holdout_queries)

                    if len(holdout_queries) < original_holdout_count:
                        print(
                            f"   ✂️ Holdout capped for state {state_id}: "
                            f"{original_holdout_count} → {len(holdout_queries)}"
                        )
                else:
                    holdout_questions_selected_so_far += len(holdout_queries)

                cache_build_queries = select_cache_build_queries(
                    primary_queries=primary_queries,
                    state_id=state_id,
                    seed=self.rag_random_seed,
                    config=cache_filter_cfg,
                )

                print(f"   Primary candidates: {len(primary_queries)}")
                print(f"   Cache-building subset: {len(cache_build_queries)}")
                print(f"   Holdout: {len(holdout_queries)}")

                if len(holdout_queries) == 0:
                    print(f" No holdout queries for state {state_id}; skipping.")
                    continue

               
                print("\n PHASE 1: Subset retrieval for compact state-cache construction")
                print("    No LLM answers are generated here; only the cache-building subset retrieves documents.")
                primary_knowledge_pool = []
                primary_phase_start = time.time()

                for i, pq in enumerate(cache_build_queries, 1):
                    if i == 1 or i % 25 == 0 or i == len(cache_build_queries):
                        print(
                            "   ├─ " + _eta_message(
                                "Cache-build retrieval",
                                i,
                                len(cache_build_queries),
                                primary_phase_start,
                            ),
                            flush=True,
                        )

                    qt = pq["query_text"]
                    state_cag_retrieve_topk = int(os.environ.get("DTSS_STATE_CAG_RETRIEVE_TOPK", "10"))
                    kno = self.retriever.retrieve(qt, top_k=state_cag_retrieve_topk)

                    # Important for thesis experiment:
                    # Primary queries are used only to extract/retrieve documents
                    # and build the state cache. We intentionally do NOT call
                    # the LLM here, otherwise 80% of the state queries are also
                    # answered, making the run unnecessarily slow.
                    all_primary_results.append({
                        "window_idx": pq["window_idx"],
                        "original_idx": pq.get("original_idx", -1),
                        "query": qt,
                        "state_id": state_id,
                        "set_type": "primary",
                        "answer": "",
                        "knowledge_source": "state_cache_build_subset_retrieval",
                        "retrieval_performed": True,
                    })
                    primary_knowledge_pool.extend(kno)

                primary_elapsed = time.time() - primary_phase_start
                print(
                    f"   └─ ✅ {len(cache_build_queries)} cache-building queries retrieved "
                    f"for compact cache construction in {_fmt_seconds(primary_elapsed)}"
                )

                use_topic_cache = os.environ.get("DTSS_USE_STATE_TOPIC_CACHE", "1") == "1"
                topic_source = os.environ.get("DTSS_STATE_TOPIC_SOURCE", "primary").lower().strip()
                topic_queries = []
                topic_retrieval_count = 0
                topic_topk = int(os.environ.get("DTSS_STATE_TOPIC_RETRIEVE_TOPK", "10"))

                if use_topic_cache:
                    top_n_terms = int(os.environ.get("DTSS_STATE_TOPIC_TOP_TERMS", "12"))
                    max_topic_queries = int(os.environ.get("DTSS_STATE_TOPIC_MAX_QUERIES", "6"))
                    representative_k = int(os.environ.get("DTSS_STATE_TOPIC_REPRESENTATIVE_K", "3"))

                    topic_base_queries = primary_queries if topic_source == "primary" else cache_build_queries
                    topic_queries = self._build_state_topic_queries(
                        primary_queries=topic_base_queries,
                        top_n_terms=top_n_terms,
                        max_topic_queries=max_topic_queries,
                        representative_k=representative_k,
                    )

                    topic_query_stats = self._compute_topic_query_support_counts(
                        state_id=int(state_id),
                        topic_queries=topic_queries,
                        primary_queries=topic_base_queries,
                        topic_retrieval_topk=int(topic_topk),
                    )
                    if not getattr(self, "_all_state_topic_summary_generated", False):
                        all_topic_usage_rows.extend(topic_query_stats)

                    print(f"\n   🧠 State {state_id} topic queries for cache expansion:")
                    if topic_queries:
                        for tq in topic_queries:
                            print(f"      - {tq}")
                    else:
                        print("      none")

                    topic_phase_start = time.time()
                    for tq in topic_queries:
                        try:
                            topic_docs = self.retriever.retrieve(tq, top_k=topic_topk)
                            topic_retrieval_count += 1
                            self.topic_retrieval_count_total = int(
                                getattr(self, "topic_retrieval_count_total", 0)
                            ) + 1

                            for doc in topic_docs:
                                if isinstance(doc, dict):
                                    dcopy = dict(doc)
                                    dcopy["state_cache_source"] = "topic_query"
                                    dcopy["state_topic_query"] = tq
                                    primary_knowledge_pool.append(dcopy)
                        except Exception as e:
                            print(f"   ⚠️ Topic retrieval failed for '{tq}': {e}")

                    topic_elapsed = time.time() - topic_phase_start
                    print(
                        f"   └─ ✅ {topic_retrieval_count} topic queries retrieved "
                        f"in {_fmt_seconds(topic_elapsed)}"
                    )

                    topic_log_path = os.path.join(
                        self.experiment_dir,
                        f"state_{state_id}_topic_queries.json",
                    )
                    with open(topic_log_path, "w", encoding="utf-8") as tf:
                        json.dump(
                            {
                                "state_id": int(state_id),
                                "topic_source": topic_source,
                                "num_primary_queries": int(len(primary_queries)),
                                "num_cache_build_queries": int(len(cache_build_queries)),
                                "topic_queries": topic_queries,
                                "topic_query_stats": topic_query_stats,
                                "topic_retrieval_topk": int(topic_topk),
                                "note": (
                                    "Topic queries are extracted only from primary/cache-building-side queries, "
                                    "never holdout queries. support_any_count counts how many primary queries "
                                    "contain at least one term from the topic query; support_all_count counts "
                                    "how many contain all topic terms. These counts can be used to inspect "
                                    "recurring state-level topics."
                                ),
                            },
                            tf,
                            indent=2,
                            ensure_ascii=False,
                        )
                    print(f"   └─ 💾 Topic queries → {topic_log_path}")

                print(f"\n   🟢 Building state {state_id} knowledge pool")

                state_cag_max_docs = int(os.environ.get("DTSS_STATE_CAG_MAX_DOCS", "80"))
                state_knowledge_pool = self.knowledge_cache.aggregate_knowledge_pool(
                    knowledge_list=[primary_knowledge_pool],
                    max_docs=state_cag_max_docs,
                    max_per_title=1,  # thesis setting: each retrieved document/title appears once
                )
                self.knowledge_cache.set(state_id, state_knowledge_pool)
                state_cache_stats = self._state_cache_doc_stats(state_knowledge_pool)
                print(f"   └─ ✅ {len(state_knowledge_pool)} unique docs cached")
                print(
                    f"      total_chars={state_cache_stats['total_doc_chars']} | "
                    f"mean_hits/doc={state_cache_stats['mean_retrieval_hits_per_doc']:.2f} | "
                    f"reused_doc_fraction={state_cache_stats['reused_doc_fraction']:.3f}"
                )

                cache_log_path = os.path.join(
                    self.experiment_dir,
                    f"cached_state_{state_id}.json",
                )
                cache_log = {
                    "state_id": int(state_id),
                    "num_docs": len(state_knowledge_pool),
                    "cache_type": "compact_state_cache_query_filtered_topic_expanded",
                    "topic_expansion_enabled": bool(use_topic_cache),
                    "topic_source": topic_source,
                    "topic_queries": topic_queries,
                    "topic_retrieval_count": int(topic_retrieval_count),
                    "topic_retrieval_topk": int(topic_topk),
                    "note": "text_preview is only for logging. Documents come from the cache-building subset of primary queries plus optional topic-query expansion. Holdout queries are not used to build this cache. During answering, only query-relevant cached documents are appended as context.",
                    "cache_filter_config": cache_filter_cfg.to_dict(),
                    "primary_candidates": len(primary_queries),
                    "cache_build_queries": len(cache_build_queries),
                    "holdout_queries": len(holdout_queries),
                    "state_cache_stats": state_cache_stats,
                    "documents": [
                        {
                            "rank": rank + 1,
                            "id": doc.get("id"),
                            "title": doc.get("title", ""),
                            "text_preview": doc.get("text", "")[:500],
                            "text_chars": len(str(doc.get("text", ""))),
                            "score": round(doc.get("retrieval_score", 0), 4),
                            "state_cache_source": doc.get("state_cache_source", "primary_query"),
                            "state_topic_query": doc.get("state_topic_query", ""),
                            "state_cache_retrieval_hits": int(doc.get("state_cache_retrieval_hits", 1)),
                            "category": doc.get("category", ""),
                        }
                        for rank, doc in enumerate(state_knowledge_pool)
                    ],
                }
                with open(cache_log_path, "w", encoding="utf-8") as _cf:
                    json.dump(cache_log, _cf, indent=2, ensure_ascii=False)
                print(f"   └─ 💾 Cache content → {cache_log_path}")

                state_cache_index = build_state_cache_index(
                    cached_docs=state_knowledge_pool,
                    encoder=self.retriever.encoder,
                    config=cache_filter_cfg,
                )

                
                print("\n   🔴 PHASE 2: Holdout 3-way comparison")
                print("   ℹ️ ETA is estimated from completed holdout queries and improves after the first few queries.")
                holdout_phase_start = time.time()

                for i, hq in enumerate(holdout_queries, 1):
                    print(
                        "   ├─ Starting " + _eta_message(
                            "holdout query",
                            max(1, i - 1),
                            len(holdout_queries),
                            holdout_phase_start,
                        ),
                        flush=True,
                    )

                    qt = hq["query_text"]

                    _t0 = time.time()
                    try:
                        baseline_answer = _generate_baseline(qt)
                    except Exception as e:
                        print(f"   ⚠️ Baseline generation failed: {e}")
                        baseline_answer = ""
                    baseline_time = round(time.time() - _t0, 3)

                   
                    _t0 = time.time()
                    kv_prepare_time = 0.0
                    cached_generation_time = 0.0
                    kv_len_used = 0
                    cached_relevant_docs = list(state_knowledge_pool)
                    cache_rel = {
                        "cache_sufficient": True,
                        "num_selected": len(cached_relevant_docs),
                        "max_sim": 1.0 if cached_relevant_docs else 0.0,
                        "mean_sim": 1.0 if cached_relevant_docs else 0.0,
                        "min_sim_threshold": 0.0,
                        "min_docs_required": 1,
                        "query_topk": len(cached_relevant_docs),
                    }

                    if cache_filter_cfg.enabled:
                        cached_relevant_docs, cache_rel = select_relevant_cached_docs(
                            query=qt,
                            cache_index=state_cache_index,
                            encoder=self.retriever.encoder,
                            config=cache_filter_cfg,
                        )

                    if not cache_rel.get("cache_sufficient", False):
                        cached_answer = baseline_answer
                        cached_generation_time = 0.0
                        cached_method = "state_cache_insufficient_baseline"
                    else:
                        
                        use_true_kv = (
                            os.environ.get("DTSS_USE_TRUE_KV", "0") == "1"
                            and not cache_filter_cfg.enabled
                        )
                        if use_true_kv:
                            try:
                                _kv_t0 = time.time()
                                kv_cache, kv_len = self.knowledge_cache.get_kv_cache(
                                    state_id,
                                    self.generator,
                                    top_k=None,
                                    max_chars_per_doc=int(os.environ.get("DTSS_STATE_CAG_MAX_CHARS_PER_DOC", "6000")),
                                )
                                kv_prepare_time = round(time.time() - _kv_t0, 3)
                                kv_len_used = int(kv_len or 0)

                                if kv_cache is not None and kv_len > 0:
                                    _gen_t0 = time.time()
                                    cached_answer = _generate_with_kv(
                                        qt,
                                        kv_cache,
                                        kv_len,
                                        cached_relevant_docs,
                                    )
                                    cached_generation_time = round(time.time() - _gen_t0, 3)
                                    cached_method = "state_cag_kv_cache_legacy_full_cache"
                                else:
                                    _gen_t0 = time.time()
                                    cached_answer = _generate_with_rag(qt, cached_relevant_docs)
                                    cached_generation_time = round(time.time() - _gen_t0, 3)
                                    cached_method = "state_cached_filtered_context"
                            except Exception as e:
                                print(f"    KV-cache injection failed: {e}")
                                _gen_t0 = time.time()
                                cached_answer = _generate_with_rag(qt, cached_relevant_docs)
                                cached_generation_time = round(time.time() - _gen_t0, 3)
                                cached_method = "state_cached_filtered_context"
                        else:
                            _gen_t0 = time.time()
                            cached_answer = _generate_with_rag(qt, cached_relevant_docs)
                            cached_generation_time = round(time.time() - _gen_t0, 3)
                            cached_method = "state_cached_filtered_context" if cache_filter_cfg.enabled else "state_cached_full_context_ablation"

                    cached_time = round(time.time() - _t0, 3)

                    _t0 = time.time()
                    try:
                        fresh_knowledge = self.retriever.retrieve(qt, top_k=fresh_topk)
                        fresh_answer = _generate_with_rag(qt, fresh_knowledge)
                    except Exception as e:
                        print(f"   ⚠️ Fresh RAG failed: {e}")
                        fresh_knowledge = []
                        fresh_answer = ""
                    fresh_time = round(time.time() - _t0, 3)

                    print(f"\n   ── Query {i}: {qt[:80]}...")
                    print(f"   BASELINE   [{baseline_time:.2f}s]: {baseline_answer[:120]}")
                    print(f"   CACHED_RAG [{cached_time:.2f}s | {cached_method} | selected_docs={cache_rel.get('num_selected', 0)} | max_sim={cache_rel.get('max_sim', 0.0):.3f}]: {cached_answer[:120]}")
                    print(f"   FRESH_RAG  [{fresh_time:.2f}s]: {fresh_answer[:120]}")

                    gold_answer = self._get_gold_answer_for_query_record(hq)

                    coverage = self._fresh_cache_coverage(
                        fresh_docs=fresh_knowledge,
                        cached_docs=state_knowledge_pool,
                        k=5,
                    )
                    selected_coverage = self._fresh_cache_coverage(
                        fresh_docs=fresh_knowledge,
                        cached_docs=cached_relevant_docs,
                        k=5,
                    )
                    answer_in_cache = self._answer_in_doc_pool(gold_answer, state_knowledge_pool)
                    answer_in_selected_cache = self._answer_in_doc_pool(gold_answer, cached_relevant_docs)

                    all_holdout_results.append({
                        "window_idx": hq["window_idx"],
                        "original_idx": hq.get("original_idx", -1),
                        "query": qt,
                        "state_id": state_id,
                        "set_type": "holdout",
                        "baseline_answer": baseline_answer,
                        "cached_rag_answer": cached_answer,
                        "fresh_rag_answer": fresh_answer,
                        "gold_answer": gold_answer,
                        "baseline_time_s": baseline_time,
                        "cached_rag_time_s": cached_time,
                        "fresh_rag_time_s": fresh_time,
                        "cached_knowledge_pool_size": len(state_knowledge_pool),
                        "cached_selected_docs": int(cache_rel.get("num_selected", 0)),
                        "cache_sufficient": bool(cache_rel.get("cache_sufficient", False)),
                        "cache_max_similarity": round(float(cache_rel.get("max_sim", 0.0)), 6),
                        "cache_mean_similarity": round(float(cache_rel.get("mean_sim", 0.0)), 6),
                        "cache_similarity_threshold": round(float(cache_rel.get("min_sim_threshold", 0.0)), 6),
                        "cached_rag_method": cached_method,
                        "fresh_retriever": "faiss",
                        "fresh_knowledge_docs": len(fresh_knowledge),
                        "kv_prepare_time_s": kv_prepare_time,
                        "cached_generation_time_s": cached_generation_time,
                        "kv_len": kv_len_used,
                        "state_cache_unique_docs": state_cache_stats["unique_docs"],
                        "state_cache_total_doc_chars": state_cache_stats["total_doc_chars"],
                        "state_cache_mean_doc_chars": round(state_cache_stats["mean_doc_chars"], 3),
                        "state_cache_mean_retrieval_hits_per_doc": round(state_cache_stats["mean_retrieval_hits_per_doc"], 3),
                        "state_cache_max_retrieval_hits_per_doc": state_cache_stats["max_retrieval_hits_per_doc"],
                        "state_cache_reused_doc_fraction": round(state_cache_stats["reused_doc_fraction"], 6),
                        "fresh_cache_coverage_at_5": round(coverage["coverage_at_k"], 6),
                        "fresh_cache_overlap_count_at_5": coverage["overlap_count"],
                        "fresh_cache_fresh_k": coverage["fresh_k"],
                        "selected_cache_coverage_at_5": round(selected_coverage["coverage_at_k"], 6),
                        "selected_cache_overlap_count_at_5": selected_coverage["overlap_count"],
                        "answer_in_state_cache": answer_in_cache,
                        "answer_in_selected_cache": answer_in_selected_cache,
                    })

                    query_total_time = baseline_time + cached_time + fresh_time
                    print(
                        "   │  " + _eta_message(
                            "Completed holdout",
                            i,
                            len(holdout_queries),
                            holdout_phase_start,
                        ) + f" | last_query_total={query_total_time:.2f}s",
                        flush=True,
                    )

                holdout_elapsed = time.time() - holdout_phase_start
                print(
                    f"   └─ ✅ {len(holdout_queries) * 3} holdout answers "
                    f"generated in {_fmt_seconds(holdout_elapsed)}"
                )

            self._save_simplified_analysis(
                all_primary_results,
                all_holdout_results,
            )

            primary_n = len(all_primary_results)
            holdout_n = len(all_holdout_results)
            total_q = primary_n + holdout_n
            saved_by_cached_branch = holdout_n
            pct = saved_by_cached_branch / max(1, total_q) * 100
            cs = self.knowledge_cache.get_stats()

            print(
                f"\n✅ DONE queries={total_q} "
                f"retrievals_saved_by_cached_branch={saved_by_cached_branch} "
                f"({pct:.1f}%)"
            )
            print(
                f"   KV caches built={cs.get('kv_caches_built', 0)} "
                f"hit_rate={cs.get('cache_hit_rate', 0):.3f}"
            )
            print(f"   Total RAG analysis time={_fmt_seconds(time.time() - analysis_start_time)}")

        except Exception as e:
            print(f"\n⚠️ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    def _aggregate_state_cache_diagnostics(self, holdout_results: list) -> dict:
       
        if not holdout_results:
            return {}
        try:
            import numpy as _np
            cov = [float(r.get("fresh_cache_coverage_at_5", 0.0) or 0.0) for r in holdout_results]
            selected_cov = [float(r.get("selected_cache_coverage_at_5", 0.0) or 0.0) for r in holdout_results]
            ans = [int(r.get("answer_in_state_cache", 0) or 0) for r in holdout_results]
            ans_selected = [int(r.get("answer_in_selected_cache", 0) or 0) for r in holdout_results]
            sufficient = [1 if bool(r.get("cache_sufficient", False)) else 0 for r in holdout_results]
            selected_docs = [int(r.get("cached_selected_docs", 0) or 0) for r in holdout_results]
            max_sims = [float(r.get("cache_max_similarity", 0.0) or 0.0) for r in holdout_results]
            docs = [int(r.get("state_cache_unique_docs", 0) or 0) for r in holdout_results]
            chars = [int(r.get("state_cache_total_doc_chars", 0) or 0) for r in holdout_results]
            hits = [float(r.get("state_cache_mean_retrieval_hits_per_doc", 0.0) or 0.0) for r in holdout_results]
            reused = [float(r.get("state_cache_reused_doc_fraction", 0.0) or 0.0) for r in holdout_results]
            return {
                "num_holdout_queries": int(len(holdout_results)),
                "cache_sufficient_rate": float(_np.mean(sufficient)) if sufficient else 0.0,
                "avg_selected_cached_docs_per_holdout": float(_np.mean(selected_docs)) if selected_docs else 0.0,
                "avg_cache_max_similarity": float(_np.mean(max_sims)) if max_sims else 0.0,
                "full_state_cache_coverage_at_5_mean": float(_np.mean(cov)) if cov else 0.0,
                "selected_cache_coverage_at_5_mean": float(_np.mean(selected_cov)) if selected_cov else 0.0,
                "coverage_at_5_mean": float(_np.mean(cov)) if cov else 0.0,
                "coverage_at_5_median": float(_np.median(cov)) if cov else 0.0,
                "answer_in_cache_rate": float(_np.mean(ans)) if ans else 0.0,
                "answer_in_selected_cache_rate": float(_np.mean(ans_selected)) if ans_selected else 0.0,
                "avg_unique_docs_per_state_seen_by_holdout": float(_np.mean(docs)) if docs else 0.0,
                "avg_state_cache_chars_seen_by_holdout": float(_np.mean(chars)) if chars else 0.0,
                "avg_retrieval_hits_per_cached_doc": float(_np.mean(hits)) if hits else 0.0,
                "avg_reused_doc_fraction": float(_np.mean(reused)) if reused else 0.0,
                "holdout_retrieval_reduction_cached_branch": 1.0,
            }
        except Exception as e:
            return {"error": str(e)}

    def _save_simplified_analysis(self, primary_results, holdout_results):
        primary_file = os.path.join(self.experiment_dir, "primary_answers.csv")
        with open(primary_file, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['window_idx','original_idx','query','state_id','answer','knowledge_source'])
            w.writeheader()
            for r in primary_results:
                w.writerow({k: r.get(k, '') for k in ['window_idx','original_idx','query','state_id','answer','knowledge_source']})
        print(f"   ✅ {primary_file}")

        holdout_file = os.path.join(self.experiment_dir, "holdout_3method_comparison.csv")
        with open(holdout_file, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'window_idx','original_idx','query','state_id',
                'gold_answer',
                'baseline_answer','cached_rag_answer','fresh_rag_answer',
                'baseline_time_s','cached_rag_time_s','fresh_rag_time_s',
                'cached_rag_method','cache_sufficient','cached_selected_docs',
                'cache_max_similarity','cache_mean_similarity','cache_similarity_threshold',
                'kv_prepare_time_s','cached_generation_time_s','kv_len',
                'state_cache_unique_docs','state_cache_total_doc_chars',
                'state_cache_mean_retrieval_hits_per_doc','state_cache_reused_doc_fraction',
                'fresh_cache_coverage_at_5','fresh_cache_overlap_count_at_5',
                'selected_cache_coverage_at_5','selected_cache_overlap_count_at_5',
                'answer_in_state_cache','answer_in_selected_cache',
            ])
            w.writeheader()
            for r in holdout_results:
                w.writerow({k: r.get(k,'') for k in [
                    'window_idx','original_idx','query','state_id',
                    'gold_answer',
                    'baseline_answer','cached_rag_answer','fresh_rag_answer',
                    'baseline_time_s','cached_rag_time_s','fresh_rag_time_s',
                    'cached_rag_method','cache_sufficient','cached_selected_docs',
                    'cache_max_similarity','cache_mean_similarity','cache_similarity_threshold',
                    'kv_prepare_time_s','cached_generation_time_s','kv_len',
                    'state_cache_unique_docs','state_cache_total_doc_chars',
                    'state_cache_mean_retrieval_hits_per_doc','state_cache_reused_doc_fraction',
                    'fresh_cache_coverage_at_5','fresh_cache_overlap_count_at_5',
                    'selected_cache_coverage_at_5','selected_cache_overlap_count_at_5',
                    'answer_in_state_cache','answer_in_selected_cache',
                ]})
        print(f"   ✅ {holdout_file}")

        full_file = os.path.join(self.experiment_dir, "holdout_full_results.json")
        with open(full_file, 'w', encoding='utf-8') as f:
            json.dump(holdout_results, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {full_file}")

        total_q = len(primary_results) + len(holdout_results)
        topic_retrievals = int(getattr(self, "topic_retrieval_count_total", 0))

       
        if all_topic_usage_rows:
            topic_usage_df = pd.DataFrame(all_topic_usage_rows)

            topic_usage_csv = os.path.join(
                self.experiment_dir,
                "state_topic_usage_counts.csv",
            )
            topic_usage_df.to_csv(topic_usage_csv, index=False)
            print(f"   ✅ {topic_usage_csv}")

            global_topic_df = (
                topic_usage_df
                .groupby("topic_query_norm", as_index=False)
                .agg(
                    topic_query=("topic_query", "first"),
                    states_using_topic=("state_id", "nunique"),
                    total_support_any_count=("support_any_count", "sum"),
                    total_support_all_count=("support_all_count", "sum"),
                    total_exact_phrase_count=("exact_phrase_count", "sum"),
                    mean_support_any_rate=("support_any_rate", "mean"),
                    mean_support_all_rate=("support_all_rate", "mean"),
                    mean_exact_phrase_rate=("exact_phrase_rate", "mean"),
                    total_topic_retrieval_count=("topic_retrieval_count", "sum"),
                )
                .sort_values(
                    ["states_using_topic", "total_support_any_count"],
                    ascending=[False, False],
                )
            )

            global_topic_csv = os.path.join(
                self.experiment_dir,
                "global_topic_usage_summary.csv",
            )
            global_topic_df.to_csv(global_topic_csv, index=False)
            print(f"   ✅ {global_topic_csv}")

            topic_usage_json = os.path.join(
                self.experiment_dir,
                "global_topic_usage_summary.json",
            )
            topic_usage_payload = {
                "num_state_topic_rows": int(len(topic_usage_df)),
                "num_unique_topic_queries": int(topic_usage_df["topic_query_norm"].nunique()),
                "top_topics": global_topic_df.head(30).to_dict(orient="records"),
                "explanation": (
                    "states_using_topic counts how many discovered states use the topic. "
                    "total_support_any_count counts how many primary/cache-building queries "
                    "contain at least one term from the topic query. These statistics can be "
                    "used to analyze recurring topics, support dataset construction, and motivate "
                    "recommendation-system extensions."
                ),
            }
            with open(topic_usage_json, "w", encoding="utf-8") as f:
                json.dump(topic_usage_payload, f, indent=2, ensure_ascii=False)
            print(f"   ✅ {topic_usage_json}")

            topic_usage_summary = {
                "num_state_topic_rows": int(len(topic_usage_df)),
                "num_unique_topic_queries": int(topic_usage_df["topic_query_norm"].nunique()),
                "top_topics": global_topic_df.head(10).to_dict(orient="records"),
                "files": {
                    "state_topic_usage_counts_csv": "state_topic_usage_counts.csv",
                    "global_topic_usage_summary_csv": "global_topic_usage_summary.csv",
                    "global_topic_usage_summary_json": "global_topic_usage_summary.json",
                },
            }
        else:
            topic_usage_summary = {
                "num_state_topic_rows": 0,
                "num_unique_topic_queries": 0,
                "top_topics": [],
                "files": {},
            }

        summary = {
            'experiment_metadata': {
                'total_queries': total_q,
                'cache_build_queries_with_retrieval': len(primary_results),
                'holdout_queries': len(holdout_results),
                'holdout_answer_cap': max_total_holdout_questions,
                'min_rag_state_queries': min_rag_state_queries,
                'fresh_rag_topk': fresh_topk,
            },
            'cached_rag_alignment': {
                'methods_observed': sorted(set(str(r.get('cached_rag_method', 'unknown')) for r in holdout_results)),
                'default_method': 'state_cached_filtered_context when cached context passes threshold; otherwise state_cache_insufficient_baseline',
                'state_cache_filtering_env': 'DTSS_STATE_CACHE_FILTERING=1 by default; DTSS_USE_TRUE_KV=0 by default in filtered mode',
                'model': 'microsoft/Phi-3.5-mini-instruct',
                'note': 'Cached RAG first filters state-cache documents by dense query/document similarity. Full-state KV cache is treated as a legacy ablation.',
            },
            'topic_usage_summary': topic_usage_summary,
            'retrieval_statistics': {
                'baseline_retrievals_if_all_fresh_rag': total_q,
                'actual_primary_retrievals': len(primary_results),
                'topic_expansion_retrievals': topic_retrievals,
                'actual_cache_construction_retrievals_total': len(primary_results) + topic_retrievals,
                'fresh_rag_holdout_retrievals': len(holdout_results),
                'cached_rag_holdout_retrievals': 0,
                'retrievals_saved_by_cached_branch': len(holdout_results),
                'holdout_retrieval_reduction_pct': 100.0 if len(holdout_results) else 0.0,
                'amortized_retrieval_reduction_vs_fresh_holdout_pct': round(
                    (1.0 - ((len(primary_results) + topic_retrievals) / max(len(holdout_results), 1))) * 100.0,
                    3,
                ),
                'amortized_note': (
                    'Cached RAG performs no retrieval at holdout time. The amortized value includes upfront '
                    'cache-building and topic-expansion retrievals, so it improves as more future queries reuse the same state cache.'
                ),
            },
            'answers_generated': {
                'total_answers': len(primary_results) + len(holdout_results) * 3,
            },
            'state_cache_diagnostics': self._aggregate_state_cache_diagnostics(holdout_results),
        }
        sf = os.path.join(self.experiment_dir, "analysis_summary.json")
        with open(sf, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ {sf}")

      
        self._plot_timing_comparison(holdout_results)

    
        self._plot_rouge_vs_gold(holdout_results)

    
        self._run_gold_answer_evaluation(holdout_results)

        
        try:
            log_path = os.path.join(self.experiment_dir, "log")

            primary_n = len(primary_results)
            holdout_n = len(holdout_results)
            total_q = primary_n + holdout_n

            if holdout_results:
                import numpy as _np

                baseline_times = [
                    float(r.get("baseline_time_s", 0.0) or 0.0)
                    for r in holdout_results
                ]
                cached_times = [
                    float(r.get("cached_rag_time_s", 0.0) or 0.0)
                    for r in holdout_results
                ]
                fresh_times = [
                    float(r.get("fresh_rag_time_s", 0.0) or 0.0)
                    for r in holdout_results
                ]

                avg_baseline_time = float(_np.mean(baseline_times)) if baseline_times else 0.0
                avg_cached_time = float(_np.mean(cached_times)) if cached_times else 0.0
                avg_fresh_time = float(_np.mean(fresh_times)) if fresh_times else 0.0
                speedup_vs_fresh = avg_fresh_time / avg_cached_time if avg_cached_time > 0 else 0.0
                cached_methods = sorted(set(
                    str(r.get("cached_rag_method", "unknown"))
                    for r in holdout_results
                ))
            else:
                avg_baseline_time = 0.0
                avg_cached_time = 0.0
                avg_fresh_time = 0.0
                speedup_vs_fresh = 0.0
                cached_methods = []

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n=== RAG / CACHED RETRIEVAL RESULTS ===\n")
                f.write(f"Total RAG queries      : {total_q}\n")
                f.write(f"Cache-build retrievals : {primary_n}\n")
                f.write(f"Holdout queries        : {holdout_n}\n")
                f.write(
                    "Cached RAG method      : "
                    f"{', '.join(cached_methods) if cached_methods else 'N/A'}\n"
                )
                f.write(f"Retrievals saved by cached branch : {holdout_n}\n")

                f.write("\n--- Timing ---\n")
                f.write(f"Baseline avg time      : {avg_baseline_time:.3f}s\n")
                f.write(f"Cached RAG avg time    : {avg_cached_time:.3f}s\n")
                f.write(f"Fresh RAG avg time     : {avg_fresh_time:.3f}s\n")
                f.write(f"Cached/Fresh speedup   : {speedup_vs_fresh:.3f}x\n")

                if hasattr(self, "knowledge_cache") and self.knowledge_cache is not None:
                    cs = self.knowledge_cache.get_stats()
                    f.write("\n--- Cache statistics ---\n")
                    f.write(f"States cached          : {cs.get('cache_size', 0)}\n")
                    f.write(f"Cache hits             : {cs.get('cache_hits', 0)}\n")
                    f.write(f"Cache misses           : {cs.get('cache_misses', 0)}\n")
                    f.write(f"Cache hit rate         : {cs.get('cache_hit_rate', 0.0):.3f}\n")
                    f.write(f"Documents cached       : {cs.get('total_docs_cached', 0)}\n")
                    f.write(f"KV caches built        : {cs.get('kv_caches_built', 0)}\n")

                f.write("\nGenerated RAG files:\n")
                f.write("  - rag_metrics.txt\n")
                f.write("  - rag_answers.csv\n")
                f.write("  - primary_answers.csv\n")
                f.write("  - holdout_3method_comparison.csv\n")
                f.write("  - holdout_full_results.json\n")
                f.write("  - analysis_summary.json\n")
                f.write("  - timing_comparison.png\n")
                f.write("  - timing_per_query.csv\n")
                f.write("  - rouge_scores.csv / rouge_aggregate.json if gold answers exist\n")

            print(f"    RAG summary appended to main log → {log_path}")

        except Exception as e:
            print(f"    Could not append RAG summary to log: {e}")

    def _run_gold_answer_evaluation(self, holdout_results: list):
        
        try:
            from .rag_evaluation import AnswerComparator

            comparator = AnswerComparator(rouge_threshold=0.3)

          
            comparisons = []
            for r in holdout_results:
                gold = r.get("gold_answer", "").strip()
                comp = comparator.compare_answers(
                    query             = r.get("query", ""),
                    baseline_answer   = r.get("baseline_answer", ""),
                    cached_rag_answer = r.get("cached_rag_answer", ""),
                    fresh_rag_answer  = r.get("fresh_rag_answer", ""),
                    gold_answer       = gold,
                )
                comparisons.append(comp)

            if not comparisons:
                print("     No comparisons — skipping gold evaluation")
                return

          
            aggregate = comparator.aggregate_comparison(comparisons)

           
            comparator.print_comparison_summary(aggregate)

          
            thresholds = comparator.accuracy_at_thresholds(
                comparisons,
                thresholds=[0.1, 0.2, 0.3, 0.5],
                metric="rougeL",
            )
            print("\n ACCURACY AT MULTIPLE THRESHOLDS (ROUGE-L)")
            print(f"   {'Threshold':>10} {'Baseline':>10} {'Cached RAG':>12} {'Fresh RAG':>10}")
            print("   " + "-" * 46)
            for t, acc in sorted(thresholds.items()):
                print(f"   {t:>10.1f} "
                      f"{acc['baseline_acc']:>9.1f}% "
                      f"{acc['cached_rag_acc']:>11.1f}% "
                      f"{acc['fresh_rag_acc']:>9.1f}%")

            csv_path = os.path.join(
                self.experiment_dir, "rouge_scores.csv"
            )
            comparator.save_per_query_results(comparisons, csv_path)

            
            import json as _json
            agg_path = os.path.join(
                self.experiment_dir, "rag_answer_quality_aggregate.json"
            )
            with open(agg_path, "w") as f:
                _json.dump(aggregate, f, indent=2)
            print(f"   ✅ Answer-quality aggregate → {agg_path}")

           
            old_agg_path = os.path.join(self.experiment_dir, "rouge_aggregate.json")
            with open(old_agg_path, "w") as f:
                _json.dump(aggregate, f, indent=2)
            print(f"   ✅ ROUGE aggregate → {old_agg_path}")

            cache_diag = self._aggregate_state_cache_diagnostics(holdout_results)
            cache_diag_path = os.path.join(self.experiment_dir, "state_cache_diagnostics.json")
            with open(cache_diag_path, "w") as f:
                _json.dump(cache_diag, f, indent=2)
            print(f"   ✅ State-cache diagnostics → {cache_diag_path}")

            
            thr_path = os.path.join(
                self.experiment_dir, "accuracy_at_thresholds.json"
            )
            with open(thr_path, "w") as f:
                _json.dump(
                    {str(k): v for k, v in thresholds.items()},
                    f, indent=2
                )
            print(f"   ✅ Threshold accuracy → {thr_path}")

        except Exception as e:
            print(f"   ⚠️  Gold answer evaluation failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    def _plot_timing_comparison(self, holdout_results: list):
        """
        Bar chart + per-query scatter comparing generation time for:
          - baseline   (parametric, no context)
          - cached_rag (state cached context by default; optional KV injection)
          - fresh_rag  (retrieval + generation — slowest)
        Also saves timing_per_query.csv for further analysis.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            b_times = [r["baseline_time_s"]   for r in holdout_results if "baseline_time_s"   in r]
            c_times = [r["cached_rag_time_s"]  for r in holdout_results if "cached_rag_time_s"  in r]
            f_times = [r["fresh_rag_time_s"]   for r in holdout_results if "fresh_rag_time_s"   in r]

            if not (b_times and c_times and f_times):
                print("   ⚠️  No timing data — skipping graph")
                return

            avg_b, std_b = float(np.mean(b_times)), float(np.std(b_times))
            avg_c, std_c = float(np.mean(c_times)), float(np.std(c_times))
            avg_f, std_f = float(np.mean(f_times)), float(np.std(f_times))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

      
            cached_methods = sorted(set(str(r.get("cached_rag_method", "state_cached_context")) for r in holdout_results))
            cached_label = "Cached RAG\n(KV injection)" if cached_methods == ["kv_cache_injection"] else "Cached RAG\n(state context)"
            methods = ["Baseline\n(no context)", cached_label, "Fresh RAG\n(retrieval+gen)"]
            avgs    = [avg_b, avg_c, avg_f]
            stds    = [std_b, std_c, std_f]
            colors  = ["#4878CF", "#6ACC65", "#D65F5F"]
            bars = axes[0].bar(methods, avgs, color=colors, alpha=0.85,
                               yerr=stds, capsize=6, edgecolor="white", linewidth=1.2)
            axes[0].set_title("Average Generation Time per Method", fontsize=13, fontweight="bold")
            axes[0].set_ylabel("Time (seconds)", fontsize=11)
            axes[0].set_ylim(0, max(avgs) * 1.4)
            for bar, avg in zip(bars, avgs):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(avgs) * 0.02,
                    f"{avg:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold"
                )
            if avg_f > 0 and avg_c > 0:
                speedup = avg_f / avg_c
                axes[0].annotate(
                    f"Cached RAG is\n{speedup:.1f}x faster\nthan Fresh RAG",
                    xy=(1, avg_c), xytext=(1.35, avg_f * 0.55),
                    fontsize=10, color="#6ACC65", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#6ACC65", lw=1.5)
                )
            axes[0].grid(axis="y", alpha=0.3)
            axes[0].spines[["top","right"]].set_visible(False)

       
            n  = min(len(b_times), len(c_times), len(f_times))
            xs = list(range(1, n + 1))
            axes[1].plot(xs, b_times[:n], "o-", color="#4878CF", label="Baseline",   alpha=0.7, markersize=4)
            axes[1].plot(xs, c_times[:n], "s-", color="#6ACC65", label="Cached RAG", alpha=0.7, markersize=4)
            axes[1].plot(xs, f_times[:n], "^-", color="#D65F5F", label="Fresh RAG",  alpha=0.7, markersize=4)
            axes[1].set_title("Generation Time per Query", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Query index", fontsize=11)
            axes[1].set_ylabel("Time (seconds)", fontsize=11)
            axes[1].legend(fontsize=10)
            axes[1].grid(alpha=0.3)
            axes[1].spines[["top","right"]].set_visible(False)

            plt.suptitle(
                "Cached RAG vs Baseline vs Fresh RAG — Generation Time",
                fontsize=13, fontweight="bold", y=1.02
            )
            plt.tight_layout()
            plot_path = os.path.join(self.experiment_dir, "timing_comparison.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   ✅ Timing graph   → {plot_path}")

          
            timing_csv = os.path.join(self.experiment_dir, "timing_per_query.csv")
            with open(timing_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_idx","query","state_id",
                             "baseline_time_s","cached_rag_time_s","fresh_rag_time_s",
                             "speedup_vs_fresh"])
                for idx, r in enumerate(holdout_results):
                    bt = r.get("baseline_time_s",  0)
                    ct = r.get("cached_rag_time_s", 0)
                    ft = r.get("fresh_rag_time_s",  0)
                    speedup = round(ft / ct, 2) if ct > 0 else 0
                    w.writerow([idx+1, r.get("query","")[:80],
                                r.get("state_id",""), bt, ct, ft, speedup])
            print(f"   ✅ Timing CSV     → {timing_csv}")

            print(f"\n   📊 TIMING SUMMARY:")
            print(f"      Baseline   : {avg_b:.3f}s avg ± {std_b:.3f}s")
            print(f"      Cached RAG : {avg_c:.3f}s avg ± {std_c:.3f}s  ({', '.join(cached_methods)})")
            print(f"      Fresh RAG  : {avg_f:.3f}s avg ± {std_f:.3f}s  (retrieval + gen)")
            if avg_f > 0 and avg_c > 0:
                print(f"      Speedup    : {avg_f/avg_c:.1f}x faster than Fresh RAG")

        except Exception as e:
            print(f"   ⚠️  Timing graph failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    def _plot_rouge_vs_gold(self, holdout_results: list):
        """
        Plot ROUGE-1/2/L for baseline, cached RAG, fresh RAG vs gold answer.
        Also shows threshold-based accuracy (ROUGE-L >= 0.3).
        All three methods are compared against the actual gold answer — not
        fresh RAG as proxy. Fresh RAG gets a real score just like the others.
        """
        try:
            from rouge_score import rouge_scorer as rs_lib
        except ImportError as e:
            print(
                "   ⚠️  rouge-score is not installed; skipping ROUGE graph. "
                "Install it with: pip install rouge-score"
            )
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            scorer = rs_lib.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

  
            scored = [r for r in holdout_results if r.get("gold_answer", "").strip()]
            if not scored:
                print("   ⚠️  No gold answers found — skipping ROUGE graph")
                return

            n = len(scored)
            print(f"   📊 Computing ROUGE vs gold for {n} queries...")

            methods      = ["baseline_answer", "cached_rag_answer", "fresh_rag_answer"]
            method_names = ["Baseline\n(no retrieval)",
                            "Cached RAG\n(state context)",
                            "Fresh RAG\n(per-query retrieval)"]
            colors       = ["#4878CF", "#6ACC65", "#D65F5F"]
            threshold    = 0.3

            scores_per_method = {m: {"rouge1": [], "rouge2": [], "rougeL": []}
                                  for m in methods}

            for r in scored:
                gold = r["gold_answer"].strip()
                for method in methods:
                    pred   = r.get(method, "").strip() or ""
                    result = scorer.score(gold, pred)
                    for rtype in ["rouge1", "rouge2", "rougeL"]:
                        scores_per_method[method][rtype].append(result[rtype].fmeasure)

            avg = {m: {rt: float(np.mean(scores_per_method[m][rt]))
                       for rt in ["rouge1", "rouge2", "rougeL"]}
                   for m in methods}

            acc = {m: float(np.mean([
                1 if s >= threshold else 0
                for s in scores_per_method[m]["rougeL"]
            ])) * 100 for m in methods}

            rouge_types  = ["rouge1", "rouge2", "rougeL"]
            rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            x     = np.arange(len(methods))
            width = 0.55

            for ax, rtype, rlabel in zip(axes[:3], rouge_types, rouge_labels):
                vals = [avg[m][rtype] for m in methods]
                bars = ax.bar(x, vals, width=width, color=colors,
                              alpha=0.85, edgecolor="white", linewidth=1.2)
                ax.set_title(rlabel, fontsize=13, fontweight="bold")
                ax.set_ylabel("F1 Score (vs Gold Answer)", fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(method_names, fontsize=9)
                ax.set_ylim(0, max(max(vals) * 1.45, 0.1))
                ax.grid(axis="y", alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals) * 0.02,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=11, fontweight="bold")
                gain = avg["cached_rag_answer"][rtype] - avg["baseline_answer"][rtype]
                ax.text(0.98, 0.97, f"Cached gain\nvs baseline: {gain:+.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color="#6ACC65", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax4      = axes[3]
            acc_vals = [acc[m] for m in methods]
            bars4    = ax4.bar(x, acc_vals, width=width, color=colors,
                               alpha=0.85, edgecolor="white", linewidth=1.2)
            ax4.set_title(f"Accuracy\n(ROUGE-L ≥ {threshold})", fontsize=13, fontweight="bold")
            ax4.set_ylabel("% Correct Answers", fontsize=10)
            ax4.set_xticks(x)
            ax4.set_xticklabels(method_names, fontsize=9)
            ax4.set_ylim(0, max(max(acc_vals) * 1.45, 10))
            ax4.grid(axis="y", alpha=0.3)
            ax4.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars4, acc_vals):
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(acc_vals) * 0.02,
                         f"{val:.1f}%", ha="center", va="bottom",
                         fontsize=11, fontweight="bold")
            gain_acc = acc["cached_rag_answer"] - acc["baseline_answer"]
            ax4.text(0.98, 0.97, f"Cached gain\nvs baseline: {gain_acc:+.1f}%",
                     transform=ax4.transAxes, ha="right", va="top",
                     fontsize=9, color="#6ACC65", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.suptitle(
                f"ROUGE vs Gold Answer — Baseline / Cached RAG / Fresh RAG  ({n} holdout queries)",
                fontsize=13, fontweight="bold", y=1.02
            )
            plt.tight_layout()
            plot_path = os.path.join(self.experiment_dir, "rouge_vs_gold.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   ✅ ROUGE graph → {plot_path}")

            rouge_csv = os.path.join(self.experiment_dir, "rouge_scores.csv")
            with open(rouge_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_idx", "query", "gold_answer",
                             "baseline_r1", "baseline_r2", "baseline_rL", "baseline_correct",
                             "cached_rag_r1", "cached_rag_r2", "cached_rag_rL", "cached_rag_correct",
                             "fresh_rag_r1", "fresh_rag_r2", "fresh_rag_rL", "fresh_rag_correct"])
                for idx, r in enumerate(scored):
                    gold = r["gold_answer"].strip()
                    row  = [idx + 1, r.get("query", "")[:60], gold[:60]]
                    for method in methods:
                        pred   = r.get(method, "").strip() or ""
                        result = scorer.score(gold, pred)
                        rL     = round(result["rougeL"].fmeasure, 4)
                        row += [round(result["rouge1"].fmeasure, 4),
                                round(result["rouge2"].fmeasure, 4),
                                rL, 1 if rL >= threshold else 0]
                    w.writerow(row)
            print(f"   ✅ ROUGE CSV → {rouge_csv}")

            print(f"\n   📊 ROUGE vs GOLD SUMMARY ({n} queries):")
            print(f"   {'Method':<28} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'Acc@0.3':>9}")
            print(f"   {'-'*56}")
            for method, name in zip(methods, ["Baseline", "Cached RAG", "Fresh RAG"]):
                print(f"   {name:<28} "
                      f"{avg[method]['rouge1']:>7.3f} "
                      f"{avg[method]['rouge2']:>7.3f} "
                      f"{avg[method]['rougeL']:>7.3f} "
                      f"{acc[method]:>8.1f}%")
            print(f"\n   Gains vs Baseline:")
            for rtype, rlabel in zip(rouge_types, rouge_labels):
                c = avg["cached_rag_answer"][rtype] - avg["baseline_answer"][rtype]
                f = avg["fresh_rag_answer"][rtype]  - avg["baseline_answer"][rtype]
                print(f"     {rlabel}: Cached {c:+.3f}  Fresh {f:+.3f}")
            print(f"     Accuracy: Cached {acc['cached_rag_answer']-acc['baseline_answer']:+.1f}%"
                  f"  Fresh {acc['fresh_rag_answer']-acc['baseline_answer']:+.1f}%")

        except Exception as e:
            print(f"     ROUGE graph failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    def _save_rag_metrics(self):
        if not self.enable_rag:
            return

        cs = self.knowledge_cache.get_stats() if self.knowledge_cache else {}
        ps = self.retriever.get_retrieval_stats() if self.retriever else {}
        fs = self.fresh_retriever.get_retrieval_stats() if self.fresh_retriever else {}

        p = os.path.join(self.experiment_dir, "rag_metrics.txt")
        with open(p, "w") as f:
            f.write("=== RAG Metrics ===\n\n")
            f.write(f"KB size          : {ps.get('knowledge_base_size','N/A')}\n")
            f.write(f"Primary retrievals: {ps.get('total_retrievals','N/A')}\n")
            f.write(f"Web searches     : {fs.get('total_retrievals','N/A')}\n\n")
            f.write(f"States cached    : {cs.get('cache_size', 0)}\n")
            f.write(f"KV caches built  : {cs.get('kv_caches_built',0)}\n")
            f.write(f"Cache hits       : {cs.get('cache_hits', 0)}\n")
            f.write(f"Cache misses     : {cs.get('cache_misses', 0)}\n")
            f.write(f"Hit rate         : {cs.get('cache_hit_rate', 0.0):.3f}\n")
            f.write(f"Retrieval reduction: {cs.get('retrieval_reduction', 0.0):.3f}\n\n")
            f.write("Cached RAG method: compact query-filtered state cache by default; DTSS_USE_TRUE_KV=0 is set in code for filtered mode\n")
            f.write("Model            : microsoft/Phi-3.5-mini-instruct\n")
        print(f"✅ RAG metrics → {p}")

        if self.rag_answers:
            ap = os.path.join(self.experiment_dir, "rag_answers.csv")
            pd.DataFrame(self.rag_answers).to_csv(ap, index=False)
            print(f" RAG answers → {ap}")
            try:
                from .rag_evaluation import RAGEvaluator
                ev = RAGEvaluator().evaluate_answers(
                    pd.DataFrame(self.rag_answers),
                    retrieved_knowledge=self.knowledge_cache.cache if self.knowledge_cache else {},
                )
                ep = os.path.join(self.experiment_dir, "rag_evaluation.json")
                with open(ep, 'w') as f:
                    json.dump(ev, f, indent=2)
                print(f" RAG evaluation → {ep}")
            except Exception as e:
                print(f" Evaluation scores failed: {e}")

        if self.retriever is not None and self.knowledge_cache is not None and self.generator is not None:
            if not hasattr(self, 'state_assignments'):
                self._run_baseline_rag_comparison()
            else:
                self._run_simplified_efficiency_analysis()
        else:
            print("Skipping efficiency analysis: RAG components not initialised.")
    def _export_state_representations(self, state_protos, state_samples_all,
                                       state_labels_all, state_counts, uf_parent,
                                       state_Pi_stad=None) -> dict:
        state_repr = {}
        for sid in sorted(state_protos.keys()):
            canon = self._uf_find(uf_parent, sid)
            if canon != sid:
                continue
            Ni     = int(state_counts.get(sid, 0))
            mu_ema = state_protos[sid].tolist()
            if sid in state_samples_all and len(state_samples_all[sid]) > 0:
                samples   = state_samples_all[sid]
                mu_batch  = samples.mean(axis=0).tolist()
                Sigma     = self._safe_cov(samples)
                sigma_diag = np.diag(Sigma).tolist()
                sigma_full = Sigma.tolist()
                Pi_stad   = (state_Pi_stad[sid] if state_Pi_stad and sid in state_Pi_stad
                             else self._stad_density_estimation(samples))
                labels    = state_labels_all.get(sid, [])
                Pi_labels = ({int(l): round(labels.count(l) / len(labels), 4) for l in sorted(set(labels))}
                             if labels else {})
            else:
                mu_batch   = mu_ema; sigma_diag = [0.0]; sigma_full = [[0.0]]
                Pi_stad    = np.ones(len(mu_ema)) / max(1, len(mu_ema)); Pi_labels = {}

            state_repr[int(sid)] = {
                "state_id": int(sid), "Ni": Ni,
                "mu_i": mu_batch, "mu_i_creation": mu_ema,
                "sigma_i_diag": sigma_diag, "sigma_i_full": sigma_full,
                "Pi_stad": Pi_stad.tolist(), "Pi_labels": Pi_labels,
            }

        kld_matrix = {}
        for i in state_repr:
            kld_matrix[i] = {
                j: round(self._symmetric_kld_stad(
                    np.array(state_repr[i]["Pi_stad"]),
                    np.array(state_repr[j]["Pi_stad"])), 6)
                for j in state_repr
            }

        rf = os.path.join(self.experiment_dir, "state_representations.json")
        with open(rf, "w") as f:
            json.dump({"states": state_repr, "pairwise_kld": kld_matrix}, f, indent=2)
        print(f"State representations → {rf}")

        sf = os.path.join(self.experiment_dir, "state_summary.txt")
        with open(sf, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("STATE REPRESENTATIONS:  Si = ( Pi , ( \u03bci , \u03a3i ) )\n")
            f.write("=" * 70 + "\n\n")
            f.write("  where:\n")
            f.write("    Pi    = softmax(Z, axis=1).mean(axis=0)   [original STAD formula]\n")
            f.write("          Z = (Ni x D) matrix of window embeddings for state i\n")
            f.write("          Softmax applied row-wise, then column mean -> D-dim prob vector\n")
            f.write("    \u03bci    = L2-normalised mean embedding of the window that created the state (fixed)\n")
            f.write("    \u03a3i    = Sample covariance of all window embeddings assigned to state i\n\n")
            f.write("-" * 70 + "\n\n")
            for sid, rep in state_repr.items():
                pi       = np.array(rep["Pi_stad"])
                mu       = np.array(rep["mu_i"])
                sig_diag = np.array(rep["sigma_i_diag"])
                top5_idx = np.argsort(pi)[::-1][:5]
                pi_str   = "  ".join(f"d{d}:{pi[d]:.4f}" for d in top5_idx)
                pi_sum   = float(pi.sum())
                mu_norm  = float(np.linalg.norm(mu))
                sig_mean = float(np.mean(sig_diag)) if len(sig_diag) > 1 else 0.0
                lstr     = " ".join(f"c{k}:{v:.2f}" for k, v in rep["Pi_labels"].items())
                f.write(f"S{sid}  (Ni = {rep['Ni']} windows)\n")
                f.write(f"  Pi (top-5) : {pi_str}  [sum={pi_sum:.4f}]\n")
                f.write(f"  ||\u03bci||    : {mu_norm:.6f}\n")
                f.write(f"  mean Var(\u03a3i): {sig_mean:.6f}\n")
                f.write(f"  Pi_labels  : {lstr if lstr else 'n/a'}\n\n")
            f.write("=" * 70 + "\n")

        print("\n" + "=" * 70)
        print("STATE REPRESENTATIONS:  Si = ( Pi , ( \u03bci , \u03a3i ) )")
        print("=" * 70)
        for sid, rep in state_repr.items():
            pi       = np.array(rep["Pi_stad"])
            mu       = np.array(rep["mu_i"])
            sig_diag = np.array(rep["sigma_i_diag"])

            # top-5 dims by Pi value
            top5_idx = np.argsort(pi)[::-1][:5]
            pi_str   = "  ".join(f"d{d}:{pi[d]:.4f}" for d in top5_idx)
            pi_sum   = float(pi.sum())

            mu_norm  = float(np.linalg.norm(mu))
            sig_mean = float(np.mean(sig_diag)) if len(sig_diag) > 1 else 0.0
            lstr     = " ".join(f"c{k}:{v:.2f}" for k, v in rep["Pi_labels"].items())

            print(f"\n  S{sid}  (Ni = {rep['Ni']} windows)")
            print(f"    Pi (top-5) : {pi_str}  [sum={pi_sum:.4f}]")
            print(f"    \u03bci norm   : {mu_norm:.6f}    mean Var(\u03a3i): {sig_mean:.6f}")
            print(f"    Pi_labels  : {lstr if lstr else 'n/a'}")

        print("\n" + "=" * 70)
        print("FORMAL DEFINITION:")
        print("  Pi     = softmax(Z, axis=1).mean(axis=0)   [original STAD formula]")
        print("         where Z = (Ni x D) matrix of window embeddings for state i")
        print("         Softmax applied row-wise, then column mean → D-dim prob vector")
        print("  \u03bci     = L2-normalised mean embedding at state creation (fixed, not updated)")
        print("  \u03a3i     = Cov({ z_k | k assigned to state i })")
        print("=" * 70 + "\n")
        return state_repr

    # ─────────────────────────── MAIN PREDICT LOOP ───────────────────────────

    def predict(self, test_set, class_labels, drift_labels, kld_epsilon):
        mode = str(self.drift_mode).lower().strip()

        # The simple hand-written SciPy KS detector was removed from this thesis
        # code path. Use Alibi KS for KS-based drift detection.
        if mode == "ks":
            print("⚠️ drift_mode='ks' has been removed. Using drift_mode='ks_alibi' instead.")
            mode = "ks_alibi"

        use_alibi_ks = mode in ("ks_alibi", "alibi_ks")
        use_mmd = mode == "mmd"
        use_energy = mode == "energy"

        if not (use_alibi_ks or use_mmd or use_energy):
            use_mmd = True
            mode = "mmd"

        # Number of permutations used to estimate p-values for MMD/Energy.
        # Increase this for final runs, for example 199 or 499.
        detector_n_permutations = int(os.environ.get("DTSS_DETECTOR_N_PERMUTATIONS", "99"))

        ks = None
        ks_alibi = AlibiKSTest(
            self.detector_alpha,
            self.l_hist_size,
            self.l_new_size,
            self.ks_hist_min_size,
        ) if use_alibi_ks else None

        mmd = MMDTest(
            self.l_hist_size,
            self.l_new_size,
            self.ks_hist_min_size,
            lambda_mmd=self.lambda_mmd,
            alpha=self.detector_alpha,
            n_permutations=detector_n_permutations,
        ) if use_mmd else None

        energy = EnergyTest(
            self.l_hist_size,
            self.l_new_size,
            self.ks_hist_min_size,
            lambda_energy=self.lambda_energy,
            alpha=self.detector_alpha,
            n_permutations=detector_n_permutations,
        ) if use_energy else None

        _ = KLDivergence()

        label_file = os.path.join(self.experiment_dir, "stream_labels.csv")
        state_assign_file = os.path.join(self.experiment_dir, "window_states.csv")
        score_csv_file = os.path.join(self.experiment_dir, "drift_window_scores.csv")

        for out_path in (
            label_file,
            state_assign_file,
            score_csv_file,
            os.path.join(self.experiment_dir, "state_label_purity.csv"),
            os.path.join(self.experiment_dir, "label_state_fragmentation.csv"),
        ):
            if os.path.exists(out_path):
                os.remove(out_path)

        with open(state_assign_file, "w") as f:
            f.write(
                "window_idx,state_id,major_label,purity,sim_to_state,"
                "is_alarm,state_transition_reason,active_state_logic\n"
            )

        with open(score_csv_file, "w") as f:
            f.write(
                "window_idx,gt,pred,detector,detector_stat_name,"
                "detector_stat,p_value,alarm_score,score_used,"
                "alarm_score_threshold,p_value_threshold,raw_stat_threshold,critical_value,"
                "state_id,major_label,purity,sim_to_state,"
                "is_alarm,state_transition_reason,active_state_logic\n"
            )

        X = np.asarray(test_set, dtype=float)
        y = np.asarray(class_labels, dtype=int).reshape(-1)

       
        if drift_labels is None:
            drift_range = np.zeros((X.shape[0],), dtype=int)
            drift_cp = None
            gt_source_parse = "none"
        else:
            d = np.asarray(drift_labels, dtype=int)

            if d.ndim == 2 and d.shape[1] >= 2:
                if d.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"drift_labels length mismatch: drift_labels={d.shape[0]}, X={X.shape[0]}"
                    )
                drift_range = d[:, 0].reshape(-1).astype(int)
                drift_cp = d[:, 1].reshape(-1).astype(int)
                gt_source_parse = "two_column_range_and_cp"
            else:
                one_col = d.reshape(-1).astype(int)
                if one_col.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"drift_labels length mismatch: drift_labels={one_col.shape[0]}, X={X.shape[0]}"
                    )

                positive_ratio = float(np.mean(one_col == 1)) if one_col.size > 0 else 0.0
                if positive_ratio <= 0.20:
                    drift_cp = one_col
                    drift_range = one_col.copy()
                    gt_source_parse = "one_column_sparse_drift_cp"
                else:
                    drift_range = one_col
                    drift_cp = None
                    gt_source_parse = "one_column_drift_range"

        
        if (drift_cp is None or int(np.sum(drift_cp)) == 0) and y is not None:
            y_flat = np.asarray(y, dtype=int).reshape(-1)
            if y_flat.size > 1 and np.unique(y_flat).size > 1:
                label_cp = np.zeros_like(y_flat, dtype=int)
                label_cp[1:] = (y_flat[1:] != y_flat[:-1]).astype(int)
                if int(label_cp.sum()) > 0:
                    drift_cp = label_cp
                    # Use the recovered CP also as range placeholder for window slicing/debug.
                    drift_range = label_cp.copy()
                    gt_source_parse = f"{gt_source_parse}_label_change_recovery"
                    print(
                        "⚠️ Recovered drift_cp from class-label changes: "
                        f"{int(drift_cp.sum())} sample-level change points"
                    )

        seqs, dseqs = self._build_windows(X, y, drift_range)
        num_w = len(seqs)

        if drift_cp is not None and int(np.sum(drift_cp)) > 0:
            drift_for_gt = drift_cp
            gt_source = "drift_cp"
        else:
            drift_for_gt = drift_range
            gt_source = "drift_range_fallback_no_cp"

        cp_s_gt = self._change_points_from_drift_labels(drift_for_gt)
        cp_w_gt = self._cp_samples_to_cp_windows(
            cp_s_gt,
            win=self.window_size,
            step=self.steps,
            N=int(X.shape[0]),
        )
       
        gt_arr = self._build_window_gt_from_cp(
            cp_windows=cp_w_gt,
            num_w=num_w,
            tolerance_k=self.pointwise_tolerance_k,
            shift_by_lnew=False,
        )

        print("\n" + "=" * 80)
        print("[DRIFT GT DEBUG]")
        print("GT source:", gt_source)
        print("GT parser:", gt_source_parse)
        print("pointwise_tolerance_k:", self.pointwise_tolerance_k)
        print("event_tolerance_k:", self.event_tolerance_k)
        print("alarm_cooldown_k:", self.alarm_cooldown_k)
        print("X shape:", X.shape)
        print("num windows:", num_w)
        print("drift_labels is None:", drift_labels is None)
        if drift_labels is not None:
            print("drift_labels shape:", np.asarray(drift_labels).shape)
        print("drift_range positives:", int(np.sum(drift_range)), "/", len(drift_range))
        if drift_cp is not None:
            print("drift_cp positives:", int(np.sum(drift_cp)), "/", len(drift_cp))
        else:
            print("drift_cp: None")
        print("sample change-points:", len(cp_s_gt))
        print("first 50 cp_s_gt:", cp_s_gt[:50])
        print("window change-points:", len(cp_w_gt))
        print("first 50 cp_w_gt:", cp_w_gt[:50])
        print("window GT positives:", int(gt_arr.sum()), "/", len(gt_arr))
        print("unique gt_arr:", np.unique(gt_arr, return_counts=True))
        print("[END DRIFT GT DEBUG]")
        print("=" * 80 + "\n")

        debug_df = pd.DataFrame({
            "sample_idx": np.arange(len(drift_range)),
            "drift_range": drift_range,
            "drift_cp": drift_cp if drift_cp is not None else np.zeros_like(drift_range),
        })
        debug_path = os.path.join(self.experiment_dir, "drift_label_debug.csv")
        debug_df.to_csv(debug_path, index=False)
        print(f"✅ Drift label debug saved to: {debug_path}")

        active_state_logic = str(self.state_logic).lower().strip()
        if active_state_logic not in ("cosine", "kld"):
            print(f"⚠️ Unknown state_logic={self.state_logic}; falling back to cosine")
            active_state_logic = "cosine"

        log_lines = [
            f"Total windows: {num_w}\n",
            f"GT positives: {int(gt_arr.sum())} / {num_w}\n",
            f"GT source: {gt_source}\n",
            f"GT parser: {gt_source_parse}\n",
            f"pointwise_tolerance_k={self.pointwise_tolerance_k}, event_tolerance_k={self.event_tolerance_k}\n",
            f"alarm_cooldown_k={self.alarm_cooldown_k}\n",
            f"window_size={self.window_size}, steps={self.steps}, drift_mode={mode}\n",
            f"active_state_logic={active_state_logic}\n",
        ]
        if self.enable_rag:
            log_lines.append("RAG: compact query-filtered state cache by default; legacy full-cache KV is disabled in filtered mode\n")

        
        state_protos: Dict[int, np.ndarray] = {}
        state_counts: Dict[int, int] = {}
        uf_parent: Dict[int, int] = {}
        current_state = 0
        revisit_count = 0
        new_state_count = 0
        stay_current_count = 0

        
        state_Pi_kld: Dict[int, np.ndarray] = {}
        state_counts_kld: Dict[int, int] = {}
        uf_parent_kld: Dict[int, int] = {}
        current_state_kld = 0
        state_assignments_kld = []
        revisit_count_kld = 0
        new_state_count_kld = 0
        stay_current_count_kld = 0
        state_labels_all_kld: Dict[int, List[int]] = {}

       
        active_state_protos: Dict[int, np.ndarray] = {}
        active_state_counts: Dict[int, int] = {}
        active_uf_parent: Dict[int, int] = {}
        active_state_samples_all: Dict[int, np.ndarray] = {}
        active_state_labels_all: Dict[int, List[int]] = {}
        active_state_Pi_stad: Dict[int, np.ndarray] = {}

        state_buf = {}
        state_samples = {}
        all_gt, all_pred = [], []
        all_detector_stat, all_p_value, all_score_used, all_critical_value = [], [], [], []
        drift_alarm_count = 0
        raw_drift_alarm_count = 0
        suppressed_alarm_count = 0
        last_kept_alarm_w = -10**9
        self.state_assignments = []
        self.rag_answers = []

        for widx in tqdm(range(num_w), desc="✅ Drift-driven state management"):
            window = np.asarray(seqs[widx], dtype=float)
            Xw = window[:, :-1]
            yw = window[:, -1].astype(int)

            pd.Series(yw.ravel()).to_csv(label_file, index=False, header=False, mode="a")

            z_raw = Xw.mean(axis=0)
            z_for_state = self._l2_norm(z_raw) if self.normalize_window else z_raw.copy()
            z_for_det = self._l2_norm(z_raw) if self.normalize_for_detectors else z_raw.copy()

            if yw.size:
                counts = np.bincount(yw)
                major = int(np.argmax(counts))
                purity = float(counts[major] / max(1, yw.size))
            else:
                major = -1
                purity = 0.0

            gt = int(gt_arr[widx])
            is_drift_ks = is_drift_mmd = is_drift_energy = False

            detector_name = mode
            detector_stat_name = "unknown"
            detector_stat = 0.0
            detector_p_value = float("nan")
            alarm_score = 0.0
            alarm_score_threshold = float("nan")
            p_value_threshold = float("nan")
            raw_stat_threshold = float("nan")
            critical_value = float("nan")

            if ks is not None:
                detector_name = "ks"
                detector_stat_name = "ks_max_d"
                dflag, stats = ks.detect_drift(z_for_det)
                if isinstance(stats, (list, tuple)):
                    if len(stats) >= 1 and np.isfinite(stats[0]):
                        detector_stat = float(stats[0])
                    if len(stats) >= 2 and np.isfinite(stats[1]):
                        detector_p_value = float(np.clip(float(stats[1]), 0.0, 1.0))
                detector_stat = float(getattr(ks, "last_statistic", detector_stat))
                detector_p_value = float(getattr(ks, "last_p_value", detector_p_value))
                detector_p_value = float(np.clip(detector_p_value, 0.0, 1.0))
                is_drift_ks = bool(dflag)
                alarm_score = float(1.0 - detector_p_value) if np.isfinite(detector_p_value) else 0.0
                alarm_score_threshold = float(1.0 - self.detector_alpha)
                p_value_threshold = float(self.detector_alpha)

            elif ks_alibi is not None:
                detector_name = "ks_alibi"
                detector_stat_name = "ks_max_d"
                dflag, stats = ks_alibi.detect_drift(z_for_det)
                if isinstance(stats, (list, tuple)):
                    if len(stats) >= 1 and np.isfinite(stats[0]):
                        detector_stat = float(stats[0])
                    if len(stats) >= 2 and np.isfinite(stats[1]):
                        detector_p_value = float(np.clip(float(stats[1]), 0.0, 1.0))
                detector_stat = float(getattr(ks_alibi, "last_statistic", detector_stat))
                detector_p_value = float(getattr(ks_alibi, "last_p_value", detector_p_value))
                detector_p_value = float(np.clip(detector_p_value, 0.0, 1.0))
                is_drift_ks = bool(dflag)
                alarm_score = float(1.0 - detector_p_value) if np.isfinite(detector_p_value) else 0.0
                alarm_score_threshold = float(1.0 - self.detector_alpha)
                p_value_threshold = float(self.detector_alpha)

            elif mmd is not None:
                detector_name = "mmd"
                detector_stat_name = "mmd2"
                dflag, stats = mmd.detect_drift(z_for_det)
                if isinstance(stats, (list, tuple)):
                    if len(stats) >= 1 and np.isfinite(stats[0]):
                        detector_stat = float(stats[0])
                    if len(stats) >= 2 and np.isfinite(stats[1]):
                        detector_p_value = float(np.clip(float(stats[1]), 0.0, 1.0))
                    if len(stats) >= 3 and np.isfinite(stats[2]):
                        critical_value = float(stats[2])
                
                detector_stat = float(getattr(mmd, "last_statistic", detector_stat))
                detector_p_value = float(getattr(mmd, "last_p_value", detector_p_value))
                critical_value = float(getattr(mmd, "last_critical_value", critical_value))
                detector_p_value = float(np.clip(detector_p_value, 0.0, 1.0))
                is_drift_mmd = bool(dflag)
                raw_stat_threshold = float(self.lambda_mmd)
                if np.isfinite(detector_p_value):
                    alarm_score = float(1.0 - detector_p_value)
                    alarm_score_threshold = float(1.0 - self.detector_alpha)
                    p_value_threshold = float(self.detector_alpha)
                else:
                    alarm_score = float(detector_stat)
                    alarm_score_threshold = float(self.lambda_mmd)

            elif energy is not None:
                detector_name = "energy"
                detector_stat_name = "energy2"
                dflag, stats = energy.detect_drift(z_for_det)
                if isinstance(stats, (list, tuple)):
                    if len(stats) >= 1 and np.isfinite(stats[0]):
                        detector_stat = float(stats[0])
                    if len(stats) >= 2 and np.isfinite(stats[1]):
                        detector_p_value = float(np.clip(float(stats[1]), 0.0, 1.0))
                    if len(stats) >= 3 and np.isfinite(stats[2]):
                        critical_value = float(stats[2])
               
                detector_stat = float(getattr(energy, "last_statistic", detector_stat))
                detector_p_value = float(getattr(energy, "last_p_value", detector_p_value))
                critical_value = float(getattr(energy, "last_critical_value", critical_value))
                detector_p_value = float(np.clip(detector_p_value, 0.0, 1.0))
                is_drift_energy = bool(dflag)
                raw_stat_threshold = float(self.lambda_energy)
                if np.isfinite(detector_p_value):
                    alarm_score = float(1.0 - detector_p_value)
                    alarm_score_threshold = float(1.0 - self.detector_alpha)
                    p_value_threshold = float(self.detector_alpha)
                else:
                    alarm_score = float(detector_stat)
                    alarm_score_threshold = float(self.lambda_energy)

            if not np.isfinite(alarm_score):
                alarm_score = 0.0

           
            score_used = float(alarm_score)

            raw_alarm = bool(is_drift_ks or is_drift_mmd or is_drift_energy)
            if raw_alarm:
                raw_drift_alarm_count += 1

            suppressed_alarm = False
            if (
                raw_alarm
                and self.alarm_cooldown_k > 0
                and (widx - last_kept_alarm_w) <= self.alarm_cooldown_k
            ):
                
                alarm = False
                suppressed_alarm = True
                suppressed_alarm_count += 1
            else:
                alarm = raw_alarm
                if alarm:
                    last_kept_alarm_w = int(widx)

            pred = 1 if alarm else 0

            
            if alarm:
                assigned_state, sim_to_state, reason = self._handle_drift_state_transition(
                    z_for_state, state_protos, state_counts, uf_parent
                )
                if reason == "revisit_old":
                    revisit_count += 1
                elif reason in ("create_new", "force_create_new"):
                    new_state_count += 1
            else:
                assigned_state, sim_to_state, reason = self._handle_no_drift_state_assignment(
                    z_for_state, current_state, state_protos, state_counts, uf_parent
                )
                if reason == "stay_current":
                    stay_current_count += 1

            current_state = self._uf_find(uf_parent, assigned_state)
            self._update_state_proto(current_state, z_for_state, state_protos)
            state_counts[current_state] = int(state_counts.get(current_state, 0) + 1)

            
            if alarm:
                assigned_kld, score_kld, reason_kld = self._handle_drift_state_transition_kld(
                    Xw, state_Pi_kld, state_counts_kld, uf_parent_kld, kld_epsilon
                )
                if reason_kld == "revisit_old":
                    revisit_count_kld += 1
                elif reason_kld == "create_new":
                    new_state_count_kld += 1
            else:
                assigned_kld, score_kld, reason_kld = self._handle_no_drift_state_assignment_kld(
                    Xw, current_state_kld, state_Pi_kld, state_counts_kld, uf_parent_kld
                )
                if reason_kld == "stay_current":
                    stay_current_count_kld += 1

            current_state_kld = self._uf_find(uf_parent_kld, assigned_kld)
            state_counts_kld[current_state_kld] = int(state_counts_kld.get(current_state_kld, 0) + 1)
            state_assignments_kld.append(current_state_kld)
            state_labels_all_kld.setdefault(current_state_kld, [])
            if major >= 0:
                state_labels_all_kld[current_state_kld].append(major)

            
            if active_state_logic == "kld":
                active_state_id = int(current_state_kld)
                active_sim_to_state = float(score_kld)
                active_reason = f"kld_{reason_kld}"
            else:
                active_state_id = int(current_state)
                active_sim_to_state = float(sim_to_state)
                active_reason = reason

            if suppressed_alarm:
                active_reason = f"suppressed_duplicate_alarm__{active_reason}"

            self.state_assignments.append(active_state_id)

            if active_state_id not in active_state_protos:
                active_state_protos[active_state_id] = z_for_state.copy()
                active_state_counts[active_state_id] = 0
                active_uf_parent[active_state_id] = active_state_id

            active_state_counts[active_state_id] = int(active_state_counts.get(active_state_id, 0) + 1)

            if active_state_id not in active_state_Pi_stad:
                active_state_Pi_stad[active_state_id] = self._stad_density_estimation(Xw)

            self._append_samples(active_state_samples_all, active_state_id, z_for_state[None, :])
            active_state_labels_all.setdefault(active_state_id, [])
            if major >= 0:
                active_state_labels_all[active_state_id].append(major)

            
            if self.enable_rag:
                qt = self._get_query_text(widx, window)
                self.rag_answers.append({
                    "window_idx": widx,
                    "query": qt,
                    "answer": "",
                    "state_id": active_state_id,
                    "cache_used": active_reason in [
                        "revisit_old", "stay_current", "kld_revisit_old", "kld_stay_current"
                    ],
                    "transition_reason": active_reason,
                    "is_drift": alarm,
                    "major_label": major,
                })

            if alarm:
                drift_alarm_count += 1
                if active_state_id not in state_buf:
                    H = z_raw[None, :]
                    state_buf[active_state_id] = {
                        "P": self._softmax_hist(H),
                        "mu": H.mean(axis=0),
                        "Sigma": self._safe_cov(H),
                    }
                    state_samples[active_state_id] = H.copy()
                else:
                    self._append_samples(state_samples, active_state_id, z_raw[None, :])

            if active_state_logic != "kld" and widx > 0 and (widx % self.merge_every) == 0:
                self._merge_states(state_protos, state_counts, uf_parent, log_lines)
                current_state = self._uf_find(uf_parent, current_state)

            all_gt.append(int(gt))
            all_pred.append(int(pred))
            all_detector_stat.append(float(detector_stat))
            all_p_value.append(float(detector_p_value) if np.isfinite(detector_p_value) else np.nan)
            all_score_used.append(float(score_used))
            all_critical_value.append(float(critical_value) if np.isfinite(critical_value) else np.nan)

            with open(state_assign_file, "a") as f:
                f.write(
                    f"{widx},{active_state_id},{major},{purity:.6f},"
                    f"{active_sim_to_state},{int(alarm)},{active_reason},{active_state_logic}\n"
                )
            with open(score_csv_file, "a") as f:
                f.write(
                    f"{widx},{gt},{pred},{detector_name},{detector_stat_name},"
                    f"{detector_stat},{detector_p_value},{alarm_score},{score_used},"
                    f"{alarm_score_threshold},{p_value_threshold},{raw_stat_threshold},{critical_value},"
                    f"{active_state_id},{major},{purity:.6f},{active_sim_to_state},"
                    f"{int(alarm)},{active_reason},{active_state_logic}\n"
                )

        self.state_protos = active_state_protos

        kld_assign_file = os.path.join(self.experiment_dir, "window_states_kld.csv")
        with open(kld_assign_file, "w") as f:
            f.write("window_idx,state_id,major_label,state_transition_reason\n")
            for widx, sid_kld in enumerate(state_assignments_kld):
                lbl = (
                    state_labels_all_kld.get(sid_kld, [-1])[-1]
                    if state_labels_all_kld.get(sid_kld) else -1
                )
                f.write(f"{widx},{sid_kld},{lbl},kld_assigned\n")
        print(f" KLD state assignments → {kld_assign_file}")

        comp_file = os.path.join(self.experiment_dir, "state_method_comparison.json")
        comp = {
            "active_state_logic": active_state_logic,
            "cosine_similarity": {
                "num_states": len(state_protos),
                "revisits": revisit_count,
                "new_states": new_state_count,
                "stay_current": stay_current_count,
                "cache_hit_rate": round(revisit_count / max(1, drift_alarm_count), 4),
                "method": "cosine similarity on L2-normalised fixed prototype",
                "threshold": self.state_sim_threshold,
            },
            "kld_original_stad": {
                "num_states": len(state_Pi_kld),
                "revisits": revisit_count_kld,
                "new_states": new_state_count_kld,
                "stay_current": stay_current_count_kld,
                "cache_hit_rate": round(revisit_count_kld / max(1, drift_alarm_count), 4),
                "method": "symmetric KLD on Pi=softmax(Z).mean(axis=0)",
                "threshold": float(kld_epsilon),
            },
            "active_states_used_for_rag": len(active_state_protos),
            "total_drift_alarms": drift_alarm_count,
            "raw_drift_alarms_before_cooldown": raw_drift_alarm_count,
            "suppressed_duplicate_alarms": suppressed_alarm_count,
            "alarm_cooldown_k": self.alarm_cooldown_k,
            "total_windows": num_w,
        }
        with open(comp_file, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"✅ Method comparison → {comp_file}")

        print("\n" + "=" * 70)
        print("STATE ASSIGNMENT METHOD COMPARISON")
        print("=" * 70)
        print(f"Active state logic for RAG: {active_state_logic}")
        print(f"{'Metric':<30} {'Cosine Sim':>15} {'KLD (STAD)':>15}")
        print("-" * 60)
        print(f"{'Num states created':<30} {len(state_protos):>15} {len(state_Pi_kld):>15}")
        print(f"{'Revisits/cache hits':<30} {revisit_count:>15} {revisit_count_kld:>15}")
        print(f"{'New states created':<30} {new_state_count:>15} {new_state_count_kld:>15}")
        print(
            f"{'Cache hit rate':<30} "
            f"{revisit_count / max(1, drift_alarm_count):>15.3f} "
            f"{revisit_count_kld / max(1, drift_alarm_count):>15.3f}"
        )
        print(f"{'Active states for RAG':<30} {len(active_state_protos):>15}")
        print("=" * 70 + "\n")

        self._export_state_representations(
            state_protos=active_state_protos,
            state_samples_all=active_state_samples_all,
            state_labels_all=active_state_labels_all,
            state_counts=active_state_counts,
            uf_parent=active_uf_parent,
            state_Pi_stad=active_state_Pi_stad,
        )

        y_true = np.asarray(all_gt, dtype=int)
        y_pred = np.asarray(all_pred, dtype=int)
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc_score_used = self._safe_auc(y_true, np.asarray(all_score_used, dtype=float))
        pr_score_used = self._safe_pr_auc(y_true, np.asarray(all_score_used, dtype=float))

        n_pos = int(np.sum(y_true))
        n_neg = int(len(y_true) - n_pos)
        gt_warning = ""
        if n_neg == 0:
            gt_warning = (
                f"  ⚠️ GT is ALL POSITIVE ({n_pos}/{len(y_true)}). "
                f"ROC-AUC/PR-AUC undefined. Fix drift label construction.\n"
            )
        elif n_pos == 0:
            gt_warning = (
                "  ⚠️ GT is ALL NEGATIVE. No drift events found. Check drift label construction.\n"
            )
        elif n_pos / len(y_true) > 0.5:
            gt_warning = (
                f"  ⚠️ GT is heavily positive: {n_pos}/{len(y_true)} "
                f"({n_pos / len(y_true) * 100:.1f}%). Drift labels may be too dense.\n"
            )

        auc_str = f"{auc_score_used:.4f}" if auc_score_used is not None else "N/A (degenerate GT)"
        pr_str = f"{pr_score_used:.4f}" if pr_score_used is not None else "N/A (degenerate GT)"

        active_revisit = revisit_count_kld if active_state_logic == "kld" else revisit_count
        active_new = new_state_count_kld if active_state_logic == "kld" else new_state_count
        active_stay = stay_current_count_kld if active_state_logic == "kld" else stay_current_count

        log_lines += [
            "\n=== Drift metrics ===\n",
            f"GT positives: {n_pos} / {len(y_true)}  GT negatives: {n_neg} / {len(y_true)}\n",
            gt_warning if gt_warning else "",
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}\n",
            f"ROC-AUC={auc_str}  PR-AUC={pr_str}\n",
            f"alarms={drift_alarm_count}  active_states={len(active_state_protos)}\n",
            f"raw_alarms_before_cooldown={raw_drift_alarm_count}  suppressed_duplicate_alarms={suppressed_alarm_count}  alarm_cooldown_k={self.alarm_cooldown_k}\n",
            "\n--- Active state assignment used for RAG ---\n",
            f"logic={active_state_logic}\n",
            f"stay={active_stay}  revisit={active_revisit}  new={active_new}\n",
            f"cache_hit_rate={active_revisit / max(1, drift_alarm_count):.3f}\n",
            "\n--- Cosine Similarity state assignment ---\n",
            f"stay={stay_current_count}  revisit={revisit_count}  new={new_state_count}\n",
            f"cosine_states={len(state_protos)}  cosine_cache_hit_rate={revisit_count / max(1, drift_alarm_count):.3f}\n",
            "\n--- KLD (original STAD) state assignment ---\n",
            f"stay={stay_current_count_kld}  revisit={revisit_count_kld}  new={new_state_count_kld}\n",
            f"kld_states={len(state_Pi_kld)}  kld_cache_hit_rate={revisit_count_kld / max(1, drift_alarm_count):.3f}\n",
        ]

        try:
            drift_for_ev = drift_cp if drift_cp is not None and int(np.sum(drift_cp)) > 0 else drift_range
            cp_s = self._change_points_from_drift_labels(drift_for_ev)
            cp_w = self._cp_samples_to_cp_windows(
                cp_s, win=self.window_size, step=self.steps, N=int(X.shape[0])
            )
            ev = self._event_eval(
                alarm=np.asarray(all_pred, dtype=int),
                event_w=cp_w,
                k=self.event_tolerance_k,
                W=self.event_window_W,
                fp_penalty=self.event_fp_penalty,
            )
            log_lines += [
                "\n=== Event eval ===\n",
                f"events={ev['num_events']} alarms={ev['num_alarms']}\n",
                f"EP={ev['event_precision']}  ER={ev['event_recall']}  "
                f"EF1={ev['event_f1']}  NAB={ev['nab_score']}\n",
        ]
        except Exception as e:
            log_lines.append(f"[WARN] Event eval failed: {e}\n")

        try:
            df = pd.read_csv(score_csv_file)
            vms = float(v_measure_score(
                df["major_label"].to_numpy().astype(int),
                df["state_id"].to_numpy().astype(int),
            ))
            sr = float(len(set(df["state_id"].tolist())) / max(1, len(df)))
            ts = float(vms - self.state_penalty_lambda * sr)
            log_lines.append(f"\nV-measure={vms:.4f}  trade_score={ts:.4f}\n")
        except Exception as e:
            log_lines.append(f"[WARN] Clustering metrics failed: {e}\n")

        with open(os.path.join(self.experiment_dir, "log"), "w") as f:
            f.writelines(log_lines)

        print("\n[RAG DEBUG BEFORE SAVE]")
        print("enable_rag:", self.enable_rag)
        print("answer_gen is None:", self.answer_gen is None)
        print("comparator is None:", self.comparator is None)
        print("retriever is None:", self.retriever is None)
        print("knowledge_cache is None:", self.knowledge_cache is None)
        print("generator is None:", self.generator is None)
        print("num state assignments:", len(getattr(self, "state_assignments", [])))
        print("unique active states:", len(set(getattr(self, "state_assignments", []))))
        print("active_state_logic:", active_state_logic)
        print("original_texts:", None if self.original_texts is None else len(self.original_texts))
        print(
            "gold answers:",
            len(getattr(self, "ground_truth_answers", []))
            if hasattr(self, "ground_truth_answers") else 0,
        )
        print("[END RAG DEBUG]\n")

        if self.enable_rag:
            self._save_rag_metrics()

        return "✅ Simplified drift-driven state management complete.\n"
