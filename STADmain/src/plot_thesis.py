import argparse
import json
import os
import re
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_fscore_support,
        v_measure_score,
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


PALETTE = ["#2D6A9F", "#E84855", "#3BB273", "#F4A261", "#9B5DE5",
           "#F15BB5", "#00BBF9", "#FEE440", "#00F5D4", "#FB5607"]
GREY    = "#AAAAAA"
BG      = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


def load_scores(exp_dir):
    path = os.path.join(exp_dir, "drift_window_scores.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ Loaded drift_window_scores.csv  ({len(df)} rows)")
        return df
    print("  ⚠️  drift_window_scores.csv not found — using synthetic data")
    np.random.seed(42)
    n = 400
    cps = [100, 200, 300]
    labels = np.zeros(n, dtype=int)
    for i, cp in enumerate(cps):
        labels[cp:] = i + 1
    score = np.random.beta(2, 8, n)
    for cp in cps:
        score[cp:cp+8] = np.random.beta(8, 2, 8)
    gt = np.zeros(n, dtype=int)
    for cp in cps:
        gt[cp:cp+6] = 1
    pred = (score > 0.65).astype(int)
    state_ids = labels.copy()
    purity = np.random.uniform(0.7, 1.0, n)
    return pd.DataFrame({
        "window_idx": np.arange(n),
        "gt": gt, "pred": pred,
        "detector": "synthetic",
        "detector_stat_name": "synthetic_score",
        "detector_stat": score,
        "p_value": 1.0 - score,
        "alarm_score": score,
        "score_used": score,
        "alarm_score_threshold": 0.65,
        "p_value_threshold": 0.35,
        "critical_value": np.nan,
        "state_id": state_ids,
        "major_label": labels,
        "purity": purity,
        "is_alarm": pred,
    })


def load_states(exp_dir):
    path = os.path.join(exp_dir, "window_states.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ Loaded window_states.csv  ({len(df)} rows)")
        return df
    print("  ⚠️  window_states.csv not found — using synthetic data")
    np.random.seed(42)
    n = 400
    cps = [100, 200, 300]
    labels = np.zeros(n, dtype=int)
    for i, cp in enumerate(cps):
        labels[cp:] = i + 1
    states = labels.copy()
    states[300:] = 0
    reasons = ["stay_current"] * n
    for cp in cps:
        reasons[cp] = "create_new"
    reasons[300] = "revisit_old"
    return pd.DataFrame({
        "window_idx": np.arange(n),
        "state_id": states,
        "major_label": labels,
        "purity": np.random.uniform(0.7, 1.0, n),
        "sim_to_state": np.random.uniform(0.8, 1.0, n),
        "is_alarm": [1 if r != "stay_current" else 0 for r in reasons],
        "state_transition_reason": reasons,
    })


def load_holdout(exp_dir):
    path = os.path.join(exp_dir, "holdout_3method_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ Loaded holdout_3method_comparison.csv  ({len(df)} rows)")
        return df
    print("  ⚠️  holdout_3method_comparison.csv not found — using synthetic data")
    np.random.seed(42)
    n = 120
    states = np.repeat([0, 1, 2, 3], n // 4)
    answers = {
        "baseline_answer":   [f"Generic answer {i}" for i in range(n)],
        "cached_rag_answer": [f"Cached answer about topic {s}" for i, s in enumerate(states)],
        "fresh_rag_answer":  [f"Fresh answer about topic {s} with detail" for i, s in enumerate(states)],
    }
    return pd.DataFrame({"window_idx": np.arange(n), "state_id": states, **answers})


def load_state_repr(exp_dir):
    path = os.path.join(exp_dir, "state_representations.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"   Loaded state_representations.json  ({len(data['states'])} states)")
        return data
    print("    state_representations.json not found — using synthetic data")
    np.random.seed(42)
    states = {}
    for i in range(4):
        mu = np.random.randn(384) * 0.1
        mu[i * 96:(i + 1) * 96] += 2.0
        pi = np.zeros(384)
        pi[i * 96:(i + 1) * 96] = np.random.dirichlet(np.ones(96))
        states[str(i)] = {
            "state_id": i, "Ni": np.random.randint(30, 100),
            "mu_i": mu.tolist(), "Pi_stad": pi.tolist(),
            "Pi_labels": {str(i): 0.85, str((i+1)%4): 0.15},
        }
    kld = {}
    for i in range(4):
        kld[str(i)] = {}
        for j in range(4):
            kld[str(i)][str(j)] = 0.0 if i == j else round(np.random.uniform(1.5, 4.0), 4)
    return {"states": states, "pairwise_kld": kld}


def compute_rouge(predictions, references, metric="rougeL"):
    """Compute ROUGE-L scores. Falls back to simple word overlap if rouge_score unavailable."""
    if HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref or pd.isna(pred) or pd.isna(ref):
                scores.append(0.0)
                continue
            s = scorer.score(str(ref), str(pred))
            scores.append(s[metric].fmeasure)
        return scores
    else:
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref or pd.isna(pred) or pd.isna(ref):
                scores.append(0.0)
                continue
            pred_words = set(str(pred).lower().split())
            ref_words  = set(str(ref).lower().split())
            if not ref_words:
                scores.append(0.0)
                continue
            overlap = len(pred_words & ref_words) / len(ref_words)
            scores.append(overlap)
        return scores





DATASET_ORDER = [
    "AGNewsStreamSim",
    "CLINCStreamSim",
    "GaliboStreamSim",
    "KILTStreamSim",
    "MultiHopRAGStreamSim",
    "PopQAStreamSim",
    "RAGBenchStreamSim",
]

DETECTOR_ORDER = ["energy", "mmd", "ks_alibi"]

DATASET_LABEL = {
    "AGNewsStreamSim": "AG News",
    "CLINCStreamSim": "CLINC",
    "GaliboStreamSim": "Galibo",
    "KILTStreamSim": "KILT",
    "MultiHopRAGStreamSim": "MultiHopRAG",
    "PopQAStreamSim": "PopQA",
    "RAGBenchStreamSim": "RAGBench",
}

DETECTOR_LABEL = {
    "energy": "Energy",
    "mmd": "MMD",
    "ks_alibi": "KS-Alibi",
}


def _safe_auc(y, s):
    if not HAS_METRICS:
        return np.nan
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]
    s = s[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return np.nan


def _safe_pr_auc(y, s):
    if not HAS_METRICS:
        return np.nan
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]
    s = s[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return np.nan


def _safe_prf(y, pred):
    if not HAS_METRICS:
        return np.nan, np.nan, np.nan
    try:
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        return float(p), float(r), float(f1)
    except Exception:
        return np.nan, np.nan, np.nan


def _number_after(pattern, text):
    """Robustly extract first number after a label pattern."""
    m = re.search(pattern, text)
    if not m:
        return np.nan
    frag = text[m.end():m.end() + 120]
    n = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", frag)
    if not n:
        return np.nan
    try:
        return float(n.group(0))
    except Exception:
        return np.nan


def _int_after(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return 0
    try:
        return int(float(m.group(1)))
    except Exception:
        return 0


def _json_safe_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_json_safe_value(x) for x in v]
    return v


def _get_score_col(df):
    for c in ["score_used", "alarm_score", "ks_p_score", "mmd2", "energy2", "detector_stat"]:
        if c in df.columns:
            return c
    return None


def _get_pred_col(df):
    if "pred" in df.columns:
        return "pred"
    if "is_alarm" in df.columns:
        return "is_alarm"
    return None


def _get_window_idx(df):
    if "window_idx" in df.columns:
        return pd.to_numeric(df["window_idx"], errors="coerce").fillna(0).astype(int).values
    return np.arange(len(df))


def _savefig_dual(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=220, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close()


def parse_current_run_log(exp_dir):
    log_path = os.path.join(exp_dir, "log")
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    gt_m = re.search(r"GT positives:\s*(\d+)\s*/\s*(\d+)", text)
    event_m = re.search(r"events=(\d+)\s+alarms=(\d+)", text)

    return {
        "log_total_windows": int(gt_m.group(2)) if gt_m else 0,
        "log_gt_positive_windows": int(gt_m.group(1)) if gt_m else 0,
        "event_count": int(event_m.group(1)) if event_m else 0,
        "log_alarm_count": int(event_m.group(2)) if event_m else _int_after(r"alarms=(\d+)", text),
        "event_precision": _number_after(r"EP\s*=", text),
        "event_recall": _number_after(r"ER\s*=", text),
        "event_f1": _number_after(r"EF1\s*=", text),
        "nab": _number_after(r"NAB\s*=", text),
        "active_states": _int_after(r"active_states=(\d+)", text),
        "raw_alarms_before_cooldown": _int_after(r"raw_alarms_before_cooldown=(\d+)", text),
        "suppressed_duplicate_alarms": _int_after(r"suppressed_duplicate_alarms=(\d+)", text),
        "alarm_cooldown_k": _int_after(r"alarm_cooldown_k=(\d+)", text),
        "cache_hit_rate": _number_after(r"cache_hit_rate\s*=", text),
        "v_measure": _number_after(r"V-measure\s*=", text),
        "trade_score": _number_after(r"trade_score\s*=", text),
        "pointwise_tolerance_k": _int_after(r"pointwise_tolerance_k=(\d+)", text),
        "event_tolerance_k": _int_after(r"event_tolerance_k=(\d+)", text),
        "window_size_log": _int_after(r"window_size=(\d+)", text),
        "steps_log": _int_after(r"steps=(\d+)", text),
    }


def _v_measure_and_trade(df, state_col="state_id", label_col="major_label", penalty_lambda=1.0):
    if not HAS_METRICS:
        return np.nan, np.nan, np.nan
    if df is None or state_col not in df.columns or label_col not in df.columns:
        return np.nan, np.nan, np.nan
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
    states = pd.to_numeric(df[state_col], errors="coerce").fillna(-1).astype(int)
    try:
        n_states = int(states.nunique())
        v = float(v_measure_score(labels, states))
        trade = float(v - penalty_lambda * (n_states / max(1, len(df))))
        return v, trade, n_states
    except Exception:
        return np.nan, np.nan, np.nan


def load_kld_states_if_present(exp_dir):
    path = os.path.join(exp_dir, "window_states_kld.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"  ✅ Loaded window_states_kld.csv  ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"  ⚠️  Could not load window_states_kld.csv: {e}")
    return None


def compute_state_method_metrics(exp_dir, df_scores, df_states):
   
    rows = []

    active_source = df_scores if "state_id" in df_scores.columns and "major_label" in df_scores.columns else df_states
    v, trade, n_states = _v_measure_and_trade(active_source)
    rows.append({
        "method": "Active/Cosine",
        "v_measure": v,
        "trade_score": trade,
        "num_states": n_states,
    })

    df_kld = load_kld_states_if_present(exp_dir)
    if df_kld is not None:
        v, trade, n_states = _v_measure_and_trade(df_kld)
        rows.append({
            "method": "KLD",
            "v_measure": v,
            "trade_score": trade,
            "num_states": n_states,
        })

    return pd.DataFrame(rows)


def compute_shifted_auc_table(df_scores, max_shift=10):
    score_col = _get_score_col(df_scores)
    if score_col is None or "gt" not in df_scores.columns:
        return pd.DataFrame()

    y = pd.to_numeric(df_scores["gt"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df_scores[score_col], errors="coerce").fillna(0.0).astype(float)

    rows = []
    for shift in range(0, max_shift + 1):
        y_shift = y.shift(shift, fill_value=0)
        rows.append({
            "shift": shift,
            "shifted_gt_positive_windows": int(y_shift.sum()),
            "shifted_roc_auc": _safe_auc(y_shift, s),
            "shifted_roc_auc_inverted": _safe_auc(y_shift, 1.0 - s),
            "shifted_pr_auc": _safe_pr_auc(y_shift, s),
        })
    return pd.DataFrame(rows)


def save_run_metrics_summary(exp_dir, df_scores, df_states, max_shift=10):
    
    out = {}

    score_col = _get_score_col(df_scores)
    pred_col = _get_pred_col(df_scores)

    if score_col is not None and "gt" in df_scores.columns:
        y = pd.to_numeric(df_scores["gt"], errors="coerce").fillna(0).astype(int)
        s = pd.to_numeric(df_scores[score_col], errors="coerce").fillna(0.0).astype(float)
        if pred_col is not None:
            pred = pd.to_numeric(df_scores[pred_col], errors="coerce").fillna(0).astype(int)
        else:
            pred = pd.Series(np.zeros(len(df_scores), dtype=int))

        p, r, f1 = _safe_prf(y, pred)
        out.update({
            "score_col": score_col,
            "pred_col": pred_col,
            "n_windows": int(len(df_scores)),
            "gt_positive_windows": int(y.sum()),
            "pred_positive_windows": int(pred.sum()),
            "alarm_ratio": float(pred.mean()) if len(pred) else np.nan,
            "window_precision": p,
            "window_recall": r,
            "window_f1": f1,
            "roc_auc": _safe_auc(y, s),
            "roc_auc_inverted": _safe_auc(y, 1.0 - s),
            "pr_auc": _safe_pr_auc(y, s),
        })

    out.update(parse_current_run_log(exp_dir))

    shifted = compute_shifted_auc_table(df_scores, max_shift=max_shift)
    if len(shifted):
        shifted.to_csv(os.path.join(exp_dir, "shifted_auc_by_shift.csv"), index=False)
        if shifted["shifted_roc_auc"].notna().any():
            best_idx = shifted["shifted_roc_auc"].idxmax()
            out["best_shift"] = int(shifted.loc[best_idx, "shift"])
            out["best_shifted_roc_auc"] = float(shifted.loc[best_idx, "shifted_roc_auc"])
            out["best_shifted_pr_auc"] = float(shifted.loc[best_idx, "shifted_pr_auc"])

    state_methods = compute_state_method_metrics(exp_dir, df_scores, df_states)
    if len(state_methods):
        state_methods.to_csv(os.path.join(exp_dir, "state_method_metrics.csv"), index=False)

    json_safe = {k: _json_safe_value(v) for k, v in out.items()}
    with open(os.path.join(exp_dir, "run_metrics_summary.json"), "w") as f:
        json.dump(json_safe, f, indent=2)
    with open(os.path.join(exp_dir, "run_metrics_summary.txt"), "w") as f:
        f.write("Run metrics summary\n")
        f.write("=" * 60 + "\n")
        for k, v in json_safe.items():
            f.write(f"{k}: {v}\n")

    print("   Saved run_metrics_summary.json")
    print("   Saved shifted_auc_by_shift.csv")
    print("   Saved state_method_metrics.csv")
    return out


def plot6_shifted_auc(df_scores, out_path, max_shift=10):
    shifted = compute_shifted_auc_table(df_scores, max_shift=max_shift)
    if shifted.empty:
        print("  Could not compute shifted AUROC — skipping plot 6")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(shifted["shift"], shifted["shifted_roc_auc"], marker="o", linewidth=2,
            label="AUROC", color=PALETTE[0])
    ax.plot(shifted["shift"], shifted["shifted_pr_auc"], marker="s", linewidth=2,
            label="PR-AUC", color=PALETTE[2])

    if shifted["shifted_roc_auc"].notna().any():
        best_idx = shifted["shifted_roc_auc"].idxmax()
        best_shift = int(shifted.loc[best_idx, "shift"])
        best_auc = float(shifted.loc[best_idx, "shifted_roc_auc"])
        ax.scatter([best_shift], [best_auc], s=120, color=PALETTE[1], zorder=5)
        ax.annotate(
            f"Best shift={best_shift}\nAUROC={best_auc:.3f}",
            xy=(best_shift, best_auc),
            xytext=(best_shift + 0.4, min(1.0, best_auc + 0.08)),
            arrowprops=dict(arrowstyle="->", color=PALETTE[1]),
            fontsize=10,
            color=PALETTE[1],
        )

    ax.set_xlabel("Ground-truth shift in windows")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Plot 6 — Shifted AUROC / PR-AUC Diagnostic")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 6 saved: {out_path}")


def plot7_score_distribution(df_scores, out_path):
    score_col = _get_score_col(df_scores)
    if score_col is None or "gt" not in df_scores.columns:
        print("  ⚠️  Cannot plot score distribution — skipping plot 7")
        return

    y = pd.to_numeric(df_scores["gt"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df_scores[score_col], errors="coerce").fillna(0.0).astype(float)
    vals0 = s[y == 0].values
    vals1 = s[y == 1].values

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot([vals0, vals1], labels=["GT=0", "GT=1"], showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], [PALETTE[0], PALETTE[2]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    rng = np.random.default_rng(42)
    for i, vals in enumerate([vals0, vals1], start=1):
        if len(vals) == 0:
            continue
        idx = rng.choice(len(vals), size=min(700, len(vals)), replace=False)
        jitter = rng.uniform(-0.08, 0.08, size=len(idx))
        ax.scatter(np.full(len(idx), i) + jitter, vals[idx], s=10, alpha=0.35, color=PALETTE[i])

    ax.set_ylabel(score_col)
    ax.set_title("Plot 7 — Detector Score Distribution by Ground Truth")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 7 saved: {out_path}")


def plot8_state_method_comparison(exp_dir, df_scores, df_states, out_path):
    df_methods = compute_state_method_metrics(exp_dir, df_scores, df_states)
    if df_methods.empty:
        print("  ⚠️  No state-method metrics available — skipping plot 8")
        return

    x = np.arange(len(df_methods))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"wspace": 0.35})
    ax_score, ax_count = axes

    ax_score.bar(x - width / 2, df_methods["v_measure"], width,
                 label="V-measure", color=PALETTE[0])
    ax_score.bar(x + width / 2, df_methods["trade_score"], width,
                 label="Trade score", color=PALETTE[3])
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(df_methods["method"])
    ax_score.set_ylim(0, 1.05)
    ax_score.set_ylabel("Score")
    ax_score.set_title("State Quality")
    ax_score.legend()
    ax_score.grid(axis="y", alpha=0.3)

    ax_count.bar(x, df_methods["num_states"], color=PALETTE[2], width=0.5)
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(df_methods["method"])
    ax_count.set_ylabel("Number of states")
    ax_count.set_title("State Count")
    ax_count.grid(axis="y", alpha=0.3)

    fig.suptitle("Plot 8 — State Assignment Comparison: Active/Cosine vs KLD", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 8 saved: {out_path}")


def infer_dataset_from_path(path):
    s = str(path)
    for ds in DATASET_ORDER:
        if ds in s:
            return ds
    low = s.lower()
    if "agnews" in low:
        return "AGNewsStreamSim"
    if "clinc" in low:
        return "CLINCStreamSim"
    if "galibo" in low:
        return "GaliboStreamSim"
    if "kilt" in low:
        return "KILTStreamSim"
    if "multihop" in low or "multihot" in low:
        return "MultiHopRAGStreamSim"
    if "popqa" in low:
        return "PopQAStreamSim"
    if "ragbench" in low:
        return "RAGBenchStreamSim"
    return None


def infer_detector_from_path(path):
    parts = [p.lower() for p in Path(path).parts]
    for d in DETECTOR_ORDER:
        if d in parts:
            return d
    s = str(path).lower()
    if "modeenergy" in s:
        return "energy"
    if "modemmd" in s:
        return "mmd"
    if "ks_alibi" in s or "modeks_alibi" in s:
        return "ks_alibi"
    return None


def find_output_root(exp_dir):
    p = Path(exp_dir).resolve()
    for q in [p] + list(p.parents):
        if q.name == "output":
            return q
    return None


def read_one_run_for_global(csv_path, max_shift=10):
    csv_path = Path(csv_path)
    exp_dir = csv_path.parent
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, []
    if "gt" not in df.columns:
        return None, []
    score_col = _get_score_col(df)
    if score_col is None:
        return None, []

    dataset = infer_dataset_from_path(csv_path)
    detector = infer_detector_from_path(csv_path)
    if dataset is None or detector is None:
        return None, []

    y = pd.to_numeric(df["gt"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)
    pred_col = _get_pred_col(df)
    pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int) if pred_col else pd.Series(np.zeros(len(df), dtype=int))
    p, r, f1 = _safe_prf(y, pred)
    log = parse_current_run_log(str(exp_dir))

    window_states_path = exp_dir / "window_states.csv"
    if window_states_path.exists():
        try:
            df_states_local = pd.read_csv(window_states_path)
        except Exception:
            df_states_local = df
    else:
        df_states_local = df

    state_methods = compute_state_method_metrics(str(exp_dir), df, df_states_local)
    cosine_v = cosine_trade = cosine_states = np.nan
    kld_v = kld_trade = kld_states = np.nan
    if len(state_methods):
        for _, rr in state_methods.iterrows():
            method = str(rr["method"]).lower()
            if "kld" in method:
                kld_v = rr["v_measure"]
                kld_trade = rr["trade_score"]
                kld_states = rr["num_states"]
            else:
                cosine_v = rr["v_measure"]
                cosine_trade = rr["trade_score"]
                cosine_states = rr["num_states"]

    shifted = compute_shifted_auc_table(df, max_shift=max_shift)
    best_shift = np.nan
    best_shifted_auc = np.nan
    if len(shifted) and shifted["shifted_roc_auc"].notna().any():
        bi = shifted["shifted_roc_auc"].idxmax()
        best_shift = int(shifted.loc[bi, "shift"])
        best_shifted_auc = float(shifted.loc[bi, "shifted_roc_auc"])

    row = {
        "dataset": dataset,
        "dataset_label": DATASET_LABEL.get(dataset, dataset),
        "detector": detector,
        "detector_label": DETECTOR_LABEL.get(detector, detector),
        "csv_path": str(csv_path),
        "run_dir": str(exp_dir),
        "score_col": score_col,
        "n_windows": len(df),
        "gt_positive_windows": int(y.sum()),
        "pred_positive_windows": int(pred.sum()),
        "alarm_ratio": float(pred.mean()) if len(pred) else np.nan,
        "window_precision": p,
        "window_recall": r,
        "window_f1": f1,
        "roc_auc": _safe_auc(y, s),
        "pr_auc": _safe_pr_auc(y, s),
        "best_shift": best_shift,
        "best_shifted_roc_auc": best_shifted_auc,
        "cosine_v_measure": cosine_v,
        "cosine_trade_score": cosine_trade,
        "cosine_num_states": cosine_states,
        "kld_v_measure": kld_v,
        "kld_trade_score": kld_trade,
        "kld_num_states": kld_states,
    }
    row.update(log)

    shift_rows = []
    if len(shifted):
        for _, sr in shifted.iterrows():
            shift_rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABEL.get(dataset, dataset),
                "detector": detector,
                "detector_label": DETECTOR_LABEL.get(detector, detector),
                "csv_path": str(csv_path),
                "shift": sr["shift"],
                "shifted_roc_auc": sr["shifted_roc_auc"],
                "shifted_pr_auc": sr["shifted_pr_auc"],
            })
    return row, shift_rows


def select_best_global_runs(df):
    tmp = df.copy()
    for c in ["event_f1", "nab", "alarm_ratio", "cosine_v_measure"]:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp["saturation_penalty"] = np.where(tmp["alarm_ratio"].fillna(0) > 0.60, 1.00, 0.0)
    tmp["negative_nab_penalty"] = np.where(tmp["nab"].fillna(0) < 0, 0.75, 0.0)
    tmp["selection_score"] = (
        0.55 * tmp["event_f1"].fillna(-1)
        + 0.45 * tmp["nab"].fillna(-1)
        + 0.05 * tmp["cosine_v_measure"].fillna(0)
        - tmp["saturation_penalty"]
        - tmp["negative_nab_penalty"]
    )
    tmp["dataset"] = pd.Categorical(tmp["dataset"], categories=DATASET_ORDER, ordered=True)
    return (
        tmp.sort_values(["dataset", "selection_score", "event_f1", "nab"], ascending=[True, False, False, False])
        .groupby("dataset", observed=True)
        .head(1)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def select_best_global_runs_per_detector(df):
    tmp = df.copy()
    for c in ["event_f1", "nab", "alarm_ratio"]:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp["saturation_penalty"] = np.where(tmp["alarm_ratio"].fillna(0) > 0.60, 1.00, 0.0)
    tmp["negative_nab_penalty"] = np.where(tmp["nab"].fillna(0) < 0, 0.75, 0.0)
    tmp["selection_score"] = (
        0.55 * tmp["event_f1"].fillna(-1)
        + 0.45 * tmp["nab"].fillna(-1)
        - tmp["saturation_penalty"]
        - tmp["negative_nab_penalty"]
    )
    tmp["dataset"] = pd.Categorical(tmp["dataset"], categories=DATASET_ORDER, ordered=True)
    tmp["detector"] = pd.Categorical(tmp["detector"], categories=DETECTOR_ORDER, ordered=True)
    return (
        tmp.sort_values(["dataset", "detector", "selection_score", "event_f1", "nab"], ascending=[True, True, False, False, False])
        .groupby(["dataset", "detector"], observed=True)
        .head(1)
        .sort_values(["dataset", "detector"])
        .reset_index(drop=True)
    )


def update_global_summary(exp_dir, max_shift=10):
   
    output_root = find_output_root(exp_dir)
    if output_root is None:
        print("  ⚠️  Could not infer output root — skipping global summary")
        return

    out_dir = output_root / "_analysis_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(output_root.rglob("drift_window_scores.csv"))
    csv_paths = [p for p in csv_paths if p.parent.name == "online_learning"]

    rows, shift_rows = [], []
    for p in csv_paths:
        row, sr = read_one_run_for_global(p, max_shift=max_shift)
        if row is not None:
            rows.append(row)
            shift_rows.extend(sr)
    if not rows:
        print("  ⚠️  No valid runs found for global summary")
        return

    df = pd.DataFrame(rows)
    shifts = pd.DataFrame(shift_rows)
    selected = select_best_global_runs(df)
    best_by_detector = select_best_global_runs_per_detector(df)

    df.to_csv(out_dir / "all_runs_metrics.csv", index=False)
    shifts.to_csv(out_dir / "all_runs_shifted_auc_by_shift.csv", index=False)
    selected.to_csv(out_dir / "selected_best_run_per_dataset.csv", index=False)
    best_by_detector.to_csv(out_dir / "best_run_per_dataset_detector.csv", index=False)

    x = np.arange(len(selected))
    width = 0.38

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, selected["event_f1"], width, label="Event F1", color=PALETTE[0])
    plt.bar(x + width / 2, selected["nab"], width, label="NAB", color=PALETTE[3])
    plt.axhline(0, linewidth=0.8, color="#333")
    plt.xticks(x, selected["dataset_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Selected event-level drift results")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig_dual(out_dir, "event_f1_nab_selected")

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, selected["event_precision"], width, label="Event precision", color=PALETTE[0])
    plt.bar(x + width / 2, selected["event_recall"], width, label="Event recall", color=PALETTE[3])
    plt.xticks(x, selected["dataset_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Selected event precision and recall")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig_dual(out_dir, "event_precision_recall_selected")

   
    pivot = best_by_detector.pivot_table(
        index="dataset_label", columns="detector_label", values="event_f1",
        aggfunc="max", observed=True
    )
    row_order = [DATASET_LABEL[x] for x in DATASET_ORDER if DATASET_LABEL[x] in pivot.index]
    col_order = [DETECTOR_LABEL[x] for x in DETECTOR_ORDER if DETECTOR_LABEL[x] in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    x2 = np.arange(len(pivot))
    width2 = 0.24
    plt.figure(figsize=(10.5, 5))
    for i, col in enumerate(pivot.columns):
        plt.bar(x2 + (i - 1) * width2, pivot[col], width2, label=col)
    plt.xticks(x2, pivot.index, rotation=30, ha="right")
    plt.ylabel("Event F1")
    plt.title("Best event F1 by dataset and detector")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig_dual(out_dir, "detector_comparison_event_f1")


    plt.figure(figsize=(12, 5.2))
    width3 = 0.20
    plt.bar(x - 1.5 * width3, selected["cosine_v_measure"], width3, label="Cosine V-measure")
    plt.bar(x - 0.5 * width3, selected["cosine_trade_score"], width3, label="Cosine trade score")
    plt.bar(x + 0.5 * width3, selected["kld_v_measure"], width3, label="KLD V-measure")
    plt.bar(x + 1.5 * width3, selected["kld_trade_score"], width3, label="KLD trade score")
    plt.xticks(x, selected["dataset_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("State quality comparison: cosine vs KLD")
    plt.ylim(0, 1.05)
    plt.legend(ncol=2)
    plt.grid(axis="y", alpha=0.3)
    _savefig_dual(out_dir, "state_quality_cosine_vs_kld_selected")

    plt.figure(figsize=(12, 5.2))
    width4 = 0.35
    plt.bar(x - width4 / 2, selected["cosine_num_states"], width4, label="Cosine states")
    plt.bar(x + width4 / 2, selected["kld_num_states"], width4, label="KLD states")
    plt.xticks(x, selected["dataset_label"], rotation=30, ha="right")
    plt.ylabel("Number of states")
    plt.title("State count comparison: cosine vs KLD")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig_dual(out_dir, "state_count_cosine_vs_kld_selected")

    print(f"  ✅ Global summary refreshed: {out_dir}")


def plot1_drift_score(df_scores, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax_score, ax_gt = axes

    windows = df_scores["window_idx"].values

  
    if "alarm_score" in df_scores.columns:
        score_col = "alarm_score"
    elif "score_used" in df_scores.columns:
        score_col = "score_used"
    elif "ks_p_score" in df_scores.columns:
        score_col = "ks_p_score"
    else:
        raise ValueError("No alarm-score column found in drift_window_scores.csv")

    scores_raw = pd.to_numeric(df_scores[score_col], errors="coerce").fillna(0.0).values

    k = min(5, len(scores_raw))
    scores_smooth = np.convolve(scores_raw, np.ones(k)/k, mode="same") if k > 1 else scores_raw

    detector_label = str(df_scores["detector"].iloc[0]) if "detector" in df_scores.columns and len(df_scores) else "detector"

    # Raw alarm score is the decision score; smoothed curve is only a visual guide.
    ax_score.plot(
        windows, scores_raw, color=PALETTE[0], lw=0.8, alpha=0.35,
        label=f"{detector_label} alarm score (raw decision score)"
    )
    ax_score.fill_between(windows, scores_smooth, alpha=0.14, color=PALETTE[0])
    ax_score.plot(
        windows, scores_smooth, color=PALETTE[0], lw=1.5,
        label="Smoothed visual guide"
    )

    if "alarm_score_threshold" in df_scores.columns:
        thr = pd.to_numeric(df_scores["alarm_score_threshold"], errors="coerce").dropna()
        if len(thr) > 0:
            ax_score.axhline(float(thr.iloc[0]), color=PALETTE[3], lw=1.2, ls=":",
                             alpha=0.9, label="Alarm threshold")

    alarm_mask = df_scores["is_alarm"].astype(bool).values if "is_alarm" in df_scores.columns \
        else df_scores["pred"].astype(bool).values
    ax_score.scatter(windows[alarm_mask], scores_raw[alarm_mask],
                     color=PALETTE[1], s=28, zorder=5, label="Alarm fired", alpha=0.9)

    gt = df_scores["gt"].values if "gt" in df_scores.columns else np.zeros(len(windows))
    cp_windows = windows[np.where(np.diff(gt.astype(int), prepend=0) == 1)[0]]
    for cp in cp_windows:
        ax_score.axvspan(cp - 2, cp + 6, color=PALETTE[2], alpha=0.18, zorder=0)
        ax_score.axvline(cp, color=PALETTE[2], lw=1.2, ls="--", alpha=0.7)

    ax_score.set_ylabel("Alarm Score = 1 - p-value")
    ax_score.set_title(
        f"Plot 1 — {detector_label} Alarm Score Used for Drift Decisions",
        pad=10,
    )

    ax_score.legend(loc="upper right", framealpha=0.85)

    ax_gt.fill_between(windows, gt, step="mid", color=PALETTE[2], alpha=0.5, label="GT drift")
    ax_gt.fill_between(windows, alarm_mask.astype(int), step="mid",
                        color=PALETTE[1], alpha=0.4, label="Predicted drift")
    ax_gt.set_yticks([0, 1])
    ax_gt.set_yticklabels(["0", "1"])
    ax_gt.set_ylabel("GT / Pred")
    ax_gt.set_xlabel("Window Index")
    ax_gt.legend(loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 1 saved: {out_path}")



def plot2_state_timeline(df_states, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2, 1.2], "hspace": 0.12})
    ax_state, ax_label, ax_reason = axes

    windows   = df_states["window_idx"].values
    state_ids = df_states["state_id"].values
    unique_states = sorted(set(state_ids.tolist()))
    n_states  = len(unique_states)
    state2col = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_states)}

    for i in range(len(windows) - 1):
        ax_state.fill_between(
            [windows[i], windows[i+1]], [0, 0], [1, 1],
            color=state2col[state_ids[i]], alpha=0.75, linewidth=0
        )
    ax_state.set_yticks([])
    ax_state.set_ylabel("Active State")
    ax_state.set_title("Plot 2 — State Timeline: Active States, Labels, and Transitions", pad=10)

    legend_patches = [mpatches.Patch(color=state2col[s], label=f"State {s}")
                      for s in unique_states]
    ax_state.legend(handles=legend_patches, loc="upper right",
                    ncol=min(n_states, 6), framealpha=0.85)

    if "is_alarm" in df_states.columns:
        alarm_w = windows[df_states["is_alarm"].astype(bool).values]
        ax_state.vlines(alarm_w, 0, 1, color="black", lw=0.6, alpha=0.4, label="Alarm")

    if "major_label" in df_states.columns:
        labels = df_states["major_label"].values
        unique_labels = sorted(set(labels.tolist()))
        label2col = {l: PALETTE[(l + n_states) % len(PALETTE)] for l in unique_labels}
        for i in range(len(windows) - 1):
            ax_label.fill_between(
                [windows[i], windows[i+1]], [0, 0], [1, 1],
                color=label2col[labels[i]], alpha=0.7, linewidth=0
            )
        ax_label.set_yticks([])
        ax_label.set_ylabel("True Label")
        label_patches = [mpatches.Patch(color=label2col[l], label=f"Label {l}")
                         for l in unique_labels]
        ax_label.legend(handles=label_patches, loc="upper right",
                        ncol=min(len(unique_labels), 6), framealpha=0.85)

    if "state_transition_reason" in df_states.columns:
        reason_map = {"stay_current": 0, "revisit_old": 1,
                      "create_new": 2, "first_state": 2,
                      "fallback_new": 2, "fallback_match": 0}
        reason_col_map = {0: PALETTE[0], 1: PALETTE[2], 2: PALETTE[1]}
        reason_label   = {0: "Stay current", 1: "Revisit old", 2: "Create new"}
        reasons = df_states["state_transition_reason"].map(
            lambda x: reason_map.get(str(x).strip(), 0)).values
        for i in range(len(windows) - 1):
            ax_reason.fill_between(
                [windows[i], windows[i+1]], [0, 0], [1, 1],
                color=reason_col_map[reasons[i]], alpha=0.7, linewidth=0
            )
        ax_reason.set_yticks([])
        ax_reason.set_ylabel("Transition")
        r_patches = [mpatches.Patch(color=reason_col_map[k], label=reason_label[k])
                     for k in sorted(reason_col_map)]
        ax_reason.legend(handles=r_patches, loc="upper right",
                         ncol=3, framealpha=0.85)

    ax_reason.set_xlabel("Window Index")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 2 saved: {out_path}")


def plot3_tsne(df_states, state_repr, out_path):
    if not HAS_TSNE:
        print("  ⚠️  sklearn not available — skipping t-SNE plot")
        return

    states_data = state_repr["states"]
    proto_vecs  = []
    proto_ids   = []
    for sid, rep in states_data.items():
        proto_vecs.append(rep["mu_i"])
        proto_ids.append(int(sid))

    if len(proto_vecs) < 2:
        print("  ⚠️  Not enough states for t-SNE — skipping")
        return

    n_states  = len(proto_ids)
    state2col = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(proto_ids)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_tsne, ax_kld = axes

    np.random.seed(42)
    all_vecs  = []
    all_sids  = []
    for sid, rep in states_data.items():
        sid_int = int(sid)
        ni = min(int(rep.get("Ni", 20)), 60)
        mu = np.array(rep["mu_i"])
        noise = np.random.randn(ni, len(mu)) * 0.08
        vecs  = mu[None, :] + noise
        all_vecs.extend(vecs.tolist())
        all_sids.extend([sid_int] * ni)

    all_matrix = np.array(all_vecs)
    perp = min(30, max(5, len(all_matrix) // 4))

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, verbose=0)
    emb = tsne.fit_transform(all_matrix)

    for sid in proto_ids:
        mask = [s == sid for s in all_sids]
        pts  = emb[mask]
        ax_tsne.scatter(pts[:, 0], pts[:, 1], color=state2col[sid],
                        alpha=0.45, s=18, linewidths=0)

    cumulative = 0
    for sid, rep in states_data.items():
        sid_int = int(sid)
        ni = min(int(rep.get("Ni", 20)), 60)
        pts = emb[cumulative:cumulative + ni]
        centre = pts.mean(axis=0)
        ax_tsne.scatter(*centre, color=state2col[sid_int], s=220,
                        marker="*", edgecolors="white", linewidths=0.8,
                        zorder=10, label=f"S{sid_int} prototype")
        ax_tsne.annotate(f"S{sid_int}", centre, fontsize=9, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points",
                         color=state2col[sid_int])
        cumulative += ni

    ax_tsne.set_title("t-SNE: Window Embeddings Coloured by State Assignment")
    ax_tsne.set_xlabel("t-SNE dim 1")
    ax_tsne.set_ylabel("t-SNE dim 2")
    legend_handles = [mpatches.Patch(color=state2col[s], label=f"State {s}")
                      for s in proto_ids]
    ax_tsne.legend(handles=legend_handles, loc="best",
                   ncol=min(n_states, 4), framealpha=0.85)

    kld_data = state_repr.get("pairwise_kld", {})
    sid_list = sorted([int(k) for k in kld_data.keys()])
    if sid_list:
        mat = np.array([[kld_data[str(i)].get(str(j), 0.0)
                         for j in sid_list] for i in sid_list])
        im = ax_kld.imshow(mat, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax_kld, label="Symmetric KLD")
        ticks = list(range(len(sid_list)))
        ax_kld.set_xticks(ticks)
        ax_kld.set_yticks(ticks)
        ax_kld.set_xticklabels([f"S{s}" for s in sid_list])
        ax_kld.set_yticklabels([f"S{s}" for s in sid_list])
        for i in range(len(sid_list)):
            for j in range(len(sid_list)):
                ax_kld.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i, j] > mat.max() * 0.6 else "black")
        ax_kld.set_title("KLD Matrix Between State Pi Distributions")
        ax_kld.set_xlabel("State")
        ax_kld.set_ylabel("State")
    else:
        ax_kld.text(0.5, 0.5, "No KLD data available",
                    ha="center", va="center", transform=ax_kld.transAxes)

    fig.suptitle("Plot 3 — State Geometry: t-SNE Clusters and KLD Separation",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 3 saved: {out_path}")




def plot4_rouge(df_holdout, out_path):
    if df_holdout is None or len(df_holdout) == 0:
        print("  ⚠️  No holdout data — skipping plot 4")
        return

    out_dir = os.path.dirname(out_path)


    has_gold = (
        "gold_answer" in df_holdout.columns
        and df_holdout["gold_answer"].notna().any()
        and (df_holdout["gold_answer"].astype(str).str.strip() != "").any()
    )
    ref_col   = "gold_answer" if has_gold else "fresh_rag_answer"
    ref_label = "Gold Answer"  if has_gold else "Fresh RAG (proxy)"
    print(f"  📏 ROUGE reference: {ref_label}")

    fresh    = df_holdout["fresh_rag_answer"].tolist()
    cached   = df_holdout["cached_rag_answer"].tolist()
    baseline = df_holdout["baseline_answer"].tolist()
    reference = df_holdout[ref_col].tolist()

    rouge_baseline = compute_rouge(baseline, reference)
    rouge_cached   = compute_rouge(cached,   reference)
    rouge_fresh    = compute_rouge(fresh,    reference)

    df = df_holdout.copy()
    df["rl_b"] = rouge_baseline
    df["rl_c"] = rouge_cached
    df["rl_f"] = rouge_fresh

    mb, mc, mf = np.mean(rouge_baseline), np.mean(rouge_cached), np.mean(rouge_fresh)
    gap = abs(mc - mf) * 100

    CLR_B = PALETTE[1]   
    CLR_C = PALETTE[2]   
    CLR_F = PALETTE[0]   
    DARK_BG = "#1a1a1a"  

   
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    methods = [
        "Baseline\n(no retrieval)",
        "Cached RAG\n(zero retrieval)",
        "Fresh RAG\n(live retrieval)",
    ]
    means  = [mb, mc, mf]
    stds   = [np.std(rouge_baseline), np.std(rouge_cached), np.std(rouge_fresh)]
    colors = [CLR_B, CLR_C, CLR_F]

    bars = ax.bar(methods, means, color=colors, width=0.52,
                  edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.errorbar(methods, means, yerr=stds, fmt="none",
                color="white", capsize=6, lw=1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(means)*0.015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="white")

    
    ax.axhline(y=mf, color=CLR_F, linestyle="--", linewidth=1.2, alpha=0.6)

   
    ax.annotate("", xy=(2, mf), xytext=(1, mc),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.8))
    ax.text(1.5, (mc + mf)/2 + max(means)*0.04,
            f"Quality gap:\n{gap:.1f}%",
            ha="center", va="bottom", fontsize=11, color="white")

    ax.set_ylim(0, max(means) * 1.45)
    ax.set_title(f"Overall Mean ROUGE-L per Method\n(reference: {ref_label})",
                 fontsize=14, color="white", pad=12)
    ax.set_ylabel(f"ROUGE-L Score (vs {ref_label})", fontsize=12, color="white")
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.yaxis.grid(True, color="#444444", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    path_a = os.path.join(out_dir, "plot4a_mean_rouge_bar.png")
    fig.savefig(path_a, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅ Plot 4a saved: {path_a}")

  
    if "state_id" in df.columns:
        sm = (df.groupby("state_id")[["rl_b", "rl_c", "rl_f"]]
                .mean()
                .reset_index()
                .sort_values("state_id"))
        n_states = len(sm)
        x = np.arange(n_states)
        w = 0.26

        fig, ax = plt.subplots(figsize=(max(11, n_states * 0.6), 6))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)

        ax.bar(x - w, sm["rl_b"], w, color=CLR_B, label="Baseline",
               edgecolor="none", alpha=0.9)
        ax.bar(x,     sm["rl_c"], w, color=CLR_C, label="Cached RAG",
               edgecolor="none", alpha=0.9)
        ax.bar(x + w, sm["rl_f"], w, color=CLR_F, label="Fresh RAG",
               edgecolor="none", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(s)}" for s in sm["state_id"]],
                           fontsize=10, color="white", rotation=45, ha="right")
        ax.set_title(f"ROUGE-L per State\n(reference: {ref_label})",
                     fontsize=14, color="white", pad=12)
        ax.set_xlabel("State ID", fontsize=12, color="white")
        ax.set_ylabel("Mean ROUGE-L Score", fontsize=12, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.yaxis.grid(True, color="#444444", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, facecolor="#333333",
                  edgecolor="#555555", labelcolor="white", loc="upper right")

        plt.tight_layout()
        path_b = os.path.join(out_dir, "plot4b_rouge_per_state.png")
        fig.savefig(path_b, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  ✅ Plot 4b saved: {path_b}")
    else:
        print("  ⚠️  No state_id column — skipping plot 4b")

  
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    rng = np.random.default_rng(42)
    for i, (vals, col, lbl) in enumerate(zip(
        [rouge_baseline, rouge_cached, rouge_fresh],
        [CLR_B, CLR_C, CLR_F],
        ["Baseline", "Cached RAG", "Fresh RAG"],
    )):
        vals = np.array(vals)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=col, alpha=0.5, s=35, edgecolors="none", label=lbl)
        
        ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2,
                color="white", linewidth=2.8, zorder=5)
       
        ax.scatter([i], [np.mean(vals)], color="white", marker="D",
                   s=70, zorder=6, linewidths=0)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Baseline", "Cached RAG", "Fresh RAG"],
                       fontsize=12, color="white")
    ax.set_title(f"ROUGE-L Score Distribution per Method\n(reference: {ref_label})",
                 fontsize=14, color="white", pad=12)
    ax.set_ylabel("ROUGE-L Score", fontsize=12, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.yaxis.grid(True, color="#444444", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    legend_els = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markersize=9, label="Mean", linestyle="None"),
        Line2D([0], [0], color="white", linewidth=2.8, label="Median"),
    ]
    ax.legend(handles=legend_els, fontsize=11, facecolor="#333333",
              edgecolor="#555555", labelcolor="white", loc="upper right")

    plt.tight_layout()
    path_c = os.path.join(out_dir, "plot4c_score_distribution.png")
    fig.savefig(path_c, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅ Plot 4c saved: {path_c}")


def plot5_quality_cost(df_holdout, df_states, out_path):
    fresh    = df_holdout["fresh_rag_answer"].tolist()
    cached   = df_holdout["cached_rag_answer"].tolist()
    baseline = df_holdout["baseline_answer"].tolist()

    rouge_cached   = np.mean(compute_rouge(cached,   fresh))
    rouge_baseline = np.mean(compute_rouge(baseline, fresh))
    rouge_fresh    = 1.0

    n_total    = len(df_holdout) + (len(df_states) if df_states is not None else 0)
    n_primary  = len(df_states) if df_states is not None else int(n_total * 0.8)
    n_holdout  = len(df_holdout)
    cost_fresh    = n_primary + n_holdout
    cost_cached   = n_primary
    cost_baseline = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"wspace": 0.35})
    ax_scatter, ax_bar = axes

    points = [
        (cost_baseline, rouge_baseline, "Baseline\n(no retrieval)",    PALETTE[1], "o", 180),
        (cost_cached,   rouge_cached,   "Cached RAG\n(thesis method)", PALETTE[2], "*", 400),
        (cost_fresh,    rouge_fresh,    "Fresh RAG\n(upper bound)",    PALETTE[0], "D", 180),
    ]
    for x, y, label, color, marker, size in points:
        ax_scatter.scatter(x, y, color=color, s=size, marker=marker,
                           zorder=5, edgecolors="white", linewidths=1.2)
        ax_scatter.annotate(label, (x, y),
                            xytext=(15, -15), textcoords="offset points",
                            fontsize=9, color=color, fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))

    xs = [cost_baseline, cost_cached, cost_fresh]
    ys = [rouge_baseline, rouge_cached, rouge_fresh]
    ax_scatter.plot(xs, ys, color=GREY, ls="--", lw=1.2, alpha=0.6, zorder=0)

    ax_scatter.annotate("", xy=(cost_cached, rouge_cached),
                        xytext=(cost_fresh, rouge_fresh),
                        arrowprops=dict(arrowstyle="<->", color=PALETTE[2], lw=1.5))
    savings_pct = (cost_fresh - cost_cached) / max(cost_fresh, 1) * 100
    ax_scatter.text((cost_cached + cost_fresh)/2,
                    (rouge_cached + rouge_fresh)/2 + 0.02,
                    f"{savings_pct:.0f}% fewer\nretrievals",
                    ha="center", fontsize=9, color=PALETTE[2])

    ax_scatter.set_xlabel("Number of Retrieval Calls")
    ax_scatter.set_ylabel("ROUGE-L Score (vs Fresh RAG)")
    ax_scatter.set_title("Quality vs Retrieval Cost\n(Efficiency-Quality Trade-off)")
    ax_scatter.set_xlim(-cost_fresh * 0.1, cost_fresh * 1.2)
    ax_scatter.set_ylim(-0.05, 1.15)

    categories = ["Baseline\n(no retrieval)", "Cached RAG\n(thesis)", "Fresh RAG\n(upper bound)"]
    costs      = [cost_baseline, cost_cached, cost_fresh]
    colors_bar = [PALETTE[1], PALETTE[2], PALETTE[0]]

    bars = ax_bar.bar(categories, costs, color=colors_bar, alpha=0.82,
                      edgecolor="white", width=0.5)
    for bar, c in zip(bars, costs):
        ax_bar.text(bar.get_x() + bar.get_width()/2, c + max(costs)*0.01,
                    f"{c:,}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    saved = cost_fresh - cost_cached
    ax_bar.annotate("",
        xy=(0.5, cost_cached),
        xytext=(0.5, cost_fresh),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5)
    )
    ax_bar.text(0.65, (cost_cached + cost_fresh)/2,
                f"{saved:,} retrievals\nsaved ({savings_pct:.0f}%)",
                fontsize=9, color="black", va="center")

    ax_bar.set_ylabel("Total Retrieval Calls")
    ax_bar.set_title("Retrieval Cost per Method")

    fig.suptitle("Plot 5 — Efficiency-Quality Trade-off: Cached RAG vs Fresh RAG",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Plot 5 saved: {out_path}")



def plot_combined(exp_dir):
    """Load individual saved plots and stitch into one combined figure."""
    paths = [
        os.path.join(exp_dir, "plot1_drift_score.png"),
        os.path.join(exp_dir, "plot2_state_timeline.png"),
        os.path.join(exp_dir, "plot3_tsne.png"),
        os.path.join(exp_dir, "plot4a_mean_rouge_bar.png"),
        os.path.join(exp_dir, "plot4b_rouge_per_state.png"),
        os.path.join(exp_dir, "plot4c_score_distribution.png"),
        os.path.join(exp_dir, "plot5_quality_vs_cost.png"),
        os.path.join(exp_dir, "plot6_shifted_auc.png"),
        os.path.join(exp_dir, "plot7_score_distribution_by_gt.png"),
        os.path.join(exp_dir, "plot8_state_method_comparison.png"),
    ]
    imgs = []
    for p in paths:
        if os.path.exists(p):
            from matplotlib.image import imread
            imgs.append((os.path.basename(p), imread(p)))

    if not imgs:
        return

    n_imgs = len(imgs)
    ncols  = 2
    nrows  = (n_imgs + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 7))
    axes = axes.ravel()

    for i, (name, img) in enumerate(imgs):
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(name.replace(".png", "").replace("_", " "),
                          fontsize=10, color="#444")

    for j in range(n_imgs, len(axes)):
        axes[j].axis("off")

    fig.suptitle("STAD — Thesis Plots Overview", fontsize=18, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out = os.path.join(exp_dir, "plot_all_combined.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  ✅ Combined plot saved: {out}")



def main():
    parser = argparse.ArgumentParser(description="STAD Thesis Plots")
    parser.add_argument("--exp_dir", type=str, default=".",
                        help="Path to experiment output directory")
    parser.add_argument("--max_shift", type=int, default=10,
                        help="Maximum ground-truth shift in windows for shifted AUROC")
    parser.add_argument("--skip_global_summary", action="store_true",
                        help="Do not refresh output/_analysis_summary after this run")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"STAD Thesis Plots — reading from: {exp_dir}")
    print(f"{'='*60}\n")

    print("📂 Loading data...")
    df_scores  = load_scores(exp_dir)
    df_states  = load_states(exp_dir)
    df_holdout = load_holdout(exp_dir)
    state_repr = load_state_repr(exp_dir)

    print(f"\n📊 Generating plots...\n")

    plot1_drift_score(
        df_scores,
        os.path.join(exp_dir, "plot1_drift_score.png"),
    )
    plot2_state_timeline(
        df_states,
        os.path.join(exp_dir, "plot2_state_timeline.png"),
    )
    plot3_tsne(
        df_states, state_repr,
        os.path.join(exp_dir, "plot3_tsne.png"),
    )
    
    plot4_rouge(
        df_holdout,
        os.path.join(exp_dir, "plot4_rouge_comparison.png"),  # dir is used, filename ignored
    )
    plot5_quality_cost(
        df_holdout, df_states,
        os.path.join(exp_dir, "plot5_quality_vs_cost.png"),
    )

    print("\n📌 Computing extra drift/state diagnostics...\n")

    save_run_metrics_summary(
        exp_dir,
        df_scores,
        df_states,
        max_shift=args.max_shift,
    )

    plot6_shifted_auc(
        df_scores,
        os.path.join(exp_dir, "plot6_shifted_auc.png"),
        max_shift=args.max_shift,
    )

    plot7_score_distribution(
        df_scores,
        os.path.join(exp_dir, "plot7_score_distribution_by_gt.png"),
    )

    plot8_state_method_comparison(
        exp_dir,
        df_scores,
        df_states,
        os.path.join(exp_dir, "plot8_state_method_comparison.png"),
    )

    if not args.skip_global_summary:
        update_global_summary(exp_dir, max_shift=args.max_shift)

    plot_combined(exp_dir)

    print(f"\n✅ All plots saved to: {exp_dir}")
    print("   plot1_drift_score.png")
    print("   plot2_state_timeline.png")
    print("   plot3_tsne.png")
    print("   plot4a_mean_rouge_bar.png")
    print("   plot4b_rouge_per_state.png")
    print("   plot4c_score_distribution.png")
    print("   plot5_quality_vs_cost.png")
    print("   plot6_shifted_auc.png")
    print("   plot7_score_distribution_by_gt.png")
    print("   plot8_state_method_comparison.png")
    print("   shifted_auc_by_shift.csv")
    print("   state_method_metrics.csv")
    print("   run_metrics_summary.json")
    print("   plot_all_combined.png")


if __name__ == "__main__":
    main()
