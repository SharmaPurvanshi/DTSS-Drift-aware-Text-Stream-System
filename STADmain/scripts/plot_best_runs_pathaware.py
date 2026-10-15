
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, v_measure_score


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
STATE_ORDER = ["cosine", "kld"]

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

SELECTED_PR_PAIRS = {
    "AGNewsStreamSim": "energy",
    "CLINCStreamSim": "energy",
    "GaliboStreamSim": "energy",
    "KILTStreamSim": "energy",
    "MultiHopRAGStreamSim": "ks_alibi",
    "PopQAStreamSim": "energy",
    "RAGBenchStreamSim": "ks_alibi",
}


def safe_auc(y, s) -> float:
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


def number_after(pattern: str, text: str) -> float:
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


def int_after(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    if not m:
        return 0

    try:
        return int(float(m.group(1)))
    except Exception:
        return 0


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["score_used", "alarm_score", "ks_p_score", "mmd2", "energy2"]:
        if c in df.columns:
            return c
    return None


def parse_log(log_path: Path) -> Dict:
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding="utf-8", errors="replace")

    event_m = re.search(r"events=(\d+)\s+alarms=(\d+)", text)
    gt_m = re.search(r"GT positives:\s*(\d+)\s*/\s*(\d+)", text)

    return {
        "event_count": int(event_m.group(1)) if event_m else 0,
        "log_alarm_count": int(event_m.group(2)) if event_m else int_after(r"alarms=(\d+)", text),
        "event_precision": number_after(r"EP\s*=", text),
        "event_recall": number_after(r"ER\s*=", text),
        "event_f1": number_after(r"EF1\s*=", text),
        "nab": number_after(r"NAB\s*=", text),
        "v_measure_log": number_after(r"V-measure\s*=", text),
        "trade_score_log": number_after(r"trade_score\s*=", text),
        "gt_positive_windows_log": int(gt_m.group(1)) if gt_m else 0,
        "total_windows_log": int(gt_m.group(2)) if gt_m else 0,
    }


def compute_v_measure_from_csv(df: pd.DataFrame) -> Tuple[float, int]:
    if "state_id" not in df.columns or "major_label" not in df.columns:
        return np.nan, 0

    labels = pd.to_numeric(df["major_label"], errors="coerce").fillna(-1).astype(int)
    states = pd.to_numeric(df["state_id"], errors="coerce").fillna(-1).astype(int)

    try:
        return float(v_measure_score(labels, states)), int(states.nunique())
    except Exception:
        return np.nan, int(states.nunique())


def shifted_auc_table(df: pd.DataFrame, max_shift: int) -> pd.DataFrame:
    if "gt" not in df.columns:
        return pd.DataFrame()

    score_col = get_score_col(df)
    if score_col is None:
        return pd.DataFrame()

    y = pd.to_numeric(df["gt"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)

    rows = []
    for shift in range(max_shift + 1):
        y_shift = y.shift(shift, fill_value=0)
        rows.append({
            "shift": shift,
            "shifted_auc": safe_auc(y_shift, s),
        })

    return pd.DataFrame(rows)


def extract_config_values(path: Path) -> Dict:
    s = str(path)

    def get_int(pattern: str) -> int:
        m = re.search(pattern, s)
        if not m:
            return 0
        for g in m.groups():
            if g is not None:
                return int(g)
        return 0

    def get_float(pattern: str) -> float:
        m = re.search(pattern, s)
        if not m:
            return np.nan
        return float(m.group(1))

    return {
        "win_len": get_int(r"win(\d+)"),
        "steps": get_int(r"step(\d+)"),
        "l_hist_size": get_int(r"lhist(\d+)"),
        "l_new_size": get_int(r"lnew(\d+)"),
        "hist_min_size": get_int(r"minHist_(\d+)|minHist(\d+)"),
        "alpha": get_float(r"alpha([0-9.]+)"),
        "lambda_value": get_float(r"_lam([0-9.]+)"),
        "state_sim_threshold": get_float(r"simth([0-9.]+)"),
        "normdet": get_int(r"normdet([01])"),
    }


def find_best_run_csvs(output_root: Path) -> List[Tuple[str, str, str, Path]]:
  
    found = []

    for detector in DETECTOR_ORDER:
        for state_logic in STATE_ORDER:
            for dataset in DATASET_ORDER:
                dataset_root = output_root / detector / state_logic / dataset

                if not dataset_root.exists():
                    continue

                candidates = sorted(
                    dataset_root.rglob("best_*/online_learning/drift_window_scores.csv")
                )

                for csv_path in candidates:
                    found.append((dataset, detector, state_logic, csv_path))

    return found


def read_best_runs(output_root: Path, max_shift: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    found = find_best_run_csvs(output_root)

    rows: List[Dict] = []
    shift_rows: List[Dict] = []

    for dataset, detector, state_logic, csv_path in found:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] Could not read {csv_path}: {e}")
            continue

        online_dir = csv_path.parent
        run_dir = online_dir.parent

        log_metrics = parse_log(online_dir / "log")
        v_csv, n_states_csv = compute_v_measure_from_csv(df)

        if np.isfinite(log_metrics.get("v_measure_log", np.nan)):
            v_measure = log_metrics["v_measure_log"]
        else:
            v_measure = v_csv

        event_f1 = log_metrics.get("event_f1", np.nan)
        nab = log_metrics.get("nab", np.nan)
        alarm_count = log_metrics.get("log_alarm_count", 0)
        total_windows = log_metrics.get("total_windows_log", 0)
        alarm_ratio = alarm_count / total_windows if total_windows else np.nan

        selection_score = (
            0.55 * (event_f1 if np.isfinite(event_f1) else -1.0)
            + 0.45 * (nab if np.isfinite(nab) else -1.0)
            - (1.0 if np.isfinite(alarm_ratio) and alarm_ratio > 0.60 else 0.0)
            - (0.75 if np.isfinite(nab) and nab < 0 else 0.0)
        )

        row = {
            "dataset": dataset,
            "dataset_label": DATASET_LABEL.get(dataset, dataset),
            "detector": detector,
            "detector_label": DETECTOR_LABEL.get(detector, detector),
            "state_logic": state_logic,
            "run_name": run_dir.name,
            "online_dir": str(online_dir),
            "csv_path": str(csv_path),
            "v_measure": v_measure,
            "v_measure_csv": v_csv,
            "num_states_csv": n_states_csv,
            "alarm_ratio": alarm_ratio,
            "selection_score": selection_score,
        }

        row.update(extract_config_values(csv_path))
        row.update(log_metrics)

        rows.append(row)

        auc_table = shifted_auc_table(df, max_shift=max_shift)
        for _, rr in auc_table.iterrows():
            shift_rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABEL.get(dataset, dataset),
                "detector": detector,
                "detector_label": DETECTOR_LABEL.get(detector, detector),
                "state_logic": state_logic,
                "run_name": run_dir.name,
                "csv_path": str(csv_path),
                "shift": int(rr["shift"]),
                "shifted_auc": rr["shifted_auc"],
            })

    return pd.DataFrame(rows), pd.DataFrame(shift_rows)


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def select_best(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    return (
        df.sort_values(
            group_cols + ["selection_score", "event_f1", "nab"],
            ascending=[True] * len(group_cols) + [False, False, False],
        )
        .groupby(group_cols, observed=True)
        .head(1)
        .reset_index(drop=True)
    )


def plot_vmeasure_cosine_vs_kld(all_runs: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "vmeasure_cosine_vs_kld"
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_state = select_best(all_runs, ["dataset", "detector", "state_logic"])

    for detector in DETECTOR_ORDER:
        rows = []

        for dataset in DATASET_ORDER:
            sub = best_state[
                (best_state["dataset"] == dataset)
                & (best_state["detector"] == detector)
            ]

            cos = sub[sub["state_logic"] == "cosine"]
            kld = sub[sub["state_logic"] == "kld"]

            rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABEL.get(dataset, dataset),
                "cosine_v_measure": float(cos["v_measure"].iloc[0]) if not cos.empty else np.nan,
                "kld_v_measure": float(kld["v_measure"].iloc[0]) if not kld.empty else np.nan,
            })

        p = pd.DataFrame(rows)

        x = np.arange(len(p))
        width = 0.36

        plt.figure(figsize=(10.5, 4.8))
        plt.bar(x - width / 2, p["cosine_v_measure"], width, label="Cosine")
        plt.bar(x + width / 2, p["kld_v_measure"], width, label="KLD")

        plt.xticks(x, p["dataset_label"], rotation=30, ha="right")
        plt.ylabel("V-measure")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        savefig(plot_dir, f"vmeasure_cosine_vs_kld_{detector}")


def plot_shifted_auc_per_dataset(all_runs: pd.DataFrame, shift_df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "shifted_auc_by_dataset"
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_det = select_best(all_runs, ["dataset", "detector"])

    for dataset in DATASET_ORDER:
        plt.figure(figsize=(7.5, 4.8))
        has_any = False

        for detector in DETECTOR_ORDER:
            selected = best_det[
                (best_det["dataset"] == dataset)
                & (best_det["detector"] == detector)
            ]

            if selected.empty:
                continue

            csv_path = selected["csv_path"].iloc[0]
            curve = shift_df[shift_df["csv_path"] == csv_path].sort_values("shift")

            if curve.empty:
                continue

            plt.plot(
                curve["shift"],
                curve["shifted_auc"],
                marker="o",
                linewidth=2,
                label=DETECTOR_LABEL.get(detector, detector),
            )
            has_any = True

        if not has_any:
            plt.close()
            continue

        plt.xlabel("Ground-truth shift in windows")
        plt.ylabel("AUROC")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)

        savefig(plot_dir, f"shifted_auc_{dataset}")


def plot_f1_nab_per_detector(all_runs: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "event_f1_nab_by_detector"
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_det = select_best(all_runs, ["dataset", "detector"])

    for detector in DETECTOR_ORDER:
        rows = []

        for dataset in DATASET_ORDER:
            selected = best_det[
                (best_det["dataset"] == dataset)
                & (best_det["detector"] == detector)
            ]

            if selected.empty:
                rows.append({
                    "dataset": dataset,
                    "dataset_label": DATASET_LABEL.get(dataset, dataset),
                    "event_f1": np.nan,
                    "nab": np.nan,
                })
            else:
                r = selected.iloc[0]
                rows.append({
                    "dataset": dataset,
                    "dataset_label": DATASET_LABEL.get(dataset, dataset),
                    "event_f1": r["event_f1"],
                    "nab": r["nab"],
                })

        p = pd.DataFrame(rows)
        x = np.arange(len(p))
        width = 0.36

        plt.figure(figsize=(10.5, 4.8))
        plt.bar(x - width / 2, p["event_f1"], width, label="Event F1")
        plt.bar(x + width / 2, p["nab"], width, label="NAB")
        plt.axhline(0, linewidth=0.8)

        plt.xticks(x, p["dataset_label"], rotation=30, ha="right")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        savefig(plot_dir, f"event_f1_nab_{detector}")


def plot_selected_precision_recall(all_runs: pd.DataFrame, out_dir: Path) -> None:
    best_det = select_best(all_runs, ["dataset", "detector"])

    rows = []

    for dataset in DATASET_ORDER:
        detector = SELECTED_PR_PAIRS.get(dataset)

        selected = best_det[
            (best_det["dataset"] == dataset)
            & (best_det["detector"] == detector)
        ]

        dataset_label = DATASET_LABEL.get(dataset, dataset)
        detector_label = DETECTOR_LABEL.get(detector, detector)

        # X-axis label now contains both dataset and selected detector.
        x_label = f"{dataset_label}\n({detector_label})"

        if selected.empty:
            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "event_precision": np.nan,
                "event_recall": np.nan,
            })
        else:
            r = selected.iloc[0]
            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "event_precision": r["event_precision"],
                "event_recall": r["event_recall"],
            })

    p = pd.DataFrame(rows)
    x = np.arange(len(p))
    width = 0.36

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, p["event_precision"], width, label="Event precision")
    plt.bar(x + width / 2, p["event_recall"], width, label="Event recall")

    plt.xticks(x, p["x_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    savefig(out_dir, "selected_event_precision_recall")


def plot_event_f1_detector_comparison(all_runs: pd.DataFrame, out_dir: Path) -> None:
    best_det = select_best(all_runs, ["dataset", "detector"])

    pivot = best_det.pivot_table(
        index="dataset",
        columns="detector",
        values="event_f1",
        aggfunc="max",
        observed=True,
    )

    pivot = pivot.reindex(index=DATASET_ORDER, columns=DETECTOR_ORDER)

    x = np.arange(len(pivot))
    width = 0.24

    plt.figure(figsize=(10.5, 4.8))

    for i, detector in enumerate(DETECTOR_ORDER):
        plt.bar(
            x + (i - 1) * width,
            pivot[detector],
            width,
            label=DETECTOR_LABEL.get(detector, detector),
        )

    plt.xticks(
        x,
        [DATASET_LABEL.get(ds, ds) for ds in pivot.index],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Event F1")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    savefig(out_dir, "event_f1_detector_comparison")

def plot_selected_vmeasure_trade_score(all_runs: pd.DataFrame, out_dir: Path) -> None:
    best_det = select_best(all_runs, ["dataset", "detector"])

    rows = []

    for dataset in DATASET_ORDER:
        detector = SELECTED_PR_PAIRS.get(dataset)

        selected = best_det[
            (best_det["dataset"] == dataset)
            & (best_det["detector"] == detector)
        ]

        dataset_label = DATASET_LABEL.get(dataset, dataset)
        detector_label = DETECTOR_LABEL.get(detector, detector)
        x_label = f"{dataset_label}\n({detector_label})"

        if selected.empty:
            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "v_measure": np.nan,
                "trade_score": np.nan,
            })
        else:
            r = selected.iloc[0]

            # v_measure is stored as "v_measure" in this script.
            v_measure = r["v_measure"] if "v_measure" in r else np.nan

            # trade score comes from the log parser as "trade_score_log".
            if "trade_score_log" in r and pd.notna(r["trade_score_log"]):
                trade_score = r["trade_score_log"]
            elif "trade_score" in r and pd.notna(r["trade_score"]):
                trade_score = r["trade_score"]
            else:
                trade_score = np.nan

            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "v_measure": v_measure,
                "trade_score": trade_score,
            })

    p = pd.DataFrame(rows)
    x = np.arange(len(p))
    width = 0.36

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, p["v_measure"], width, label="V-measure")
    plt.bar(x + width / 2, p["trade_score"], width, label="Trade score")

    plt.xticks(x, p["x_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    savefig(out_dir, "selected_vmeasure_trade_score")


def plot_selected_event_f1_nab(all_runs: pd.DataFrame, out_dir: Path) -> None:
   
    best_det = select_best(all_runs, ["dataset", "detector"])

    rows = []

    for dataset in DATASET_ORDER:
        detector = SELECTED_PR_PAIRS.get(dataset)

        selected = best_det[
            (best_det["dataset"] == dataset)
            & (best_det["detector"] == detector)
        ]

        dataset_label = DATASET_LABEL.get(dataset, dataset)
        detector_label = DETECTOR_LABEL.get(detector, detector)
        x_label = f"{dataset_label}\n({detector_label})"

        if selected.empty:
            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "event_f1": np.nan,
                "nab": np.nan,
            })
        else:
            r = selected.iloc[0]
            rows.append({
                "dataset": dataset,
                "dataset_label": dataset_label,
                "detector": detector,
                "detector_label": detector_label,
                "x_label": x_label,
                "event_f1": r["event_f1"],
                "nab": r["nab"],
            })

    p = pd.DataFrame(rows)
    x = np.arange(len(p))
    width = 0.36

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, p["event_f1"], width, label="Event F1")
    plt.bar(x + width / 2, p["nab"], width, label="NAB")
    plt.axhline(0, linewidth=0.8)

    plt.xticks(x, p["x_label"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(min(-0.05, np.nanmin(p[["event_f1", "nab"]].values) - 0.05), 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    savefig(out_dir, "selected_event_f1_nab")

def plot_selected_vmeasure_cosine_vs_kld(all_runs: pd.DataFrame, out_dir: Path) -> None:
   
    best_state = select_best(all_runs, ["dataset", "detector", "state_logic"])

    rows = []

    for dataset in DATASET_ORDER:
        detector = SELECTED_PR_PAIRS.get(dataset)

        sub = best_state[
            (best_state["dataset"] == dataset)
            & (best_state["detector"] == detector)
        ]

        cos = sub[sub["state_logic"] == "cosine"]
        kld = sub[sub["state_logic"] == "kld"]

        dataset_label = DATASET_LABEL.get(dataset, dataset)
        detector_label = DETECTOR_LABEL.get(detector, detector)
        x_label = f"{dataset_label}\n({detector_label})"

        rows.append({
            "dataset": dataset,
            "dataset_label": dataset_label,
            "detector": detector,
            "detector_label": detector_label,
            "x_label": x_label,
            "cosine_v_measure": float(cos["v_measure"].iloc[0]) if not cos.empty else np.nan,
            "kld_v_measure": float(kld["v_measure"].iloc[0]) if not kld.empty else np.nan,
        })

    p = pd.DataFrame(rows)

    x = np.arange(len(p))
    width = 0.36

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width / 2, p["cosine_v_measure"], width, label="Cosine")
    plt.bar(x + width / 2, p["kld_v_measure"], width, label="KLD")

    plt.xticks(x, p["x_label"], rotation=30, ha="right")
    plt.ylabel("V-measure")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    savefig(out_dir, "selected_vmeasure_cosine_vs_kld")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/purvanshi.sharma/thesis/code/STADmain/output",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/purvanshi.sharma/thesis/code/STADmain/output/_best_run_plots",
    )
    parser.add_argument("--max-shift", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs, shift_df = read_best_runs(
        output_root=output_root,
        max_shift=args.max_shift,
    )

    if all_runs.empty:
        raise RuntimeError(
            f"No best_* runs found under {output_root}. Expected paths like "
            f"output/energy/cosine/AGNewsStreamSim/minHist_50/best_*/online_learning/drift_window_scores.csv"
        )

    all_runs["dataset"] = pd.Categorical(
        all_runs["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    all_runs["detector"] = pd.Categorical(
        all_runs["detector"],
        categories=DETECTOR_ORDER,
        ordered=True,
    )
    all_runs["state_logic"] = pd.Categorical(
        all_runs["state_logic"],
        categories=STATE_ORDER,
        ordered=True,
    )

    all_runs = all_runs.sort_values(
        ["dataset", "detector", "state_logic", "selection_score"],
        ascending=[True, True, True, False],
    )

    all_runs.to_csv(out_dir / "best_runs_metrics.csv", index=False)
    shift_df.to_csv(out_dir / "best_runs_shifted_auc.csv", index=False)

    select_best(all_runs, ["dataset", "detector"]).to_csv(
        out_dir / "selected_best_per_dataset_detector.csv",
        index=False,
    )

    select_best(all_runs, ["dataset", "detector", "state_logic"]).to_csv(
        out_dir / "selected_best_per_dataset_detector_state.csv",
        index=False,
    )

    plot_vmeasure_cosine_vs_kld(all_runs, out_dir)
    plot_selected_vmeasure_cosine_vs_kld(all_runs, out_dir)
    plot_shifted_auc_per_dataset(all_runs, shift_df, out_dir)
    plot_f1_nab_per_detector(all_runs, out_dir)
    plot_selected_precision_recall(all_runs, out_dir)
    plot_selected_event_f1_nab(all_runs, out_dir)
    plot_selected_vmeasure_trade_score(all_runs, out_dir)
    plot_event_f1_detector_comparison(all_runs, out_dir)
    print("\nSaved outputs in:")
    print(f"  {out_dir.resolve()}")

    print("\nMain CSV files:")
    print(f"  {out_dir / 'best_runs_metrics.csv'}")
    print(f"  {out_dir / 'best_runs_shifted_auc.csv'}")
    print(f"  {out_dir / 'selected_best_per_dataset_detector.csv'}")
    print(f"  {out_dir / 'selected_best_per_dataset_detector_state.csv'}")

    print("\nPlot folders/files:")
    print(f"  {out_dir / 'vmeasure_cosine_vs_kld'}")
    print(f"  {out_dir / 'shifted_auc_by_dataset'}")
    print(f"  {out_dir / 'event_f1_nab_by_detector'}")
    print(f"  {out_dir / 'selected_event_precision_recall.pdf'}")
    print(f"  {out_dir / 'event_f1_detector_comparison.pdf'}")
    print(f"  {out_dir / 'selected_vmeasure_trade_score.pdf'}")
    print(f"  {out_dir / 'selected_event_f1_nab.pdf'}")
    print(f"  {out_dir / 'selected_vmeasure_cosine_vs_kld.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
