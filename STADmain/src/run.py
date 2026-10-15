import os
import re
import json
import time
import itertools
from datetime import datetime
from pathlib import Path
import argparse
import shutil
import random
import csv

from loguru import logger
from model.predict import Predict


def ensure_list(x):
    return x if isinstance(x, list) else [x]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def config_key_for_dataset(dataset_name: str) -> str:
  
    return dataset_name


try:
    nice_val = os.nice(10)
    logger.info(f"Process niceness increased by 10; current nice value = {nice_val}")
except AttributeError:
    logger.warning("os.nice is not available on this platform; using default priority.")
except OSError as e:
    logger.warning(f"Could not change process niceness: {e}")



parser = argparse.ArgumentParser(
    description="Run drift-driven state management with comprehensive RAG analysis"
)

parser.add_argument("--gpu", type=int, default=-1)

parser.add_argument(
    "--drift_mode",
    type=str,
    default="mmd",
    choices=["ks", "ks_alibi", "mmd", "energy"],
)

parser.add_argument("--lambda_mmd", type=float, default=0.001)

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=[
        "CLINCStreamSim",
        "GaliboStreamSim",
        "AGNewsStreamSim",
        "MINDStreamSim",
        "BEIRStreamSim",

       
        "KILTStreamSim",
        "RAGBenchStreamSim",
        "PopQAStreamSim",
        "MultiHopRAGStreamSim",
        "StateCacheQAStreamSim",
    ],
)

parser.add_argument("--enable_rag", action="store_true")
parser.add_argument("--knowledge_base", type=str, default=None)
parser.add_argument("--num_test_queries", type=int, default=10)

args = parser.parse_args()



if args.gpu is not None and args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(f"Restricting CUDA_VISIBLE_DEVICES to physical GPU {args.gpu}")
else:
    logger.info("No GPU index provided; running on CPU or default CUDA config.")

import torch

from dataPrepare.text_stream import (
    CLINCStreamSim,
    GaliboStreamSim,
    AGNewsStreamSim,
    MINDStreamSim,
    BEIRStreamSim,

    KILTStreamSim,
    MultiHopRAGStreamSim,
    PopQAStreamSim,
    RAGBenchStreamSim,
    StateCacheQAStreamSim,
)


if torch.cuda.is_available():
    if torch.cuda.device_count() > 0 and (args.gpu is not None and args.gpu >= 0):
        gpu = 0
        torch.cuda.set_device(gpu)
        logger.info(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")
    else:
        gpu = None
        logger.info("CUDA available but no GPU requested; using CPU.")
else:
    gpu = None
    logger.info("CUDA not available; using CPU.")


root = Path(__file__).resolve().parent.parent

dataset_obj = {
    "CLINCStreamSim": CLINCStreamSim,

    "GaliboStreamSim": GaliboStreamSim,

    "AGNewsStreamSim": AGNewsStreamSim,

    "MINDStreamSim": MINDStreamSim,

    "BEIRStreamSim": BEIRStreamSim,


    "KILTStreamSim": KILTStreamSim,
    "MultiHopRAGStreamSim": MultiHopRAGStreamSim,
    "PopQAStreamSim": PopQAStreamSim,
    "RAGBenchStreamSim": RAGBenchStreamSim,
    "StateCacheQAStreamSim": StateCacheQAStreamSim,
}



_LLM_LABELED_DATASETS = set()

_RAG_BENCHMARK_DATASETS = {
    "KILTStreamSim",
    "MultiHopRAGStreamSim",
    "PopQAStreamSim",
    "RAGBenchStreamSim",
    "StateCacheQAStreamSim",
}


if args.dataset:
    if args.dataset not in dataset_obj:
        logger.error(f"Unknown dataset: {args.dataset}")
        raise SystemExit(1)

    if args.dataset in _LLM_LABELED_DATASETS and not os.environ.get("GROQ_API_KEY"):
        logger.warning(f"{args.dataset} uses LLM labeling via Groq. Set GROQ_API_KEY.")

    datasets = [args.dataset]
    logger.info(f"Running single dataset: {args.dataset}")
else:
    datasets = ["KILTStreamSim"]
    logger.info(f"Running default datasets: {datasets}")


config_path = root / "configuration" / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)


_RAG_DATASET_DEFAULT_CONFIG = {
    "win_len": [25],
    "steps": [25],
    "hidden_size": [128],
    "epochs": [10],
    "lr": [1e-4],
    "dropout": [0.2],
    "batch_size": [32],
    "weight_decay": [1e-5],
    "dim": [384],
    "early_stopping_patience": [5],
    "ks_alpha": [0.05],
    "ks_hist_min_size": [50],
    "kld_epsilon": [1e-3],

 
    "state_logic": ["cosine"],

    "max_state_num": [50],
    "l_hist_size": [100],
    "l_new_size": [10],
    "mmd_l_hist_size": [100],
    "mmd_l_new_size": [10],
    "mmd_hist_min_size": [50],
    "lambda_mmd": [0.001],
    "homo_threshold": [0.8],
    "state_sim_threshold": [0.90],
    "state_update_alpha": [0.05],
    "force_new_state_on_alarm": [False],
    "log_every": [10],
    "normalize_for_detectors": [False],
    "event_tolerance_k": [3],
    "event_window_W": [10],
    "event_fp_penalty": [0.11],
    "state_penalty_lambda": [1.0],

  
    "cap": [15000],
}


result_dir = "output"
model_name = "STAD_TEXT_COMPREHENSIVE"
seeds = [42]

drift_mode = args.drift_mode.lower().strip()
lambda_mmd_cli = float(args.lambda_mmd)

logger.info(f"[run.py] drift_mode: {drift_mode}")

if args.enable_rag:
    logger.info("[run.py] RAG ENABLED - Simplified 3-Way Comparison")
else:
    logger.info("[run.py] RAG disabled - running drift detection only")



def _text_is_question(text: str) -> bool:
    t = text.strip()

    if not t or t.endswith("?"):
        return True

    first_word = t.split()[0].lower()

    question_starters = {
        "who", "what", "when", "where", "why", "how", "which",
        "is", "are", "was", "were", "does", "did", "do",
        "can", "could", "would", "will", "has", "have", "had",
    }

    return first_word in question_starters and len(t.split()) < 25


def _build_kilt_kb(output_dir: str, max_docs: int = 8000) -> str:
    kb_path = os.path.join(output_dir, "knowledge_base_kilt_wiki.json")

    if os.path.exists(kb_path):
        try:
            with open(kb_path) as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f"📚 Reusing KILT KB: {kb_path} ({len(existing)} passages)")
                return kb_path
        except Exception:
            pass

    print("📚 Building KILT KB from provenance passages...")

    from datasets import load_dataset

    passages = []
    seen = set()

    for task_config in ["nq", "hotpotqa", "triviaqa", "fever"]:
        try:
            ds = load_dataset("kilt_tasks", task_config, split="train")

            for row in ds:
                for prov_item in (row.get("output") or []):
                    for p in (prov_item.get("provenance") or []):
                        text = (p.get("text") or p.get("snippet", "")).strip()

                        if not text or len(text) < 30 or _text_is_question(text):
                            continue

                        title = p.get("title", "").strip()
                        full = (f"{title}. {text}" if title else text)[:1200]

                        if full in seen:
                            continue

                        seen.add(full)

                        passages.append({
                            "id": len(passages),
                            "text": full,
                            "title": title,
                            "category": task_config,
                        })

                    if len(passages) >= max_docs:
                        break

                if len(passages) >= max_docs:
                    break

            print(f"   KILT/{task_config}: {len(passages)} passages")

            if len(passages) >= max_docs:
                break

        except Exception as e:
            print(f"   KILT/{task_config} failed: {e}")

    if not passages:
        return _build_wikipedia_kb(output_dir, max_docs=max_docs)

    os.makedirs(output_dir, exist_ok=True)

    with open(kb_path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"KILT KB built: {len(passages)} passages → {kb_path}")

    return kb_path


def _build_wikipedia_kb(output_dir: str, max_docs: int = 5000) -> str:
    kb_path = os.path.join(output_dir, "knowledge_base_wikipedia.json")

    if os.path.exists(kb_path):
        try:
            with open(kb_path) as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f" Reusing Wikipedia KB: {kb_path} ({len(existing)} passages)")
                return kb_path
        except Exception:
            pass

    print(" Building Wikipedia KB (streaming)...")

    from datasets import load_dataset

    passages = []
    seen = set()

    wiki_sources = [
        ("Cohere/wikipedia-22-12-en-embeddings", "train", "text", "title"),
        ("wikipedia", "20220301.en", "text", "title"),
    ]

    for hf_id, split_or_config, text_col, title_col in wiki_sources:
        try:
            ds = (
                load_dataset(hf_id, split="train", streaming=True)
                if split_or_config == "train"
                else load_dataset(hf_id, split_or_config, split="train", streaming=True)
            )

            for row in ds:
                text = str(row.get(text_col, "")).strip()

                if len(text) < 40 or _text_is_question(text):
                    continue

                title = str(row.get(title_col, "")).strip()
                passage = text[:1200]

                if passage in seen:
                    continue

                seen.add(passage)

                passages.append({
                    "id": len(passages),
                    "text": passage,
                    "title": title,
                    "category": "wikipedia",
                })

                if len(passages) >= max_docs:
                    break

            if passages:
                print(f"   Wikipedia KB: {len(passages)} passages from {hf_id}")
                break

        except Exception as e:
            print(f"   {hf_id} failed: {e}")

    if not passages:
        os.makedirs(output_dir, exist_ok=True)
        with open(kb_path, "w") as f:
            json.dump([], f)
        return kb_path

    os.makedirs(output_dir, exist_ok=True)

    with open(kb_path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"Wikipedia KB built: {len(passages)} passages → {kb_path}")

    return kb_path


def _build_ragbench_kb(dataset_instance, output_dir: str) -> str:
    kb_path = os.path.join(output_dir, "knowledge_base_ragbench.json")

    if os.path.exists(kb_path):
        try:
            with open(kb_path) as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f"📚 Reusing RAGBench KB: {kb_path} ({len(existing)} passages)")
                return kb_path
        except Exception:
            pass

    print(" Building RAGBench KB from document fields...")

    from datasets import load_dataset

    passages = []
    seen = set()

    for subset, category in [
        ("techqa", "tech_support"),
        ("covidqa", "medical"),
        ("finqa", "finance"),
        ("cuad", "law_legal"),
        ("expertqa", "general"),
    ]:
        try:
            ds = load_dataset("rungalileo/ragbench", subset, split="train")

            for row in ds:
                docs = row.get("documents") or []

                if isinstance(docs, str):
                    docs = [docs]

                for doc in docs:
                    text = str(doc).strip()

                    if len(text) < 40 or _text_is_question(text):
                        continue

                    passage = text[:1200]

                    if passage in seen:
                        continue

                    seen.add(passage)

                    passages.append({
                        "id": len(passages),
                        "text": passage,
                        "category": category,
                    })

        except Exception as e:
            print(f"   RAGBench/{subset} failed: {e}")

    if not passages:
        return _build_wikipedia_kb(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(kb_path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"RAGBench KB built: {len(passages)} passages → {kb_path}")

    return kb_path


def _build_multihop_kb(output_dir: str, max_docs: int = 5000) -> str:
    kb_path = os.path.join(output_dir, "knowledge_base_multihoprag.json")

    if os.path.exists(kb_path):
        try:
            with open(kb_path) as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f"📚 Reusing MultiHopRAG KB: {kb_path} ({len(existing)} passages)")
                return kb_path
        except Exception:
            pass

    print(" Building MultiHopRAG KB...")

    from datasets import load_dataset

    passages = []
    seen = set()

    for hf_id, cfg in [("yixuantt/MultiHopRAG", None), ("MultiHop-RAG", None)]:
        for split in ("test", "train"):
            try:
                ds = (
                    load_dataset(hf_id, split=split)
                    if cfg is None
                    else load_dataset(hf_id, cfg, split=split)
                )

                context_cols = [
                    c for c in ds.column_names
                    if c in (
                        "context",
                        "contexts",
                        "passage",
                        "passages",
                        "evidence",
                        "documents",
                    )
                ]

                if not context_cols:
                    continue

                col = context_cols[0]

                for row in ds:
                    ctx = row.get(col, "")

                    items = (
                        ctx
                        if isinstance(ctx, list)
                        else ([ctx] if isinstance(ctx, str) else [])
                    )

                    for item in items:
                        text = str(item).strip()

                        if len(text) < 40 or _text_is_question(text):
                            continue

                        passage = text[:1200]

                        if passage in seen:
                            continue

                        seen.add(passage)

                        passages.append({
                            "id": len(passages),
                            "text": passage,
                            "category": "multihop",
                        })

                    if len(passages) >= max_docs:
                        break

                if passages:
                    break

            except Exception:
                continue

        if passages:
            break

    if not passages:
        return _build_wikipedia_kb(output_dir, max_docs=max_docs)

    os.makedirs(output_dir, exist_ok=True)

    with open(kb_path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"MultiHopRAG KB built: {len(passages)} passages → {kb_path}")

    return kb_path


def _build_dataset_kb(dataset_name, texts, output_dir, max_docs=5000, passage_words=80):
    import re as _re

    kb_path = os.path.join(output_dir, f"knowledge_base_{dataset_name.lower()}.json")

    if os.path.exists(kb_path):
        try:
            with open(kb_path) as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f"📚 Reusing existing KB: {kb_path} ({len(existing)} passages)")
                return kb_path
        except Exception:
            pass

    print(f"📚 Building dataset KB for {dataset_name}...")

    passages = []
    seen = set()

    for text in texts:
        if not text or len(text.strip()) < 20 or _text_is_question(text.strip()):
            continue

        text = _re.sub(r"\s+", " ", text).strip()
        words = text.split()

        for i in range(0, len(words), passage_words):
            chunk = " ".join(words[i:i + passage_words])

            if len(chunk.strip()) < 30 or chunk in seen:
                continue

            seen.add(chunk)

            passages.append({
                "id": len(passages),
                "text": chunk,
                "category": dataset_name.lower(),
            })

            if len(passages) >= max_docs:
                break

        if len(passages) >= max_docs:
            break

    os.makedirs(output_dir, exist_ok=True)

    with open(kb_path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"KB built: {len(passages)} passages → {kb_path}")

    return kb_path


def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""


def _extract_metric_lines(log_text):
    if not log_text:
        return ""

    lines = log_text.splitlines(True)
    keep = []
    in_block = False

    for ln in lines:
        if "===" in ln:
            in_block = True

        if in_block:
            keep.append(ln)

    return "".join(keep) if keep else "".join(lines[-200:])


def _print_rag_results_summary(experiment_dir):
    online_dir = os.path.join(experiment_dir, "online_learning")

    print("\n" + "=" * 80)
    print("RAG ANALYSIS RESULTS")
    print("=" * 80)

    summary_file = os.path.join(online_dir, "analysis_summary.json")

    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)

            meta = summary.get("experiment_metadata", {})

            print(f"\nTotal queries : {meta.get('total_queries', 0)}")
            print(f"Primary (80%) : {meta.get('primary_queries', 0)}")
            print(f"Holdout (20%) : {meta.get('holdout_queries', 0)}")

            retr = summary.get("retrieval_statistics", {})

            print(f"\nBaseline retrievals : {retr.get('baseline_retrievals', 0)}")
            print(f"Actual retrievals   : {retr.get('actual_retrievals', 0)}")
            print(f"Retrievals saved    : {retr.get('retrievals_saved', 0)}")

        except Exception as e:
            print(f"Could not read analysis summary: {e}")

    print("\n" + "=" * 80)
    print(f"Full results: {online_dir}")
    print("=" * 80 + "\n")


def _extract_mind_gold_answers(texts):
    queries = []
    gold_answers = []

    for text in texts:
        sep = text.find(". ")

        if sep != -1 and sep < len(text) - 2:
            queries.append(text[:sep])
            gold_answers.append(text[sep + 2:])
        else:
            queries.append(text)
            gold_answers.append("")

    return queries, gold_answers


def _resolve_kb_for_dataset(dataset_name, dataset_instance, output_dir):
   
    if args.knowledge_base and os.path.exists(args.knowledge_base):
        print(f"📚 Using user-provided knowledge base: {args.knowledge_base}")
        return args.knowledge_base

    dn = dataset_name

    if dn == "KILTStreamSim":
        return _build_kilt_kb(output_dir, max_docs=8000)

    if dn == "RAGBenchStreamSim":
        return _build_ragbench_kb(dataset_instance, output_dir)

    if dn == "PopQAStreamSim":
        return _build_wikipedia_kb(output_dir, max_docs=5000)

    if dn == "MultiHopRAGStreamSim":
        return _build_multihop_kb(output_dir, max_docs=5000)

    if dn == "StateCacheQAStreamSim":
        kb_path = os.path.join(root, "data", "knowledge_base_state_cache_qa.json")
        if os.path.exists(kb_path):
            print(f"📚 Using StateCacheQA synthetic KB: {kb_path}")
            return kb_path
        raise FileNotFoundError(
            f"StateCacheQA KB not found: {kb_path}. "
            "Place knowledge_base_state_cache_qa.json inside STADmain/data/."
        )

    raw_texts = []

    try:
        raw_texts = dataset_instance.get_original_texts() or []
    except Exception:
        pass

    if not raw_texts:
        return os.path.join(output_dir, f"knowledge_base_{dn.lower()}.json")

    return _build_dataset_kb(
        dn,
        raw_texts,
        output_dir,
        max_docs=5000,
        passage_words=80,
    )



def predict_func(
    test_set,
    class_labels,
    drift_labels,
    experiment_dir,
    hidden_dim,
    batch_size,
    dropout_rate,
    window_size,
    steps,
    epochs,
    early_stopping_patience,
    learning_rate,
    weight_decay,
    input_dim,
    l_hist_size,
    l_new_size,
    ks_alpha,
    ks_hist_min_size,
    kld_epsilon,
    drift_mode,
    lambda_mmd,
    state_sim_threshold,
    state_update_alpha,
    force_new_state_on_alarm,
    log_every,
    event_tolerance_k,
    event_window_W,
    event_fp_penalty,
    pointwise_tolerance_k,
    alarm_cooldown_k,
    state_penalty_lambda,
    normalize_for_detectors,
    enable_rag,
    knowledge_base_path,
    num_test_queries,
    original_texts=None,
    ground_truth_answers=None,
    dataset_name=None,
    state_logic="cosine",
    max_state_num=50,
):
    online_exp_dir = os.path.join(experiment_dir, "online_learning")
    os.makedirs(online_exp_dir, exist_ok=True)

    kb_path = knowledge_base_path

    if enable_rag and not (kb_path and os.path.exists(kb_path)):
        generic_kb = os.path.join(root, "data", "knowledge_base_wikipedia.json")

        if os.path.exists(generic_kb):
            kb_path = generic_kb
            print(f"📚 Fallback to Wikipedia KB: {kb_path}")
        else:
            print("No KB found — retriever will use dummy KB")
            kb_path = None

    pred = Predict(
        experiment_dir=online_exp_dir,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        window_size=window_size,
        steps=steps,
        input_dim=input_dim,
        l_hist_size=l_hist_size,
        l_new_size=l_new_size,
        ks_alpha=ks_alpha,
        ks_hist_min_size=ks_hist_min_size,
        gpu=gpu,
        drift_mode=drift_mode,
        lambda_mmd=lambda_mmd,
        state_sim_threshold=state_sim_threshold,
        state_update_alpha=state_update_alpha,
        force_new_state_on_alarm=force_new_state_on_alarm,
        log_every=log_every,
        event_tolerance_k=event_tolerance_k,
        event_window_W=event_window_W,
        event_fp_penalty=event_fp_penalty,
        pointwise_tolerance_k=pointwise_tolerance_k,
        alarm_cooldown_k=alarm_cooldown_k,
        state_penalty_lambda=state_penalty_lambda,
        normalize_for_detectors=normalize_for_detectors,
        enable_rag=enable_rag,
        knowledge_base_path=kb_path,
        num_test_queries=num_test_queries,
        state_logic=str(state_logic),
        max_state_num=int(max_state_num),
    )

    if enable_rag and original_texts is not None:
        pred.set_original_texts(original_texts)

    if enable_rag and ground_truth_answers is not None:
        pred.set_ground_truth_answers(ground_truth_answers)

    msg = pred.predict(test_set, class_labels, drift_labels, kld_epsilon)

    return msg


for dataset_name in datasets:
    now = datetime.now()
    run_id = re.sub(r"[-: ]", "_", str(now).split(".")[0])

    cfg_key = config_key_for_dataset(dataset_name)

    if cfg_key not in config:
        if dataset_name in _RAG_BENCHMARK_DATASETS:
            logger.warning(f"'{cfg_key}' not in config.json — using built-in defaults.")
            config_exp = _RAG_DATASET_DEFAULT_CONFIG.copy()
        else:
            raise KeyError(f"Dataset key '{cfg_key}' not found in config.json")
    else:
        config_exp = config[cfg_key]

    win_lens = ensure_list(config_exp.get("win_len", 25))
    steps_list = ensure_list(config_exp.get("steps", 25))

    hds = ensure_list(config_exp.get("hidden_size", 128))
    epochs_list = ensure_list(config_exp.get("epochs", 10))
    lrs = ensure_list(config_exp.get("lr", 1e-4))
    dropouts = ensure_list(config_exp.get("dropout", 0.2))
    batch_sizes = ensure_list(config_exp.get("batch_size", 32))
    weight_decays = ensure_list(config_exp.get("weight_decay", 1e-5))
    input_dims = ensure_list(config_exp.get("dim", 384))
    early_stopping_patiences = ensure_list(config_exp.get("early_stopping_patience", 5))

 
    ks_alphas_default = ensure_list(config_exp.get("ks_alpha", 0.05))
    mmd_alphas_config = ensure_list(config_exp.get("mmd_alpha", ks_alphas_default))
    energy_alphas_config = ensure_list(config_exp.get("energy_alpha", ks_alphas_default))

    ks_hist_min_sizes = ensure_list(config_exp.get("ks_hist_min_size", 50))
    kld_epsilons = ensure_list(config_exp.get("kld_epsilon", 1e-3))

    state_logics = ensure_list(config_exp.get("state_logic", "cosine"))
    max_state_nums = ensure_list(config_exp.get("max_state_num", 50))

    l_hist_sizes_default = ensure_list(config_exp.get("l_hist_size", 100))
    l_new_sizes_default = ensure_list(config_exp.get("l_new_size", 10))

    mmd_l_hist_sizes = ensure_list(config_exp.get("mmd_l_hist_size", l_hist_sizes_default))
    mmd_l_new_sizes = ensure_list(config_exp.get("mmd_l_new_size", l_new_sizes_default))
    mmd_hist_min_sizes = ensure_list(config_exp.get("mmd_hist_min_size", [50]))
    lambda_mmd_grid = ensure_list(config_exp.get("lambda_mmd", [lambda_mmd_cli]))

    energy_l_hist_sizes = ensure_list(config_exp.get("energy_l_hist_size", mmd_l_hist_sizes))
    energy_l_new_sizes = ensure_list(config_exp.get("energy_l_new_size", mmd_l_new_sizes))
    energy_hist_min_sizes = ensure_list(config_exp.get("energy_hist_min_size", mmd_hist_min_sizes))
    lambda_energy_grid = ensure_list(config_exp.get("lambda_energy", lambda_mmd_grid))

    homo_thresholds = ensure_list(config_exp.get("homo_threshold", 0.8))
    state_sim_thresholds = ensure_list(config_exp.get("state_sim_threshold", 0.90))
    state_update_alphas = ensure_list(config_exp.get("state_update_alpha", 0.05))
    force_new_state_on_alarms = ensure_list(config_exp.get("force_new_state_on_alarm", False))
    log_every_list = ensure_list(config_exp.get("log_every", 10))
    normalize_for_detectors_list = ensure_list(config_exp.get("normalize_for_detectors", False))

    event_tolerance_ks = ensure_list(config_exp.get("event_tolerance_k", 3))
    event_window_Ws = ensure_list(config_exp.get("event_window_W", 10))
    event_fp_penalties = ensure_list(config_exp.get("event_fp_penalty", 0.11))
    pointwise_tolerance_ks = ensure_list(config_exp.get("pointwise_tolerance_k", 1))
    alarm_cooldown_ks = ensure_list(config_exp.get("alarm_cooldown_k", 0))
    state_penalty_lambdas = ensure_list(config_exp.get("state_penalty_lambda", 1.0))

    mind_articles_per_category_list = ensure_list(config_exp.get("articles_per_category", [500]))
    mind_num_cycles_list = ensure_list(config_exp.get("num_cycles", [3]))
    beir_num_cycles_list = ensure_list(config_exp.get("num_cycles", [3]))

    rag_cap_list = ensure_list(config_exp.get("cap", [15000]))

    mode = drift_mode

    if mode in ("ks", "ks_alibi"):
        l_hist_sizes = l_hist_sizes_default
        l_new_sizes = l_new_sizes_default
        hist_min_sizes = ks_hist_min_sizes
        alpha_grid = ks_alphas_default
        lambda_grid = [lambda_mmd_cli]

    elif mode == "mmd":
        l_hist_sizes = mmd_l_hist_sizes
        l_new_sizes = mmd_l_new_sizes
        hist_min_sizes = mmd_hist_min_sizes
        alpha_grid = mmd_alphas_config
        lambda_grid = lambda_mmd_grid

    elif mode == "energy":
        l_hist_sizes = energy_l_hist_sizes
        l_new_sizes = energy_l_new_sizes
        hist_min_sizes = energy_hist_min_sizes
        alpha_grid = energy_alphas_config
        lambda_grid = lambda_energy_grid

    else:
        l_hist_sizes = mmd_l_hist_sizes
        l_new_sizes = mmd_l_new_sizes
        hist_min_sizes = mmd_hist_min_sizes
        alpha_grid = mmd_alphas_config
        lambda_grid = lambda_mmd_grid

    
    run_configs = []
    cosine_added = False

    for sl in state_logics:
        sl = str(sl).lower().strip()

        if sl == "cosine":
            if cosine_added:
                continue

            cosine_added = True

            out_dir = (
                f"{root}/{result_dir}/{mode}/cosine"
                f"/{dataset_name}"
            )

            run_configs.append(("cosine", kld_epsilons[0], out_dir))

        else:
            for eps in kld_epsilons:
                out_dir = (
                    f"{root}/{result_dir}/{mode}/kld"
                    f"/{dataset_name}"
                )

                run_configs.append(("kld", eps, out_dir))

    logger.info(f"[run.py] Total run configurations: {len(run_configs)}")

    if len(run_configs) == 0:
        raise RuntimeError(
            f"No run configurations were created for dataset={dataset_name}. "
            f"Check state_logic in configuration/config.json. It must contain "
            f"at least one of: cosine, kld."
        )

    for sl, eps, od in run_configs:
        logger.info(f"   {sl:8s}  eps={eps}  → {od}")


    base_output_dir = None
    cleared_output_dirs = set()

    for state_logic_run, kld_epsilon, base_output_dir in run_configs:
        # Clear each base output directory only once per run.py execution.
        # This is important because multiple KLD epsilon values can share:
        #   output/{mode}/kld/{dataset}
        if base_output_dir not in cleared_output_dirs:
            if os.path.exists(base_output_dir):
                print(f"[run.py] Removing old output directory: {base_output_dir}")
                shutil.rmtree(base_output_dir)
            cleared_output_dirs.add(base_output_dir)

        os.makedirs(base_output_dir, exist_ok=True)

        print(f"\n{'#' * 80}")
        print(
            f"# METHOD : {state_logic_run.upper()}"
            + (f"  (epsilon={kld_epsilon})" if state_logic_run == "kld" else "")
        )
        print(f"# OUTPUT : {base_output_dir}")
        print(f"{'#' * 80}")

        
        if dataset_name == "MINDStreamSim":
            dataset_param_combos = [
                (
                    f"apc{apc}_nc{nc}",
                    {
                        "articles_per_category": int(apc),
                        "num_cycles": int(nc),
                    },
                )
                for apc, nc in itertools.product(
                    mind_articles_per_category_list,
                    mind_num_cycles_list,
                )
            ]

        elif dataset_name == "BEIRStreamSim":
            dataset_param_combos = [
                (
                    f"nc{nc}",
                    {
                        "num_cycles": int(nc),
                    },
                )
                for nc in beir_num_cycles_list
            ]

        elif dataset_name in (
            "KILTStreamSim",
            "MultiHopRAGStreamSim",
            "PopQAStreamSim",
            "RAGBenchStreamSim",
            "StateCacheQAStreamSim",
        ):
            dataset_param_combos = [
                (
                    f"max{cap}",
                    {
                        "max_samples": int(cap),
                    },
                )
                for cap in rag_cap_list
            ]

        else:
            dataset_param_combos = [("", {})]

        for win_len in win_lens:
            for extra_label, dataset_kwargs in dataset_param_combos:
                print(f"\n{'=' * 80}")
                print(f"Loading dataset: {dataset_name}")

                if dataset_kwargs:
                    print(f"   Params: {dataset_kwargs}")

                print(f"{'=' * 80}")

                # No SavedDataset and no rag_datasets.py anymore.
                dataset_instance = dataset_obj[dataset_name](
                    win_len=int(win_len),
                    **dataset_kwargs,
                )

                x_train, y_train, x_test, y_test, y_test_binary = dataset_instance.data()

                print(f"Dataset loaded: {len(y_test)} test samples")

                original_texts = None
                ground_truth_answers = None
                dataset_kb_path = None

                if args.enable_rag:
                    dataset_kb_path = _resolve_kb_for_dataset(
                        dataset_name=dataset_name,
                        dataset_instance=dataset_instance,
                        output_dir=str(root / "data"),
                    )

                    print(f"KB path: {dataset_kb_path}")

                    print("\nExtracting original texts for RAG...")

                    try:
                        raw_texts = dataset_instance.get_original_texts()

                        if raw_texts and len(raw_texts) > 0:
                            if dataset_name == "MINDStreamSim":
                                original_texts, ground_truth_answers = _extract_mind_gold_answers(raw_texts)

                                print(
                                    f"MIND: {len(original_texts)} queries + "
                                    f"{sum(1 for a in ground_truth_answers if a)} gold answers"
                                )

                            elif dataset_name in _RAG_BENCHMARK_DATASETS:
                                original_texts = raw_texts

                                if hasattr(dataset_instance, "get_ground_truth_answers"):
                                    ground_truth_answers = dataset_instance.get_ground_truth_answers()
                                    answered = sum(1 for a in ground_truth_answers if a)

                                    print(
                                        f"Extracted {len(original_texts):,} queries + "
                                        f"{answered:,} gold answers"
                                    )
                                else:
                                    print(f"Extracted {len(original_texts):,} queries (no gold answers)")

                            else:
                                original_texts = raw_texts
                                print(f"Extracted {len(original_texts)} texts")

                        else:
                            print("No original texts — using generic queries")
                            original_texts = [f"Sample query {i}" for i in range(len(y_test))]

                    except Exception as e:
                        print(f"Error extracting texts: {e}")

                        import traceback
                        traceback.print_exc()

                        original_texts = [f"Sample query {i}" for i in range(len(y_test))]

                # ── Inner parameter grid ──
                for (
                    steps,
                    hd,
                    epoch,
                    l_hist_size,
                    lnew,
                    min_hist,
                    alpha,
                    lr,
                    dr,
                    bs,
                    wd,
                    in_dim,
                    lam,
                    homo_th,
                    sim_th,
                    upd_a,
                    force_new_state,
                    log_every,
                    esp,
                    seed,
                    ev_k,
                    ev_W,
                    ev_fp,
                    point_k,
                    cooldown_k,
                    st_lam,
                    norm_det,
                    max_state_num,
                ) in itertools.product(
                    steps_list,
                    hds,
                    epochs_list,
                    l_hist_sizes,
                    l_new_sizes,
                    hist_min_sizes,
                    alpha_grid,
                    lrs,
                    dropouts,
                    batch_sizes,
                    weight_decays,
                    input_dims,
                    lambda_grid,
                    homo_thresholds,
                    state_sim_thresholds,
                    state_update_alphas,
                    force_new_state_on_alarms,
                    log_every_list,
                    early_stopping_patiences,
                    seeds,
                    event_tolerance_ks,
                    event_window_Ws,
                    event_fp_penalties,
                    pointwise_tolerance_ks,
                    alarm_cooldown_ks,
                    state_penalty_lambdas,
                    normalize_for_detectors_list,
                    max_state_nums,
                ):
                    set_all_seeds(int(seed))

                    state_logic = state_logic_run

                    print(f"\n{'=' * 80}")
                    print("STARTING EXPERIMENT")
                    print(f"{'=' * 80}")
                    print(f"Dataset    : {dataset_name}")
                    print(f"Window     : {win_len}  Step: {steps}  Mode: {mode}")
                    print(f"Lambda     : {lam}  Sim threshold: {sim_th}")
                    print(
                        f"State logic: {state_logic}"
                        + (
                            f"  (epsilon={kld_epsilon})"
                            if state_logic == "kld"
                            else f"  (threshold={sim_th})"
                        )
                    )
                    print(f"Norm det   : {norm_det}  RAG: {args.enable_rag}  Seed: {seed}")

                    identifier = (
                        f"win{win_len}_step{steps}_hd{hd}_epoch{epoch}_"
                        f"lhist{l_hist_size}_lnew{lnew}_minHist{min_hist}_"
                        f"alpha{alpha}_lr{lr}_drop{dr}_bs{bs}_wd{wd}_dim{in_dim}_"
                        f"mode{mode}_lam{lam}_simth{sim_th}_normdet{int(norm_det)}_"
                        f"state{state_logic}"
                        + (f"_eps{kld_epsilon}" if state_logic == "kld" else "")
                        + f"_rag{int(args.enable_rag)}_nq{args.num_test_queries}"
                        + (f"_{extra_label}" if extra_label else "")
                        + f"_seed{seed}_{run_id}"
                    )

                    exp_root = os.path.join(base_output_dir, f"minHist_{min_hist}")
                    experiment_dir = os.path.join(exp_root, identifier)

                    if os.path.exists(experiment_dir):
                        print(f"Experiment dir exists, skipping: {experiment_dir}")
                        continue

                    os.makedirs(experiment_dir, exist_ok=True)

                    meta = {
                        "dataset": dataset_name,
                        "config_key": cfg_key,
                        "run_id": run_id,
                        "seed": int(seed),
                        "mode": mode,
                        "state_logic": state_logic,
                        "kld_epsilon": float(kld_epsilon) if state_logic == "kld" else None,
                        "analysis_type": "simplified_3way" if args.enable_rag else "drift_only",
                        "rag_enabled": args.enable_rag,
                        "kb_path": str(dataset_kb_path or ""),
                        "original_texts_count": len(original_texts) if original_texts else 0,
                        "gold_answers_count": (
                            sum(1 for a in ground_truth_answers if a)
                            if ground_truth_answers
                            else 0
                        ),
                        "params": {
                            "win_len": int(win_len),
                            "steps": int(steps),
                            "l_hist_size": int(l_hist_size),
                            "l_new_size": int(lnew),
                            "lambda": float(lam),
                            "state_sim_threshold": float(sim_th),
                            "state_logic": str(state_logic),
                            "kld_epsilon": float(kld_epsilon),
                            "pointwise_tolerance_k": int(point_k),
                            "alarm_cooldown_k": int(cooldown_k),
                            **dataset_kwargs,
                        },
                    }

                    with open(os.path.join(experiment_dir, "run_meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)

                    print("\nStarting prediction...")

                    pred_start = time.time()

                    msg = predict_func(
                        x_test,
                        y_test,
                        y_test_binary,
                        experiment_dir,
                        hidden_dim=int(hd),
                        batch_size=int(bs),
                        dropout_rate=float(dr),
                        window_size=int(win_len),
                        steps=int(steps),
                        epochs=int(epoch),
                        early_stopping_patience=int(esp),
                        learning_rate=float(lr),
                        weight_decay=float(wd),
                        input_dim=int(in_dim),
                        l_hist_size=int(l_hist_size),
                        l_new_size=int(lnew),
                        ks_alpha=float(alpha),
                        ks_hist_min_size=int(min_hist),
                        kld_epsilon=float(kld_epsilon),
                        drift_mode=mode,
                        lambda_mmd=float(lam),
                        state_sim_threshold=float(sim_th),
                        state_update_alpha=float(upd_a),
                        force_new_state_on_alarm=bool(force_new_state),
                        log_every=int(log_every),
                        event_tolerance_k=int(ev_k),
                        event_window_W=int(ev_W),
                        event_fp_penalty=float(ev_fp),
                        pointwise_tolerance_k=int(point_k),
                        alarm_cooldown_k=int(cooldown_k),
                        state_penalty_lambda=float(st_lam),
                        normalize_for_detectors=bool(norm_det),
                        enable_rag=args.enable_rag,
                        knowledge_base_path=dataset_kb_path or args.knowledge_base,
                        num_test_queries=args.num_test_queries,
                        original_texts=original_texts,
                        ground_truth_answers=ground_truth_answers,
                        dataset_name=dataset_name,
                        state_logic=str(state_logic),
                        max_state_num=int(max_state_num),
                    )

                    pred_time = round(time.time() - pred_start, 3)

                    online_exp_dir = os.path.join(experiment_dir, "online_learning")
                    pred_log_text = _read_text(os.path.join(online_exp_dir, "log"))
                    metrics_excerpt = _extract_metric_lines(pred_log_text)

                    with open(os.path.join(online_exp_dir, "time_log.txt"), "w") as f:
                        f.write(
                            f"Method: {state_logic}"
                            + (
                                f" (epsilon={kld_epsilon})\n"
                                if state_logic == "kld"
                                else "\n"
                            )
                        )

                        if args.enable_rag:
                            f.write("RAG pipeline enabled (3-way comparison)\n")
                            f.write(f"  KB path: {dataset_kb_path}\n")
                            f.write(f"  Original texts: {len(original_texts) if original_texts else 0}\n")
                            f.write(
                                f"  Gold answers: "
                                f"{sum(1 for a in ground_truth_answers if a) if ground_truth_answers else 0}\n"
                            )

                        f.write(f"Processing time (sec): {pred_time}\n")
                        f.write(f"Result: {str(msg).strip()}\n\n")
                        f.write("=== Metrics ===\n")
                        f.write(metrics_excerpt if metrics_excerpt else "(no log)\n")

                    print(f"\nCompleted in {pred_time}s")

                    if args.enable_rag:
                        _print_rag_results_summary(experiment_dir)

                    
                    try:
                        import importlib.util

                        _script = os.path.join(os.path.dirname(__file__), "plot_thesis.py")

                        if os.path.exists(_script):
                            spec = importlib.util.spec_from_file_location("plot_thesis", _script)
                            _pt = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(_pt)

                            _plot_dir = online_exp_dir

                            print(f"\nGenerating thesis plots → {_plot_dir}")

                            _pt.plot1_drift_score(
                                _pt.load_scores(_plot_dir),
                                os.path.join(_plot_dir, "plot1_drift_score.png"),
                            )

                            _pt.plot2_state_timeline(
                                _pt.load_states(_plot_dir),
                                os.path.join(_plot_dir, "plot2_state_timeline.png"),
                            )

                            _sr = _pt.load_state_repr(_plot_dir)

                            _pt.plot3_tsne(
                                _pt.load_states(_plot_dir),
                                _sr,
                                os.path.join(_plot_dir, "plot3_tsne.png"),
                            )

                            _pt.plot4_rouge(
                                _pt.load_holdout(_plot_dir),
                                os.path.join(_plot_dir, "plot4_rouge_comparison.png"),
                            )

                            _pt.plot5_quality_cost(
                                _pt.load_holdout(_plot_dir),
                                _pt.load_states(_plot_dir),
                                os.path.join(_plot_dir, "plot5_quality_vs_cost.png"),
                            )

                            _pt.plot_combined(_plot_dir)

                            print(f"All plots saved to {_plot_dir}")

                        else:
                            print("plot_thesis.py not found — skipping plots")

                    except Exception as _pe:
                        print(f"Plotting failed (non-fatal): {_pe}")

                        import traceback
                        traceback.print_exc()


print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 80)

if base_output_dir:
    print(f"Results root: {Path(base_output_dir).parent.parent}")

print("=" * 80 + "\n")
