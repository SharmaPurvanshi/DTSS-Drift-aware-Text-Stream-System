from typing import List, Dict, Optional, Tuple
import numpy as np
import csv
import os
import re
import string
from collections import Counter


# ===========================================================================
# AnswerComparator
# ===========================================================================

class AnswerComparator:
   
    def __init__(self, rouge_threshold: float = 0.3, strict: bool = True):
        
        self.rouge_threshold = rouge_threshold
        self.strict = bool(strict)

        try:
            from rouge_score import rouge_scorer as rs
            self._scorer = rs.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self._rouge_available = True
            print(f" AnswerComparator initialized "
                  f"(ROUGE scoring, threshold={rouge_threshold})")
        except ImportError as e:
            msg = "rouge-score is required for RAG evaluation. Install it with: pip install rouge-score"
            if self.strict:
                raise ImportError(msg) from e
            print(f"  {msg}")
            self._scorer = None
            self._rouge_available = False

    
    @staticmethod
    def normalize_answer(text: str) -> str:
       
        text = str(text or "").lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

    @classmethod
    def exact_match(cls, prediction: str, gold: str) -> int:
        if not str(gold or "").strip():
            return 0
        return int(cls.normalize_answer(prediction) == cls.normalize_answer(gold))

    @classmethod
    def contains_gold(cls, prediction: str, gold: str) -> int:
        gold_n = cls.normalize_answer(gold)
        pred_n = cls.normalize_answer(prediction)
        if not gold_n or not pred_n:
            return 0
        return int(gold_n in pred_n)

    @classmethod
    def token_f1(cls, prediction: str, gold: str) -> float:
        gold_toks = cls.normalize_answer(gold).split()
        pred_toks = cls.normalize_answer(prediction).split()
        if not gold_toks or not pred_toks:
            return 0.0
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        return float(2 * precision * recall / (precision + recall))

    def _qa_metrics(self, prediction: str, gold: str) -> Dict:
        return {
            "exact_match": self.exact_match(prediction, gold),
            "token_f1": self.token_f1(prediction, gold),
            "contains_gold": self.contains_gold(prediction, gold),
        }

    
    def _rouge(self, hypothesis: str, reference: str) -> Dict:
        """ROUGE-1/2/L F1 for one hypothesis/reference pair."""
        if (not self._rouge_available
                or not hypothesis.strip()
                or not reference.strip()):
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        s = self._scorer.score(reference, hypothesis)
        return {
            'rouge1': float(s['rouge1'].fmeasure),
            'rouge2': float(s['rouge2'].fmeasure),
            'rougeL': float(s['rougeL'].fmeasure),
        }

    def _correct(self, rouge_scores: Dict, metric: str = 'rougeL') -> int:
        """Return 1 if ROUGE score >= threshold, else 0."""
        return int(rouge_scores.get(metric, 0.0) >= self.rouge_threshold)

    
    def compare_answers(
        self,
        query:             str,
        baseline_answer:   str = "",
        cached_rag_answer: Optional[str] = None,
        fresh_rag_answer:  Optional[str] = None,
        gold_answer:       str = "",
        **legacy_kwargs,
    ) -> Dict:
       
        
        if cached_rag_answer is None and "rag_answer" in legacy_kwargs:
            cached_rag_answer = legacy_kwargs.get("rag_answer", "")
        if fresh_rag_answer is None and "rag_answer" in legacy_kwargs:
            fresh_rag_answer = legacy_kwargs.get("rag_answer", "")

        cached_rag_answer = cached_rag_answer or ""
        fresh_rag_answer = fresh_rag_answer or ""
        baseline_answer = baseline_answer or ""
        gold_answer = gold_answer or ""

        
        has_gold  = bool(gold_answer.strip())
        reference = gold_answer if has_gold else fresh_rag_answer
        ref_label = "gold"      if has_gold else "fresh_rag"

        baseline_rouge   = self._rouge(baseline_answer,   reference)
        cached_rag_rouge = self._rouge(cached_rag_answer, reference)
        fresh_rag_rouge  = self._rouge(fresh_rag_answer,  reference)

        
        baseline_qa   = self._qa_metrics(baseline_answer, gold_answer) if has_gold else {"exact_match": 0, "token_f1": 0.0, "contains_gold": 0}
        cached_rag_qa = self._qa_metrics(cached_rag_answer, gold_answer) if has_gold else {"exact_match": 0, "token_f1": 0.0, "contains_gold": 0}
        fresh_rag_qa  = self._qa_metrics(fresh_rag_answer, gold_answer) if has_gold else {"exact_match": 0, "token_f1": 0.0, "contains_gold": 0}

        return {
            'query':             query,
            'reference_used':    ref_label,
            'has_gold':          has_gold,

            
            'baseline_answer':   baseline_answer,
            'cached_rag_answer': cached_rag_answer,
            'fresh_rag_answer':  fresh_rag_answer,
            'gold_answer':       gold_answer,

         
            'baseline_rouge':    baseline_rouge,
            'cached_rag_rouge':  cached_rag_rouge,
            'fresh_rag_rouge':   fresh_rag_rouge,

       
            'baseline_qa':       baseline_qa,
            'cached_rag_qa':     cached_rag_qa,
            'fresh_rag_qa':      fresh_rag_qa,

           
            'baseline_correct':   self._correct(baseline_rouge),
            'cached_rag_correct': self._correct(cached_rag_rouge),
            'fresh_rag_correct':  self._correct(fresh_rag_rouge),

            'baseline_correct_r1':   self._correct(baseline_rouge,   'rouge1'),
            'cached_rag_correct_r1': self._correct(cached_rag_rouge, 'rouge1'),
            'fresh_rag_correct_r1':  self._correct(fresh_rag_rouge,  'rouge1'),

           
            'rouge1_gap': cached_rag_rouge['rouge1'] - baseline_rouge['rouge1'],
            'rouge2_gap': cached_rag_rouge['rouge2'] - baseline_rouge['rouge2'],
            'rougeL_gap': cached_rag_rouge['rougeL'] - baseline_rouge['rougeL'],
        }

   
    def aggregate_comparison(self, comparisons: List[Dict]) -> Dict:
        
        if not comparisons:
            return {}

        n           = len(comparisons)
        gold_rows   = [c for c in comparisons if c.get('has_gold')]
        n_gold      = len(gold_rows)
        ref_str     = "gold answer" if n_gold > 0 else "fresh_rag (proxy)"

        def mean_rouge(key, r):
            vals = [c[key][r] for c in comparisons if key in c]
            return float(np.mean(vals)) if vals else 0.0

        def mean_field(f):
            vals = [c[f] for c in comparisons if f in c]
            return float(np.mean(vals)) if vals else 0.0

      
        br = {m: mean_rouge('baseline_rouge',   m) for m in ('rouge1','rouge2','rougeL')}
        cr = {m: mean_rouge('cached_rag_rouge', m) for m in ('rouge1','rouge2','rougeL')}
        fr = {m: mean_rouge('fresh_rag_rouge',  m) for m in ('rouge1','rouge2','rougeL')}

    
        def mean_qa(key, field):
            vals = [c.get(key, {}).get(field, 0.0) for c in gold_rows]
            return float(np.mean(vals)) if vals else 0.0

        bqa = {m: mean_qa('baseline_qa',   m) for m in ('exact_match', 'token_f1', 'contains_gold')}
        cqa = {m: mean_qa('cached_rag_qa', m) for m in ('exact_match', 'token_f1', 'contains_gold')}
        fqa = {m: mean_qa('fresh_rag_qa',  m) for m in ('exact_match', 'token_f1', 'contains_gold')}

     
        def accuracy(correct_key):
            vals = [c.get(correct_key, 0) for c in comparisons]
            return float(np.mean(vals)) * 100  # as percentage

        baseline_acc   = accuracy('baseline_correct')
        cached_rag_acc = accuracy('cached_rag_correct')
        fresh_rag_acc  = accuracy('fresh_rag_correct')

   
        baseline_acc_r1   = accuracy('baseline_correct_r1')
        cached_rag_acc_r1 = accuracy('cached_rag_correct_r1')
        fresh_rag_acc_r1  = accuracy('fresh_rag_correct_r1')

        def win_rate(key_a, key_b, metric):
            
            wins = sum(
                1 for c in comparisons
                if c.get(key_a, {}).get(metric, 0)
                   > c.get(key_b, {}).get(metric, 0)
            )
            return float(wins / n * 100)

        return {
            'num_queries':        n,
            'num_gold_queries':   n_gold,
            'reference_used':     ref_str,
            'rouge_threshold':    self.rouge_threshold,

        
            'baseline_rouge':    br,
            'cached_rag_rouge':  cr,
            'fresh_rag_rouge':   fr,

            
            'accuracy': {
                'threshold':              self.rouge_threshold,
                'baseline_rougeL':        round(baseline_acc,   2),
                'cached_rag_rougeL':      round(cached_rag_acc, 2),
                'fresh_rag_rougeL':       round(fresh_rag_acc,  2),
                'baseline_rouge1':        round(baseline_acc_r1,   2),
                'cached_rag_rouge1':      round(cached_rag_acc_r1, 2),
                'fresh_rag_rouge1':       round(fresh_rag_acc_r1,  2),
                
                'gain_cached_over_baseline_rougeL': round(
                    cached_rag_acc - baseline_acc, 2),
                
                'retention_cached_vs_fresh_rougeL': round(
                    cached_rag_acc / max(fresh_rag_acc, 1e-6) * 100, 1),
            },

            
            'answer_quality': {
                'baseline': bqa,
                'cached_rag': cqa,
                'fresh_rag': fqa,
                'cached_minus_baseline': {
                    'exact_match': cqa['exact_match'] - bqa['exact_match'],
                    'token_f1': cqa['token_f1'] - bqa['token_f1'],
                    'contains_gold': cqa['contains_gold'] - bqa['contains_gold'],
                },
                'cached_retention_vs_fresh': {
                    'exact_match': cqa['exact_match'] / max(fqa['exact_match'], 1e-12),
                    'token_f1': cqa['token_f1'] / max(fqa['token_f1'], 1e-12),
                    'contains_gold': cqa['contains_gold'] / max(fqa['contains_gold'], 1e-12),
                },
            },

      
            'quality_gap': {
                'rouge1': mean_field('rouge1_gap'),
                'rouge2': mean_field('rouge2_gap'),
                'rougeL': mean_field('rougeL_gap'),
            },

            
            'win_rates': {
                'cached_beats_baseline_rouge1': win_rate(
                    'cached_rag_rouge', 'baseline_rouge', 'rouge1'),
                'cached_beats_baseline_rouge2': win_rate(
                    'cached_rag_rouge', 'baseline_rouge', 'rouge2'),
                'cached_beats_baseline_rougeL': win_rate(
                    'cached_rag_rouge', 'baseline_rouge', 'rougeL'),
                'cached_beats_fresh_rouge1': win_rate(
                    'cached_rag_rouge', 'fresh_rag_rouge', 'rouge1'),
                'cached_beats_fresh_rougeL': win_rate(
                    'cached_rag_rouge', 'fresh_rag_rouge', 'rougeL'),
              
                'cached_correct_rougeL': cached_rag_acc,
                'fresh_correct_rougeL':  fresh_rag_acc,
                'baseline_correct_rougeL': baseline_acc,
            },
        }

 
    def accuracy_at_thresholds(
        self,
        comparisons: List[Dict],
        thresholds: List[float] = None,
        metric: str = 'rougeL',
    ) -> Dict:
        
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]

        results = {}
        for thresh in thresholds:
            def acc(key):
                vals = [
                    1 if c.get(key, {}).get(metric, 0) >= thresh else 0
                    for c in comparisons
                ]
                return round(float(np.mean(vals)) * 100, 2) if vals else 0.0

            results[thresh] = {
                'baseline_acc':   acc('baseline_rouge'),
                'cached_rag_acc': acc('cached_rag_rouge'),
                'fresh_rag_acc':  acc('fresh_rag_rouge'),
            }

        return results

   
    def save_per_query_results(
        self,
        comparisons: List[Dict],
        output_path: str,
    ):
        
        if not comparisons:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "query_idx", "query", "reference_used", "has_gold",
                "gold_answer",
                "baseline_r1", "baseline_r2", "baseline_rL",
                "cached_r1",   "cached_r2",   "cached_rL",
                "fresh_r1",    "fresh_r2",    "fresh_rL",
                "baseline_em", "cached_em", "fresh_em",
                "baseline_token_f1", "cached_token_f1", "fresh_token_f1",
                "baseline_contains_gold", "cached_contains_gold", "fresh_contains_gold",
                "baseline_correct", "cached_correct", "fresh_correct",
                "rougeL_gap",
            ])
            for idx, c in enumerate(comparisons):
                br = c.get('baseline_rouge',   {})
                cr = c.get('cached_rag_rouge', {})
                fr = c.get('fresh_rag_rouge',  {})
                bq = c.get('baseline_qa', {})
                cq = c.get('cached_rag_qa', {})
                fq = c.get('fresh_rag_qa', {})
                w.writerow([
                    idx + 1,
                    str(c.get('query', ''))[:80],
                    c.get('reference_used', 'unknown'),
                    int(c.get('has_gold', False)),
                    str(c.get('gold_answer', ''))[:80],
                    round(br.get('rouge1', 0), 4),
                    round(br.get('rouge2', 0), 4),
                    round(br.get('rougeL', 0), 4),
                    round(cr.get('rouge1', 0), 4),
                    round(cr.get('rouge2', 0), 4),
                    round(cr.get('rougeL', 0), 4),
                    round(fr.get('rouge1', 0), 4),
                    round(fr.get('rouge2', 0), 4),
                    round(fr.get('rougeL', 0), 4),
                    bq.get('exact_match', 0),
                    cq.get('exact_match', 0),
                    fq.get('exact_match', 0),
                    round(bq.get('token_f1', 0.0), 4),
                    round(cq.get('token_f1', 0.0), 4),
                    round(fq.get('token_f1', 0.0), 4),
                    bq.get('contains_gold', 0),
                    cq.get('contains_gold', 0),
                    fq.get('contains_gold', 0),
                    c.get('baseline_correct',   0),
                    c.get('cached_rag_correct', 0),
                    c.get('fresh_rag_correct',  0),
                    round(c.get('rougeL_gap', 0), 4),
                ])
        print(f"    Per-query ROUGE CSV → {output_path}")

    
    def print_comparison_summary(self, aggregate: Dict):
        """Print human-readable ROUGE + accuracy summary."""
        if not aggregate:
            print(" No comparison data.")
            return

        n     = aggregate['num_queries']
        ng    = aggregate['num_gold_queries']
        ref   = aggregate['reference_used']
        thresh= aggregate['rouge_threshold']
        br    = aggregate['baseline_rouge']
        cr    = aggregate['cached_rag_rouge']
        fr    = aggregate['fresh_rag_rouge']
        qg    = aggregate['quality_gap']
        wr    = aggregate['win_rates']
        acc   = aggregate['accuracy']
        aq    = aggregate.get('answer_quality', {})

        print("\n" + "=" * 72)
        print("3-WAY HOLDOUT COMPARISON — ROUGE + ACCURACY")
        print(f"   Reference : {ref}  |  {n} queries  ({ng} with gold answers)")
        print(f"   Threshold : ROUGE-L >= {thresh} → answer counted as correct")
        print("=" * 72)

        print(f"\n{'Method':<26} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} "
              f"{'Accuracy':>10}")
        print("-" * 64)
        print(f"{'Baseline (no retrieval)':<26} "
              f"{br['rouge1']:>8.3f} {br['rouge2']:>8.3f} {br['rougeL']:>8.3f} "
              f"{acc['baseline_rougeL']:>9.1f}%")
        print(f"{'Cached RAG':<26} "
              f"{cr['rouge1']:>8.3f} {cr['rouge2']:>8.3f} {cr['rougeL']:>8.3f} "
              f"{acc['cached_rag_rougeL']:>9.1f}%")
        print(f"{'Fresh RAG (retrieval)':<26} "
              f"{fr['rouge1']:>8.3f} {fr['rouge2']:>8.3f} {fr['rougeL']:>8.3f} "
              f"{acc['fresh_rag_rougeL']:>9.1f}%")
        print("-" * 64)

        print(f"\n QUALITY GAP  (Cached RAG − Baseline)")
        print(f"   ROUGE-1:  {qg['rouge1']:+.3f}")
        print(f"   ROUGE-2:  {qg['rouge2']:+.3f}")
        print(f"   ROUGE-L:  {qg['rougeL']:+.3f}")
        print(f"   Accuracy: {acc['gain_cached_over_baseline_rougeL']:+.1f}%")

        print(f"\n ACCURACY @ threshold={thresh} (ROUGE-L)")
        print(f"   Baseline   : {acc['baseline_rougeL']:.1f}%")
        print(f"   Cached RAG : {acc['cached_rag_rougeL']:.1f}%  "
              f"(retains {acc['retention_cached_vs_fresh_rougeL']:.1f}% of Fresh RAG accuracy)")
        print(f"   Fresh RAG  : {acc['fresh_rag_rougeL']:.1f}%")

        print(f"\n WIN RATES  (across {n} queries)")
        print(f"   Cached beats Baseline  ROUGE-L: "
              f"{wr['cached_beats_baseline_rougeL']:.1f}%")
        print(f"   Cached beats Fresh     ROUGE-L: "
              f"{wr['cached_beats_fresh_rougeL']:.1f}%")
        print("=" * 72)


class RAGEvaluator:
 
    def evaluate_answers(self, answers_df, retrieved_knowledge=None):
        try:
            n = int(len(answers_df))
            non_empty = int((answers_df.get("answer", "") != "").sum()) if "answer" in answers_df else 0
            unique_states = int(answers_df["state_id"].nunique()) if "state_id" in answers_df else 0
            return {
                "num_rows": n,
                "non_empty_answers": non_empty,
                "unique_states": unique_states,
                "note": (
                    "Per-window answer generation may be disabled; use "
                    "holdout_3method_comparison.csv and rouge_aggregate.json "
                    "for thesis RAG evaluation."
                ),
            }
        except Exception as e:
            return {"error": str(e)}
