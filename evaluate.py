#!/usr/bin/env python3
"""
evaluate.py

Simple evaluator for outputs produced by `inference.py`.
It supports JSON outputs that are either:
 - a list of result objects (each with fields: 'image_path', 'question', 'extracted_answer', optional 'ground_truth')
 - or a dict with a 'results' key containing such a list

Metrics computed:
 - Exact Match (EM)
 - Token-level F1 (prec/rec/F1)
 - Counting accuracy (when both prediction and ground truth contain numbers)

Usage:
    python evaluate.py --pred_file ./eval_results/results.json --out_dir ./eval_results/summary

"""
from __future__ import annotations
import argparse
import json
import os
import re
import csv
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

# --- Normalization utilities ---
_WORD_NUMBERS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12
}

PUNCT_RE = re.compile(r"[^0-9a-zA-Z\s]")


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = PUNCT_RE.sub(' ', s)
    s = re.sub(r"\s+", ' ', s)
    return s.strip()


def extract_number(s: str) -> Optional[int]:
    """Try to extract a single integer from text. Falls back to word numbers."""
    if not s:
        return None
    # digits first
    m = re.search(r"\b(\d+)\b", s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    # words
    toks = normalize_text(s).split()
    for t in toks:
        if t in _WORD_NUMBERS:
            return _WORD_NUMBERS[t]
    return None


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0, 1.0, 1.0
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# --- Core evaluation ---

def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    em_count = 0
    f1_sum = 0.0
    f1_count = 0
    counting_total = 0
    counting_correct = 0

    per_example = []

    for item in results:
        pred = item.get('extracted_answer') or item.get('prediction') or item.get('raw_output') or ''
        gold = item.get('ground_truth') or item.get('answer') or ''
        em = None
        p, r, f1 = 0.0, 0.0, 0.0

        if gold:
            em = 1 if exact_match(pred, gold) else 0
            p, r, f1 = token_f1(pred, gold)
            em_count += em
            f1_sum += f1
            f1_count += 1

            # counting
            pred_n = extract_number(pred)
            gold_n = extract_number(gold)
            if pred_n is not None and gold_n is not None:
                counting_total += 1
                if pred_n == gold_n:
                    counting_correct += 1

        per_example.append({
            'image_path': item.get('image_path') or item.get('image') or item.get('image_path', ''),
            'question': item.get('question', ''),
            'ground_truth': gold,
            'prediction': pred,
            'exact_match': em,
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f1, 4),
        })

    overall = {
        'total_examples': total,
        'em_count': em_count,
        'exact_match_accuracy': em_count / total if total > 0 else None,
        'avg_f1': (f1_sum / f1_count) if f1_count > 0 else None,
        'counting_total': counting_total,
        'counting_correct': counting_correct,
        'counting_accuracy': (counting_correct / counting_total) if counting_total > 0 else None,
    }

    return {'overall': overall, 'per_example': per_example}


# --- CLI / I/O ---

def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    # try common alternatives
    if isinstance(data, dict) and 'predictions' in data:
        return data['predictions']
    raise ValueError('Unsupported prediction file format: expected list or dict with key "results"')


def save_summary(summary: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'eval_summary.json')
    csv_path = os.path.join(out_dir, 'per_example.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary['overall'], f, indent=2, ensure_ascii=False)
    # write CSV
    rows = summary['per_example']
    if rows:
        keys = ['image_path', 'question', 'ground_truth', 'prediction', 'exact_match', 'precision', 'recall', 'f1']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, '') for k in keys})
    print(f"Saved JSON summary to {json_path}")
    if rows:
        print(f"Saved per-example CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate inference.py outputs')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to predictions JSON (output of inference.py)')
    parser.add_argument('--out_dir', type=str, default='./eval_results', help='Directory to save summary files')
    args = parser.parse_args()

    preds = load_predictions(args.pred_file)
    summary = evaluate(preds)
    save_summary(summary, args.out_dir)

    o = summary['overall']
    print('\n=== Evaluation Summary ===')
    print(f"Total examples: {o['total_examples']}")
    if o['exact_match_accuracy'] is not None:
        print(f"Exact Match: {o['exact_match_accuracy']*100:.2f}% ({o['em_count']}/{o['total_examples']})")
    if o['avg_f1'] is not None:
        print(f"Avg token F1: {o['avg_f1']:.4f}")
    if o['counting_accuracy'] is not None:
        print(f"Counting accuracy (on {o['counting_total']} numeric cases): {o['counting_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()

