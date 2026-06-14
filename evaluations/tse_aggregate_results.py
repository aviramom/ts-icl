#!/usr/bin/env python3
"""
Aggregate per-tid TSE results from the outputs/ directory or W&B.

Usage:
    python evaluations/tse_aggregate_results.py \
        --results_dir outputs/ \
        --augmented_path qa_dataset_augmented.json \
        --method Qwen/Qwen3-4B-Instruct-2507

Reads all JSON files in results_dir matching icl_tse_* and the given method,
then prints a summary table by difficulty and category.
"""

import os
import json
import argparse
from collections import defaultdict


def load_results(results_dir: str, method: str):
    """Load per-tid result JSONs from outputs/. Returns list of (tid, result_dict)."""
    method_tag = method.replace("/", "_").replace(".", "v")
    records = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        if "icl_tse_" not in fname:
            continue
        if method_tag and method_tag not in fname:
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            obj = json.load(f)
        task_id = obj.get("args", {}).get("task_id", "")
        if not task_id.startswith("icl_tse_"):
            continue
        tid = int(task_id.replace("icl_tse_", ""))
        records.append((tid, obj.get("results", {})))
    return records


def load_tid_metadata(augmented_path: str):
    """Load {tid -> {category, subcategory, difficulty}} from augmented JSON."""
    with open(augmented_path) as f:
        entries = json.load(f)
    return {e["tid"]: e for e in entries}


def aggregate(records, tid_meta):
    by_difficulty = defaultdict(list)
    by_category = defaultdict(list)
    by_subcategory = defaultdict(list)
    all_acc = []

    for tid, res in records:
        acc = res.get("balanced_accuracy")
        if acc is None:
            continue
        all_acc.append(acc)
        meta = tid_meta.get(tid, {})
        diff = meta.get("difficulty", "unknown")
        cat = meta.get("category", "unknown")
        sub = meta.get("subcategory", "unknown")
        by_difficulty[diff].append(acc)
        by_category[cat].append(acc)
        by_subcategory[sub].append(acc)

    return all_acc, by_difficulty, by_category, by_subcategory


def mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")


def print_table(title, d):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Group':<40} {'N':>4}  {'Mean Bal-Acc':>12}")
    print(f"  {'-'*60}")
    for key in sorted(d):
        vals = d[key]
        print(f"  {key:<40} {len(vals):>4}  {mean(vals):>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs/")
    parser.add_argument("--augmented_path", type=str, default="qa_dataset_augmented.json")
    parser.add_argument("--method", type=str, default="",
                        help="Filter by model method string (substring match)")
    args = parser.parse_args()

    records = load_results(args.results_dir, args.method)
    if not records:
        print("No matching result files found.")
        return

    tid_meta = load_tid_metadata(args.augmented_path)
    all_acc, by_diff, by_cat, by_sub = aggregate(records, tid_meta)

    print(f"\nTotal templates evaluated: {len(all_acc)}")
    print(f"Overall mean balanced accuracy: {mean(all_acc):.4f}")

    print_table("By Difficulty", by_diff)
    print_table("By Category", by_cat)
    print_table("By Subcategory", by_sub)


if __name__ == "__main__":
    main()
