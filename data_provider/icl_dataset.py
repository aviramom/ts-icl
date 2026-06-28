"""Standalone ICL dataset for UCR time series classification.

Extracted from MultiTSDataset.from_icl_ucr_dataset() in the original multimodalTS repo.
No dependency on the large dataset.py.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from picking_strategy import get_support_set
from utils.formatting import icl_classification_format


class ICLUCRDataset(Dataset):
    """Builds few-shot ICL prompts from a UCR train/test split."""

    ICL_KEYS = ["input_text", "output_text", "input_ts", "task_id", "options", "mean", "std"]

    def __init__(self):
        self.data = {k: [] for k in self.ICL_KEYS}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_ucr_dataset(cls, train_dataset, test_dataset, task_id: str, args) -> "ICLUCRDataset":
        """Build ICL prompts from a UCR train/test split.

        Args:
            train_dataset: UCRDataset or TimeSeriesExamDataset (train split)
            test_dataset:  UCRDataset or TimeSeriesExamDataset (test split)
            task_id:       e.g. "icl_ucr_GunPoint" or "icl_tse_3"
            args:          parsed CLI args (picking_strategy, num_shots, random_seed,
                           use_label_desc, desc_dir, num_samples)
        """
        self = cls()

        label_names = getattr(train_dataset, "label_names", None)
        is_two_series = getattr(train_dataset, "is_two_series", False)
        prompt_format = getattr(args, "prompt_format", None)
        num_shots = getattr(args, "num_shots", 1)

        # Determine if this is a zero-shot run (no support examples).
        is_zeroshot = prompt_format == "no_support" or \
                      (prompt_format == "tse_official" and num_shots == 0)

        if is_zeroshot:
            examples = []
            if label_names is not None:
                options = label_names
            else:
                all_ints = sorted(set(
                    int(train_dataset[i][1].item()) if hasattr(train_dataset[i][1], "item")
                    else int(train_dataset[i][1])
                    for i in range(len(train_dataset))
                ))
                options = [str(x) for x in all_ints]
        else:
            examples = get_support_set(
                train_dataset,
                strategy=args.picking_strategy,
                k_shots=num_shots,
                seed=getattr(args, "random_seed", None),
            )
            # Map integer labels -> display strings
            if label_names is not None:
                examples = [(ts_list, label_names[label]) for ts_list, label in examples]
            else:
                examples = [(ts_list, str(label)) for ts_list, label in examples]
            options = sorted(set(ex[1] for ex in examples))

        if prompt_format == "tse_official":
            # Build official MCQ description from raw question + option texts (bypasses _load_description).
            option_texts = getattr(train_dataset, "option_texts", None)
            question = getattr(train_dataset, "question", "")
            if option_texts and label_names and question:
                opts_block = "\n".join(f"{l}) {t}" for l, t in zip(label_names, option_texts))
                description = f"{question}\n\n{opts_block}"
                letter_to_text = dict(zip(label_names, option_texts))
            else:
                description = self._load_description(train_dataset, task_id, args)
                letter_to_text = {}
            # Remap example labels to "A) Linear" format for official MCQ answers.
            examples_official = [
                (ts, f"{letter}) {letter_to_text.get(letter, letter)}")
                for ts, letter in examples
            ]
            if is_two_series:
                input_prompt, support_ts_list = self._build_input_tse_official_two_series(
                    examples_official, description, options)
            else:
                input_prompt, support_ts_list = self._build_input_tse_official(
                    examples_official, description, options)
        else:
            description = self._load_description(train_dataset, task_id, args)
            if is_two_series:
                input_prompt, support_ts_list = self._build_input_two_series(
                    examples, description, options, prompt_format)
            else:
                input_prompt, support_ts_list = self._build_input(
                    examples, description, options, prompt_format)

        n = len(test_dataset)
        if getattr(args, "num_samples", None) not in (None, float("inf"), -1):
            rng = random.Random(args.random_seed)
            indices = sorted(rng.sample(range(n), min(args.num_samples, n)))
        else:
            indices = range(n)

        # "separate": keep <ts><ts/> placeholders; image/ChatTS models read raw input_ts.
        # "combined": embed TS arrays as text; text-only models read input_text.
        input_mode = getattr(args, "input_mode", "combined")

        for i in indices:
            item = test_dataset[i]
            label_i = item[1]
            label_val = label_i.item() if hasattr(label_i, "item") else label_i
            output_str = label_names[label_val] if label_names is not None else str(label_val)

            if is_two_series:
                ts_pair = item[0]  # (ts1_tensor, ts2_tensor)
                ts1_q = ts_pair[0].tolist() if hasattr(ts_pair[0], "tolist") else list(ts_pair[0])
                ts2_q = ts_pair[1].tolist() if hasattr(ts_pair[1], "tolist") else list(ts_pair[1])
                current_ts = support_ts_list + [ts1_q, ts2_q]
            else:
                ts_i = item[0]
                ts_query = ts_i.tolist() if hasattr(ts_i, "tolist") else list(ts_i)
                current_ts = support_ts_list + [ts_query]

            if input_mode == "separate":
                current_text = input_prompt
            else:
                current_text = self._combine_ts_text(input_prompt, current_ts)
            mean, std = self._compute_mean_std(current_ts)

            self._add_sample(
                input_text=current_text,
                output_text=output_str,
                input_ts=current_ts,
                task_id=task_id,
                options=options,
                mean=mean,
                std=std,
            )

        return self

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data["input_text"])

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.ICL_KEYS}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_sample(self, **kwargs):
        for k in self.ICL_KEYS:
            self.data[k].append(kwargs.get(k))

    @staticmethod
    def _load_description(train_dataset, task_id: str, args) -> str:
        prompt_format = getattr(args, "prompt_format", None)
        if prompt_format == "no_desc":
            return ""
        if prompt_format is None and not getattr(args, "use_label_desc", 0):
            return ""
        # TSE datasets store question text in self.desc directly
        if task_id.startswith("icl_tse_"):
            return getattr(train_dataset, "desc", "") or ""
        name = task_id.replace("ICL_UCR_", "").replace("icl_ucr_", "")
        desc_dir = getattr(args, "desc_dir", "ucr_descriptions")
        desc_path = os.path.join(desc_dir, name, "description.txt")
        if os.path.isfile(desc_path):
            with open(desc_path, encoding="utf-8") as f:
                return f.read().strip()
        return getattr(train_dataset, "desc", "") or ""

    @staticmethod
    def _build_input(examples, desc: str, opts: list, prompt_format=None):
        text = "\n--- EXAMPLES ---\n"
        ts_list = []
        for i, (ts, label) in enumerate(examples):
            text += f"\nExample {i+1} Time Series: <ts><ts/>\nLabel: {label}\n"
            ts_list.append(ts[0])
        target_ts = "New Time Series: <ts><ts/>"
        return icl_classification_format(desc, text, target_ts, opts, prompt_format), ts_list

    @staticmethod
    def _build_input_two_series(examples, desc: str, opts: list, prompt_format=None):
        """Like _build_input but emits two <ts><ts/> placeholders per example."""
        text = "\n--- EXAMPLES ---\n"
        ts_list = []
        for i, (ts, label) in enumerate(examples):
            # ts[0] = (ts1_tensor, ts2_tensor) tuple from the picking strategy
            ts_pair = ts[0]
            ts1 = ts_pair[0].tolist() if hasattr(ts_pair[0], "tolist") else list(ts_pair[0])
            ts2 = ts_pair[1].tolist() if hasattr(ts_pair[1], "tolist") else list(ts_pair[1])
            text += (
                f"\nExample {i+1} Time Series 1: <ts><ts/>\n"
                f"Example {i+1} Time Series 2: <ts><ts/>\n"
                f"Label: {label}\n"
            )
            ts_list.append(ts1)
            ts_list.append(ts2)
        target_ts = "New Time Series 1: <ts><ts/>\nNew Time Series 2: <ts><ts/>"
        return icl_classification_format(desc, text, target_ts, opts, prompt_format), ts_list

    @staticmethod
    def _build_input_tse_official(examples, desc: str, opts: list):
        """Official MCQ format: 'Answer: A) Linear' labels, 'Time Series:' target."""
        text = ""
        ts_list = []
        for i, (ts, label_str) in enumerate(examples):
            text += f"\nExample {i+1} Time Series: <ts><ts/>\nAnswer: {label_str}\n"
            ts_list.append(ts[0])
        target_ts = "Time Series: <ts><ts/>"
        return icl_classification_format(desc, text, target_ts, opts, "tse_official"), ts_list

    @staticmethod
    def _build_input_tse_official_two_series(examples, desc: str, opts: list):
        """Official MCQ format for two-series templates."""
        text = ""
        ts_list = []
        for i, (ts, label_str) in enumerate(examples):
            ts_pair = ts[0]
            ts1 = ts_pair[0].tolist() if hasattr(ts_pair[0], "tolist") else list(ts_pair[0])
            ts2 = ts_pair[1].tolist() if hasattr(ts_pair[1], "tolist") else list(ts_pair[1])
            text += (
                f"\nExample {i+1} Time Series 1: <ts><ts/>\n"
                f"Example {i+1} Time Series 2: <ts><ts/>\n"
                f"Answer: {label_str}\n"
            )
            ts_list.append(ts1)
            ts_list.append(ts2)
        target_ts = "Time Series 1: <ts><ts/>\nTime Series 2: <ts><ts/>"
        return icl_classification_format(desc, text, target_ts, opts, "tse_official"), ts_list

    @staticmethod
    def _combine_ts_text(input_text: str, ts_list: list) -> str:
        placeholder = "<ts><ts/>"
        for ts in ts_list:
            inner = ts[0] if isinstance(ts[0], list) else ts
            ts_str = ", ".join(f"{x:.4f}" for x in inner)
            input_text = input_text.replace(placeholder, f"[{ts_str}]", 1)
        return input_text

    @staticmethod
    def _compute_mean_std(ts_list):
        means, stds = [], []
        for series in ts_list:
            arr = np.array([np.nan if x is None else x for x in series], dtype=float)
            means.append(float(np.nanmean(arr)))
            stds.append(float(np.nanstd(arr)))
        return means, stds


# ------------------------------------------------------------------
# Collate function for DataLoader
# ------------------------------------------------------------------

def collate_icl(batch):
    """Collate ICL samples; keeps variable-length lists as lists."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if k in ("input_ts",):
            out[k] = vals
        elif isinstance(vals[0], (str, type(None))):
            out[k] = vals
        else:
            try:
                out[k] = torch.stack([
                    torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                    for v in vals
                ])
            except Exception:
                out[k] = vals
    return out
