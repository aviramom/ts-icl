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
            train_dataset: UCRDataset (train split)
            test_dataset:  UCRDataset (test split)
            task_id:       e.g. "icl_ucr_GunPoint"
            args:          parsed CLI args (picking_strategy, num_shots, random_seed,
                           use_label_desc, desc_dir, num_samples)
        """
        self = cls()

        examples = get_support_set(
            train_dataset,
            strategy=args.picking_strategy,
            k_shots=args.num_shots,
            seed=getattr(args, "random_seed", None),
        )

        description = self._load_description(train_dataset, task_id, args)
        options = sorted(set(ex[1] for ex in examples))
        input_prompt, support_ts_list = self._build_input(examples, description, options)

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
            ts_i, label_i = test_dataset[i]
            ts_query = ts_i.tolist() if hasattr(ts_i, "tolist") else list(ts_i)
            label_val = label_i.item() if hasattr(label_i, "item") else label_i

            current_ts = support_ts_list + [ts_query]
            if input_mode == "separate":
                current_text = input_prompt  # placeholders left intact for model
            else:
                current_text = self._combine_ts_text(input_prompt, current_ts)
            mean, std = self._compute_mean_std(current_ts)

            self._add_sample(
                input_text=current_text,
                output_text=str(label_val),
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
        if not getattr(args, "use_label_desc", 0):
            return ""
        name = task_id.replace("ICL_UCR_", "").replace("icl_ucr_", "")
        desc_dir = getattr(args, "desc_dir", "ucr_descriptions")
        desc_path = os.path.join(desc_dir, name, "description.txt")
        if os.path.isfile(desc_path):
            with open(desc_path, encoding="utf-8") as f:
                return f.read().strip()
        return getattr(train_dataset, "desc", "") or ""

    @staticmethod
    def _build_input(examples, desc: str, opts: list):
        text = "\n--- EXAMPLES ---\n"
        ts_list = []
        for i, (ts, label) in enumerate(examples):
            text += f"\nExample {i+1} Time Series: <ts><ts/>\nLabel: {label}\n"
            ts_list.append(ts[0])
        target = "\n--- TARGET ---\n" + "New Time Series: <ts><ts/>\n"
        return icl_classification_format(desc, text, target, opts), ts_list

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
