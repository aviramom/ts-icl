"""TimeSeriesExam dataset wrapper for ICL classification.

Each TimeSeriesExamDataset instance wraps ONE question template (identified by
`tid`).  It treats each option as a classification class and exposes the
per-class generated TS variants as a standard Dataset.

Interface matches UCRDataset so the existing ICLUCRDataset pipeline works:
  - __len__ / __getitem__ -> (ts_tensor, label_tensor)
  - self.labels   : torch.long tensor of all labels
  - self.data     : tensor of all TS  (single-TS questions)
  - self.data1/2  : tensors for two-series questions
  - self.label_names  : list of option-letter strings ["A", "B", ...]
  - self.option_texts : list of option display names per class
  - self.is_two_series: bool
  - self.desc     : question text + option legend (for --use_label_desc 1)
"""

import json
import math
import random
import torch
from torch.utils.data import Dataset


_LETTERS = "ABCDEFGHIJ"


class TimeSeriesExamDataset(Dataset):
    """One TSE question template as a k-class classification dataset."""

    def __init__(
        self,
        augmented_path: str,
        tid: int,
        split: str = "train",
        test_fraction: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            augmented_path: path to qa_dataset_augmented.json
            tid:            template id (matches the `tid` field in the augmented file)
            split:          "train" or "test"
            test_fraction:  fraction of variants reserved for the test (query) split
            seed:           shuffle seed for train/test split
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        with open(augmented_path, encoding="utf-8") as f:
            all_entries = json.load(f)

        entry = next((e for e in all_entries if e["tid"] == tid), None)
        if entry is None:
            available = sorted({e["tid"] for e in all_entries})
            raise ValueError(f"tid={tid} not found in {augmented_path}. Available: {available}")

        self.tid = tid
        self.is_two_series: bool = entry["is_two_series"]
        option_names: list = entry["option_names"]   # display text per option
        ts_variants: dict = entry["ts_variants"]      # option_name -> list of variants

        num_options = len(option_names)
        letters = list(_LETTERS[:num_options])
        self.label_names: list = letters              # ["A", "B", ...]
        self.option_texts: list = option_names        # display text per option

        # Build description: question + option legend
        q = entry.get("question", "")
        opt_lines = "\n".join(
            f"{letter}) {text}" for letter, text in zip(letters, option_names)
        )
        self.desc: str = f"Question: {q}\n\nOptions:\n{opt_lines}"

        # Split variants into train / test
        rng = random.Random(seed)
        ts_tensors = []
        ts1_tensors, ts2_tensors = [], []
        label_ints = []

        for opt_idx, opt_name in enumerate(option_names):
            variants = ts_variants.get(opt_name, [])
            if not variants:
                continue
            variants = list(variants)
            rng.shuffle(variants)
            n_test = max(1, math.ceil(len(variants) * test_fraction))
            n_train = len(variants) - n_test

            if split == "train":
                selected = variants[:n_train]
            else:
                selected = variants[n_train:]

            for v in selected:
                if self.is_two_series:
                    ts1_tensors.append(torch.tensor(v[0], dtype=torch.float32))
                    ts2_tensors.append(torch.tensor(v[1], dtype=torch.float32))
                else:
                    ts_tensors.append(torch.tensor(v, dtype=torch.float32))
                label_ints.append(opt_idx)

        self.labels = torch.tensor(label_ints, dtype=torch.long)

        if self.is_two_series:
            if ts1_tensors:
                self.data1 = torch.stack(ts1_tensors)
                self.data2 = torch.stack(ts2_tensors)
            else:
                self.data1 = torch.zeros(0, 1)
                self.data2 = torch.zeros(0, 1)
            self.data = self.data1  # satisfy any code that reads .data
        else:
            if ts_tensors:
                self.data = torch.stack(ts_tensors)
            else:
                self.data = torch.zeros(0, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.is_two_series:
            return (self.data1[idx], self.data2[idx]), label
        return self.data[idx], label
