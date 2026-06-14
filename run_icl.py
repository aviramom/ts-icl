#!/usr/bin/env python3
"""Entry point for ICL time series classification evaluation."""

import os
import sys
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()

from utils.args import get_parser
from utils.model import method_wrapper_dict
from data_provider.dataset_iclucr import UCRDataset
from data_provider.dataset_tse import TimeSeriesExamDataset
from data_provider.icl_dataset import ICLUCRDataset, collate_icl
from evaluations.icl_ucr_eval import run_evaluation_icl_ucr
from loggers import setup_logger


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Let the model class inject its own args (e.g. tensor_parallel_size)
    wrapper_class = method_wrapper_dict[args.method]
    args = wrapper_class.get_relevant_args(args, parser)

    # Reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Logging
    logger = setup_logger(args)
    if logger.is_completed():
        print("Run already completed, exiting.")
        return
    logger.log_hparams(vars(args))

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if args.task_id.startswith("icl_tse_"):
        tid = int(args.task_id.replace("icl_tse_", ""))
        tse_data_path = getattr(args, "tse_data_path", "qa_dataset_augmented.json")
        tse_test_fraction = getattr(args, "tse_test_fraction", 0.3)
        print(f"Loading TimeSeriesExam template tid={tid} from {tse_data_path}")
        train_ds = TimeSeriesExamDataset(
            tse_data_path, tid, split="train",
            test_fraction=tse_test_fraction, seed=args.random_seed,
        )
        test_ds = TimeSeriesExamDataset(
            tse_data_path, tid, split="test",
            test_fraction=tse_test_fraction, seed=args.random_seed,
        )
    else:
        dataset_name = args.task_id.replace("ICL_UCR_", "").replace("icl_ucr_", "")
        ucr_path = os.path.join(args.data_path, "Univariate_arff", dataset_name)
        print(f"Loading UCR dataset: {dataset_name}")
        train_ds = UCRDataset(ucr_path, split="train")
        test_ds = UCRDataset(ucr_path, split="test")

    print("Building ICL prompts...")
    icl_dataset = ICLUCRDataset.from_ucr_dataset(
        train_ds, test_ds, task_id=args.task_id, args=args
    )
    dataloader = DataLoader(
        icl_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_icl,
    )
    print(f"  {len(icl_dataset)} test samples, batch_size={args.batch_size}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.method}")
    model = wrapper_class(args, device=getattr(args, "device", "cuda"))
    if hasattr(model, "setup"):
        model.setup(args)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("Running evaluation...")
    results, input_output = run_evaluation_icl_ucr(model, dataloader, args)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\n==== Results ====")
    for k, v in results.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    print("\n==== Sample Predictions ====")
    n_display = getattr(args, "display_samples", 3)
    for i in range(min(n_display, len(input_output["questions"]))):
        print(f"  [{i+1}] input: {input_output['questions'][i][:120]}...")
        print(f"       pred:  {input_output['generated_texts'][i]}")
        print(f"       gold:  {input_output['gold_answers'][i]}")

    # ------------------------------------------------------------------
    # Log & save
    # ------------------------------------------------------------------
    loggable = {k: v for k, v in results.items() if not isinstance(v, list)}
    logger.log_metrics(loggable)

    os.makedirs("outputs", exist_ok=True)
    model_tag = args.method.replace("/", "_").replace(".", "v")
    out_path = f"outputs/{args.task_id}_{args.num_samples}_{model_tag}_exp_{args.exp_id}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": loggable}, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    logger.close()


if __name__ == "__main__":
    main()
