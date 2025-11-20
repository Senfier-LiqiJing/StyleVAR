import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import build_dataset


def inspect_batches(loader: DataLoader, num_batches: int):
    fetched = 0
    for batch_idx, batch in enumerate(loader):
        if batch is None:
            print(f"[Test] Batch {batch_idx} was empty after filtering.")
            continue

        target, style, content = batch
        print(
            f"[Test] batch={batch_idx} "
            f"target={tuple(target.shape)} style={tuple(style.shape)} content={tuple(content.shape)} "
            f"dtype={target.dtype}"
        )

        fetched += 1
        if fetched >= num_batches:
            break

    if fetched == 0:
        print("[Test] No valid batches were produced with this configuration.")


def try_split(name: str, dataset, batch_sizes: Iterable[int], workers: int, batches_per_size: int):
    length = len(dataset)
    if length == 0:
        print(f"[Test] Split '{name}' is empty, skip.")
        return

    print(f"[Test] Split '{name}' has {length} samples.")
    for bs in sorted(set(batch_sizes)):
        print(f"\n[Test] === {name} loader with batch_size={bs} ===")
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        inspect_batches(loader, batches_per_size)


def main():
    parser = argparse.ArgumentParser(description="Validate StyleTransfer build_dataset pipeline.")
    parser.add_argument("--data-root", default="dataset", help="Path to the dataset root containing target/style/content.")
    parser.add_argument("--final-reso", type=int, default=256, help="Final crop size.")
    parser.add_argument("--mid-reso", type=float, default=1.125, help="Intermediate resize multiplier.")
    parser.add_argument("--hflip", action="store_true", help="Enable random horizontal flip in training transform.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to try.")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--batches-per-size", type=int, default=2, help="Batches inspected per loader.")
    args = parser.parse_args()

    num_classes, train_set, val_set = build_dataset(
        data_path=args.data_root,
        final_reso=args.final_reso,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
    )
    print(f"[Test] build_dataset -> num_classes={num_classes}, train_len={len(train_set)}, val_len={len(val_set)}")

    try_split("train", train_set, args.batch_sizes, args.workers, args.batches_per_size)
    try_split("val", val_set, args.batch_sizes, args.workers, args.batches_per_size)


if __name__ == "__main__":
    main()
