"""
Split OmniStyle-150K into 95% train / 5% val.

Reads target filenames directly from data/OmniStyle-150K/OmniStyle-150K/,
writes train_files.txt and val_files.txt under data/OmniStyle-150K/.

Usage:
    python split_omnistyle.py [--data_dir ./data/OmniStyle-150K/OmniStyle-150K]
                              [--output_dir ./data/OmniStyle-150K]
                              [--val_ratio 0.05] [--seed 42]
"""

import argparse
import os
import random


def main():
    p = argparse.ArgumentParser("Split OmniStyle-150K train/val")
    p.add_argument("--data_dir", type=str,
                   default="./data/OmniStyle-150K/OmniStyle-150K",
                   help="Directory containing target images (content&&style.png)")
    p.add_argument("--output_dir", type=str,
                   default="./data/OmniStyle-150K",
                   help="Where to write train_files.txt / val_files.txt")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_files = sorted([f for f in os.listdir(args.data_dir) if "&&" in f])
    total = len(all_files)
    if total == 0:
        print("No target files found (expecting filenames with '&&').")
        return

    random.seed(args.seed)
    random.shuffle(all_files)
    val_count = max(int(total * args.val_ratio), 1)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_files.txt")
    val_path = os.path.join(args.output_dir, "val_files.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_files) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_files) + "\n")

    print(f"Total: {total}")
    print(f"Train: {len(train_files)} -> {train_path}")
    print(f"Val:   {len(val_files)} -> {val_path}")


if __name__ == "__main__":
    main()
