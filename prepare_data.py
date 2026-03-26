"""
Data preparation script for StyleVAR mixed fine-tuning.

1. OmniStyle-150K  ──  split 90% train / 10% val (file-list based, no copying)
2. ImagePulse      ──  extract all tar.gz, then build symlink-based
                        content / style / target directories so it is
                        compatible with StyleTransferDataset.

Usage:
    python prepare_data.py [--data_root ./data]
"""

import argparse
import glob
import json
import os
import os.path as osp
import random
import shutil
import tarfile
from collections import OrderedDict


# ====================================================================
# 1.  OmniStyle-150K  ─  90 / 10 train / val split  (file-list only)
# ====================================================================

def split_omnistyle(omnistyle_root: str, val_ratio: float = 0.10, seed: int = 42):
    """
    Write  train_files.txt  and  val_files.txt  under ``omnistyle_root``.
    Each line is a target filename (the ``content_name&&style_name`` format).

    Also creates a ``target`` symlink → ``OmniStyle-150K`` so that
    ``StyleTransferDataset(root_dir=omnistyle_root)`` works out of the box.
    """
    # The actual target images live in the nested OmniStyle-150K/ subdir
    real_target_dir = osp.join(omnistyle_root, "OmniStyle-150K")
    if not osp.isdir(real_target_dir):
        print(f"[OmniStyle] target dir not found: {real_target_dir}")
        return

    # Create  target -> OmniStyle-150K  symlink for StyleTransferDataset compat
    target_link = osp.join(omnistyle_root, "target")
    if not osp.exists(target_link):
        os.symlink(osp.abspath(real_target_dir), target_link)
        print(f"[OmniStyle] Created symlink: target -> OmniStyle-150K")
    target_dir = real_target_dir

    all_files = sorted([f for f in os.listdir(target_dir) if "&&" in f])
    total = len(all_files)
    if total == 0:
        print("[OmniStyle] No target files found.")
        return

    random.seed(seed)
    random.shuffle(all_files)
    val_count = max(int(total * val_ratio), 1)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    train_path = osp.join(omnistyle_root, "train_files.txt")
    val_path = osp.join(omnistyle_root, "val_files.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(train_files) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_files) + "\n")

    print(f"[OmniStyle] total={total}, train={len(train_files)}, val={len(val_files)}")
    print(f"  -> {train_path}")
    print(f"  -> {val_path}")


# ====================================================================
# 2.  ImagePulse  ─  extract tar.gz  +  organise into triplet dirs
# ====================================================================

def extract_all_tars(tar_dir: str, extract_dir: str):
    """Extract every .tar.gz in *tar_dir* into *extract_dir* (skip existing)."""
    os.makedirs(extract_dir, exist_ok=True)
    tar_files = sorted(glob.glob(osp.join(tar_dir, "*.tar.gz")))
    print(f"[ImagePulse] Found {len(tar_files)} tar.gz in {tar_dir}")

    for i, tf_path in enumerate(tar_files):
        archive_name = osp.basename(tf_path).replace(".tar.gz", "")
        dest = osp.join(extract_dir, archive_name)
        if osp.isdir(dest) and len(os.listdir(dest)) > 0:
            continue
        print(f"  [{i+1}/{len(tar_files)}] Extracting {osp.basename(tf_path)} ...")
        try:
            with tarfile.open(tf_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"  WARNING: {e}")


def organise_imagepulse(extracted_dir: str, output_dir: str,
                        val_ratio: float = 0.10, seed: int = 42):
    """
    Walk through extracted ImagePulse archives and create an OmniStyle-
    compatible layout under *output_dir*::

        output_dir/
        ├── content/        (symlinks  →  image_1)
        ├── style/          (symlinks  →  image_2)
        ├── target/         (symlinks  →  image_4)
        ├── train_files.txt
        └── val_files.txt

    The "target" filename follows the OmniStyle convention:
        ``<content_basename>&&<style_basename>.png``
    so that ``StyleTransferDataset`` can parse it identically.
    """
    content_dir = osp.join(output_dir, "content")
    style_dir = osp.join(output_dir, "style")
    target_dir = osp.join(output_dir, "target")
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    json_files = sorted(glob.glob(osp.join(extracted_dir, "**", "*.json"),
                                  recursive=True))
    print(f"[ImagePulse] Scanning {len(json_files)} JSON metadata files ...")

    target_names = []   # list of constructed target filenames
    skipped = 0

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
        except Exception:
            skipped += 1
            continue

        needed = ("image_1", "image_2", "image_4")
        if not all(k in meta for k in needed):
            skipped += 1
            continue

        base = osp.dirname(jf)
        c_src = osp.join(base, meta["image_1"])   # content
        s_src = osp.join(base, meta["image_2"])   # style
        t_src = osp.join(base, meta["image_4"])   # target

        if not (osp.isfile(c_src) and osp.isfile(s_src) and osp.isfile(t_src)):
            skipped += 1
            continue

        # Build OmniStyle-compatible target filename:
        #   <content_basename>&&<style_basename>.png
        c_base = osp.basename(c_src)               # e.g. 174402738358.png
        s_base = osp.basename(s_src)               # e.g. 174402738411.png
        tgt_name = f"{c_base}&&{s_base}.png"

        # Symlink (or copy) into flat dirs
        c_dst = osp.join(content_dir, c_base)
        s_dst = osp.join(style_dir, s_base)
        t_dst = osp.join(target_dir, tgt_name)

        # content & style may be shared across samples; skip if exists
        if not osp.exists(c_dst):
            os.symlink(osp.abspath(c_src), c_dst)
        if not osp.exists(s_dst):
            os.symlink(osp.abspath(s_src), s_dst)
        if not osp.exists(t_dst):
            os.symlink(osp.abspath(t_src), t_dst)

        target_names.append(tgt_name)

    print(f"[ImagePulse] Organised {len(target_names)} triplets "
          f"(skipped {skipped} incomplete entries)")

    # ---- train / val split ----
    random.seed(seed)
    random.shuffle(target_names)
    val_count = max(int(len(target_names) * val_ratio), 1)
    val_files = target_names[:val_count]
    train_files = target_names[val_count:]

    with open(osp.join(output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files) + "\n")
    with open(osp.join(output_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_files) + "\n")

    print(f"[ImagePulse] train={len(train_files)}, val={len(val_files)}")


# ====================================================================
# main
# ====================================================================

def main():
    p = argparse.ArgumentParser("Prepare data for StyleVAR mixed fine-tuning")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Root data directory")
    p.add_argument("--val_ratio", type=float, default=0.10,
                   help="Fraction held out for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = args.data_root

    # ---- OmniStyle ----
    omnistyle = osp.join(root, "OmniStyle-150K")
    if osp.isdir(omnistyle):
        split_omnistyle(omnistyle, val_ratio=args.val_ratio, seed=args.seed)
    else:
        print(f"[OmniStyle] Not found at {omnistyle}, skipping.")

    # ---- ImagePulse ----
    ip_tar_dir = osp.join(root, "ImagePulse", "data")
    ip_extracted = osp.join(root, "ImagePulse", "extracted")
    ip_organised = osp.join(root, "ImagePulse", "organised")

    if osp.isdir(ip_tar_dir):
        extract_all_tars(ip_tar_dir, ip_extracted)
        organise_imagepulse(ip_extracted, ip_organised,
                            val_ratio=args.val_ratio, seed=args.seed)
    else:
        print(f"[ImagePulse] tar dir not found at {ip_tar_dir}, skipping.")

    print("\n[Done] Data preparation complete.")


if __name__ == "__main__":
    main()
