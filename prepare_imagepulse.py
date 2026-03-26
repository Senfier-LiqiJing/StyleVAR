"""
ImagePulse dataset preparation script for space-constrained environments.

Extracts tar.gz files one by one, processes each sample into a clean folder,
then deletes the tar.gz and raw extracted data before moving to the next archive.

Output structure:
    output_dir/
    ├── 0000000/
    │   ├── content.png      (image_1 - original/baseline)
    │   ├── style.png        (image_2 - style reference)
    │   ├── target.png       (image_4 - high quality edited)
    │   └── meta.json        (image correspondence + prompt info)
    ├── 0000001/
    │   ├── ...
    ├── train_files.txt
    └── val_files.txt

Usage:
    python prepare_imagepulse.py --tar_dir ./data/ImagePulse/data --output_dir ./data/ImagePulse/processed
"""

import argparse
import glob
import json
import os
import os.path as osp
import random
import shutil
import tarfile


def process_one_tar(tar_path: str, tmp_dir: str, output_dir: str,
                    sample_counter: int) -> int:
    """
    Extract one tar.gz into tmp_dir, process each sample, write to output_dir.
    Returns the updated sample counter.
    """
    archive_name = osp.basename(tar_path).replace(".tar.gz", "")

    # Extract
    print(f"  Extracting {osp.basename(tar_path)} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)

    # The tar extracts into tmp_dir/<archive_name>/
    extracted_root = osp.join(tmp_dir, archive_name)
    if not osp.isdir(extracted_root):
        # Some tars may extract flat - try tmp_dir itself
        extracted_root = tmp_dir

    # Find all JSON metadata files
    json_files = sorted(glob.glob(osp.join(extracted_root, "*.json")))
    processed = 0
    skipped = 0

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
        except Exception:
            skipped += 1
            continue

        # Need image_1 (content), image_2 (style), image_4 (target HQ)
        if not all(k in meta for k in ("image_1", "image_2", "image_4")):
            skipped += 1
            continue

        base_dir = osp.dirname(jf)
        content_src = osp.join(base_dir, meta["image_1"])
        style_src = osp.join(base_dir, meta["image_2"])
        target_src = osp.join(base_dir, meta["image_4"])

        if not (osp.isfile(content_src) and osp.isfile(style_src) and osp.isfile(target_src)):
            skipped += 1
            continue

        # Create sample folder: 7-digit zero-padded index
        sample_dir = osp.join(output_dir, f"{sample_counter:07d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Move images (move instead of copy to save space during processing)
        shutil.move(content_src, osp.join(sample_dir, "content.png"))
        shutil.move(style_src, osp.join(sample_dir, "style.png"))
        shutil.move(target_src, osp.join(sample_dir, "target.png"))

        # Build clean metadata JSON
        sample_meta = {
            "sample_id": sample_counter,
            "content": "content.png",
            "style": "style.png",
            "target": "target.png",
            "prompt": meta.get("prompt", ""),
            "editing_instruction": meta.get("editing_instruction", ""),
            "reverse_editing_instruction": meta.get("reverse_editing_instruction", ""),
            "image_content_description": meta.get("image_content_description", ""),
            "image_content_style_description": meta.get("image_content_style_description", ""),
            "content_caption": meta.get("image_1_caption", ""),
            "target_caption": meta.get("image_4_caption", ""),
            "content_preference_score": meta.get("image_1_preference_score", None),
            "target_preference_score": meta.get("image_4_preference_score", None),
            "source_archive": archive_name,
            "source_json": osp.basename(jf),
        }

        with open(osp.join(sample_dir, "meta.json"), "w") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)

        sample_counter += 1
        processed += 1

    print(f"    -> {processed} samples processed, {skipped} skipped")
    return sample_counter


def split_train_val(output_dir: str, val_ratio: float = 0.10, seed: int = 42):
    """Write train_files.txt and val_files.txt listing sample folder names."""
    all_samples = sorted([
        d for d in os.listdir(output_dir)
        if osp.isdir(osp.join(output_dir, d)) and d.isdigit()
    ])
    total = len(all_samples)
    if total == 0:
        print("[Split] No samples found.")
        return

    random.seed(seed)
    random.shuffle(all_samples)
    val_count = max(int(total * val_ratio), 1)
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    with open(osp.join(output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_samples) + "\n")
    with open(osp.join(output_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_samples) + "\n")

    print(f"[Split] total={total}, train={len(train_samples)}, val={len(val_samples)}")


def main():
    p = argparse.ArgumentParser("Prepare ImagePulse dataset (space-efficient)")
    p.add_argument("--tar_dir", type=str, required=True,
                   help="Directory containing .tar.gz files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for processed samples")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_tar", action="store_true",
                   help="Do NOT delete tar.gz after processing (default: delete)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = osp.join(args.output_dir, "_tmp_extract")

    tar_files = sorted(glob.glob(osp.join(args.tar_dir, "*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz files in {args.tar_dir}")

    # Check for existing samples to support resume
    existing = [
        d for d in os.listdir(args.output_dir)
        if osp.isdir(osp.join(args.output_dir, d)) and d.isdigit()
    ]
    sample_counter = len(existing)
    if sample_counter > 0:
        print(f"Resuming from sample {sample_counter} (found {sample_counter} existing)")

    for i, tar_path in enumerate(tar_files):
        print(f"[{i+1}/{len(tar_files)}] {osp.basename(tar_path)}")

        # Check if this archive was already processed (marker file)
        marker = osp.join(args.output_dir, f".done_{osp.basename(tar_path)}")
        if osp.exists(marker):
            print("    -> already processed, skipping")
            continue

        # Clean tmp dir
        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        try:
            sample_counter = process_one_tar(
                tar_path, tmp_dir, args.output_dir, sample_counter
            )
            # Write marker
            with open(marker, "w") as f:
                f.write(f"processed {sample_counter}\n")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        finally:
            # Always clean up tmp
            if osp.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        # Delete tar.gz to free space
        if not args.keep_tar:
            os.remove(tar_path)
            print(f"    -> deleted {osp.basename(tar_path)}")

    # Train/val split
    print(f"\nTotal samples: {sample_counter}")
    split_train_val(args.output_dir, val_ratio=args.val_ratio, seed=args.seed)
    print("Done!")


if __name__ == "__main__":
    main()
