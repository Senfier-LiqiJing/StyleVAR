import glob
import json
import os
import os.path as osp
import random
import tarfile

import PIL.Image as PImage
import torch
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class StyleTransferDataset(Dataset):
    def __init__(self, root_dir, transform=None, file_list=None):
        """
        root_dir: path containing  target/, style/, content/  sub-directories.
        transform: torchvision transforms
        file_list: optional list of target filenames (for train/val split).
                   If None, scans the target dir for all files with '&&'.
        """
        self.root_dir = root_dir
        self.target_dir = osp.join(root_dir, 'target')
        self.style_dir = osp.join(root_dir, 'style')
        self.content_dir = osp.join(root_dir, 'content')
        self.transform = transform

        if file_list is not None:
            self.target_files = list(file_list)
        else:
            self.target_files = [f for f in os.listdir(self.target_dir) if '&&' in f]

        print(f"[Dataset] Found {len(self.target_files)} target images.")
        
    def __len__(self):
        return len(self.target_files)
    
    def __getitem__(self, idx):
        target_filename = self.target_files[idx]
        
        # 2. Parse filenames
        # Format: content_part&&style_part
        try:
            content_name_raw, style_name_raw = target_filename.split('&&')
            
            # Content image always ends with .png (from prompt req 3)
            # The target filename seems to keep the content extension, e.g., "name.png"
            content_path = osp.join(self.content_dir, content_name_raw)
            
            # Style image logic (from prompt req 3):
            # "the first might be jpg,jpeg or wpeg... but the last suffix must be .jpg"
            # The target filename usually has the full style name including original extensions.
            # We assume style_name_raw matches the file in style_dir directly, 
            # OR we need to ensure it ends with .jpg if the split stripped it.
            style_name_raw = style_name_raw[:-4]
            style_path = osp.join(self.style_dir, style_name_raw)
            
            target_path = osp.join(self.target_dir, target_filename)
            
            # 3. Load Images
            content_img = pil_loader(content_path)
            style_img = pil_loader(style_path)
            target_img = pil_loader(target_path)
            
            # 4. Apply Transforms
            if self.transform:
                content_img = self.transform(content_img)
                style_img = self.transform(style_img)
                target_img = self.transform(target_img)
                
            return target_img, style_img, content_img
            
        except Exception as e:
            print(f"[Dataset Error] Failed to load idx {idx}: {target_filename}. Error: {e}")
            # Return a dummy valid sample (or handle error appropriately)
            # For now, just recursively get the next one to avoid crashing training
            return self.__getitem__((idx + 1) % len(self))

class SubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        t, s, c = self.subset[idx] # These are PIL images because transform=None above
        return self.transform(t), self.transform(s), self.transform(c)

def _read_file_list(path: str):
    """Read a newline-separated file list, return list of non-empty lines."""
    if not osp.isfile(path):
        return None
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations (Same as vanilla VAR)
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso

    common_transform = transforms.Compose([
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])

    if hflip:
        print("Warning: hflip is disabled to ensure Content-Target alignment during fine-tuning.")

    train_aug = common_transform
    val_aug = common_transform

    # If prepare_data.py has been run, use the pre-computed splits
    train_files = _read_file_list(osp.join(data_path, "train_files.txt"))
    val_files = _read_file_list(osp.join(data_path, "val_files.txt"))

    if train_files is not None and val_files is not None:
        print(f"[Dataset] Using pre-computed splits from {data_path}")
        train_set = StyleTransferDataset(root_dir=data_path, transform=train_aug,
                                         file_list=train_files)
        val_set = StyleTransferDataset(root_dir=data_path, transform=val_aug,
                                       file_list=val_files)
    else:
        # Fallback: random split (old behaviour)
        full_dataset = StyleTransferDataset(root_dir=data_path, transform=None)
        total_len = len(full_dataset)
        val_len = int(total_len * 0.05)
        train_len = total_len - val_len
        from torch.utils.data import random_split
        train_subset, val_subset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42))
        train_set = SubsetWrapper(train_subset, train_aug)
        val_set = SubsetWrapper(val_subset, val_aug)

    num_classes = 1000
    print(f'[Dataset] Train: {len(train_set)}, Val: {len(val_set)}')
    return num_classes, train_set, val_set


# ===========================================================================
# ImagePulse-StyleTransfer dataset & Curriculum Mixed training
# ===========================================================================

def extract_imagepulse_tars(tar_dir: str, extract_dir: str) -> str:
    """
    Extract all .tar.gz files from `tar_dir` into `extract_dir`.
    Skips archives that are already extracted.  Returns `extract_dir`.
    """
    os.makedirs(extract_dir, exist_ok=True)
    tar_files = sorted(glob.glob(osp.join(tar_dir, "*.tar.gz")))
    print(f"[ImagePulse] Found {len(tar_files)} tar.gz files in {tar_dir}")

    for tf_path in tar_files:
        archive_name = osp.basename(tf_path).replace(".tar.gz", "")
        dest = osp.join(extract_dir, archive_name)
        if osp.isdir(dest) and len(os.listdir(dest)) > 0:
            continue  # already extracted
        print(f"[ImagePulse] Extracting {osp.basename(tf_path)} ...")
        try:
            with tarfile.open(tf_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"[ImagePulse] WARNING: failed to extract {tf_path}: {e}")
    return extract_dir


class ImagePulseDataset(Dataset):
    """
    Dataset for DiffSynth-Studio/ImagePulse-StyleTransfer (extracted tar.gz).

    Expected layout after extraction::

        root_dir/
        ├── <archive_id_1>/
        │   ├── <sample>.json   (has image_1, image_2, image_4 fields)
        │   ├── <image_1>.png   (content)
        │   ├── <image_2>.png   (style reference)
        │   ├── <image_4>.png   (target / final style-transferred result)
        │   └── ...
        └── <archive_id_2>/ ...

    Returns (target_img, style_img, content_img) — same order as
    ``StyleTransferDataset`` so both are interchangeable in the trainer.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (content_path, style_path, target_path)
        self._index(root_dir)
        print(f"[ImagePulse] Indexed {len(self.samples)} triplets from {root_dir}")

    # ------------------------------------------------------------------ #
    def _index(self, root_dir: str):
        json_files = sorted(glob.glob(osp.join(root_dir, "**", "*.json"), recursive=True))
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue
            # Require the three image fields
            if not all(k in meta for k in ("image_1", "image_2", "image_4")):
                continue
            base_dir = osp.dirname(jf)
            content_path = osp.join(base_dir, meta["image_1"])
            style_path   = osp.join(base_dir, meta["image_2"])
            target_path  = osp.join(base_dir, meta["image_4"])
            if osp.isfile(content_path) and osp.isfile(style_path) and osp.isfile(target_path):
                self.samples.append((content_path, style_path, target_path))

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        content_path, style_path, target_path = self.samples[idx]
        try:
            content_img = pil_loader(content_path)
            style_img   = pil_loader(style_path)
            target_img  = pil_loader(target_path)
        except Exception as e:
            print(f"[ImagePulse Error] idx {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            content_img = self.transform(content_img)
            style_img   = self.transform(style_img)
            target_img  = self.transform(target_img)
        return target_img, style_img, content_img


class CurriculumMixedDataset(Dataset):
    """
    Dynamically mixes *old_dataset* and *new_dataset* according to
    ``new_data_ratio``.  In ``__getitem__``, each sample is drawn from the
    new dataset with probability ``new_data_ratio`` (rest from old).

    Call :meth:`set_new_data_ratio` between epochs to implement the
    curriculum schedule.

    If only one dataset is provided the ratio is ignored and all samples
    come from the available dataset.
    """

    def __init__(self, old_dataset=None, new_dataset=None,
                 new_data_ratio: float = 0.1):
        assert old_dataset is not None or new_dataset is not None, \
            "At least one dataset must be provided"
        self.old_dataset = old_dataset
        self.new_dataset = new_dataset
        self._new_data_ratio = new_data_ratio
        # effective length = max of both (wrapping via modulo)
        old_len = len(old_dataset) if old_dataset is not None else 0
        new_len = len(new_dataset) if new_dataset is not None else 0
        self._len = max(old_len, new_len, 1)
        print(f"[CurriculumMixed] old={old_len}, new={new_len}, "
              f"effective_len={self._len}, init_ratio={new_data_ratio:.2f}")

    @property
    def new_data_ratio(self):
        return self._new_data_ratio

    def set_new_data_ratio(self, ratio: float):
        self._new_data_ratio = max(0.0, min(1.0, ratio))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # --- decide which dataset to sample from ---
        use_new = False
        if self.old_dataset is None:
            use_new = True
        elif self.new_dataset is None:
            use_new = False
        else:
            use_new = (random.random() < self._new_data_ratio)

        if use_new:
            return self.new_dataset[idx % len(self.new_dataset)]
        else:
            return self.old_dataset[idx % len(self.old_dataset)]


def build_curriculum_dataset(
    old_data_path: str,
    new_data_path: str,
    new_data_tar_dir: str,
    final_reso: int,
    start_ratio: float = 0.1,
    hflip=False, mid_reso=1.125,
):
    """
    Build a :class:`CurriculumMixedDataset` for mixed fine-tuning.

    *   ``old_data_path`` – root of OmniStyle-150K (may be empty / missing).
    *   ``new_data_path`` – root of *extracted* ImagePulse data.
    *   ``new_data_tar_dir`` – directory with raw ``.tar.gz`` files.
        If ``new_data_path`` is empty or does not exist, tars are
        auto-extracted to ``<new_data_tar_dir>/../imagepulse_extracted``.
    """
    common_transform = transforms.Compose([
        transforms.Resize((final_reso, final_reso),
                          interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])

    from torch.utils.data import random_split

    # ---- Old dataset (OmniStyle) ----
    old_train = old_val = None
    old_has_data = (old_data_path
                    and (osp.isdir(osp.join(old_data_path, "target"))
                         or osp.islink(osp.join(old_data_path, "target"))))
    if old_has_data:
        train_fl = _read_file_list(osp.join(old_data_path, "train_files.txt"))
        val_fl = _read_file_list(osp.join(old_data_path, "val_files.txt"))
        if train_fl is not None and val_fl is not None:
            old_train = StyleTransferDataset(root_dir=old_data_path,
                                             transform=common_transform,
                                             file_list=train_fl)
            old_val = StyleTransferDataset(root_dir=old_data_path,
                                           transform=common_transform,
                                           file_list=val_fl)
        else:
            full_old = StyleTransferDataset(root_dir=old_data_path, transform=None)
            n = len(full_old)
            val_n = max(int(n * 0.05), 1)
            tr_sub, va_sub = random_split(full_old, [n - val_n, val_n],
                                          generator=torch.Generator().manual_seed(42))
            old_train = SubsetWrapper(tr_sub, common_transform)
            old_val = SubsetWrapper(va_sub, common_transform)
        print(f"[Curriculum] OmniStyle  train={len(old_train)}, val={len(old_val)}")
    else:
        print(f"[Curriculum] OmniStyle data NOT found at {old_data_path} — skipped")

    # ---- New dataset (ImagePulse) ----
    new_train = new_val = None
    # Auto-extract tar.gz if needed
    if (not new_data_path or not osp.isdir(new_data_path)) and \
       new_data_tar_dir and osp.isdir(new_data_tar_dir):
        new_data_path = osp.join(osp.dirname(new_data_tar_dir),
                                 "imagepulse_extracted")
        extract_imagepulse_tars(new_data_tar_dir, new_data_path)

    new_has_data = (new_data_path and osp.isdir(new_data_path))
    if new_has_data:
        # Prefer organised layout (content/style/target + file lists)
        organised_target = osp.join(new_data_path, "target")
        train_fl = _read_file_list(osp.join(new_data_path, "train_files.txt"))
        val_fl = _read_file_list(osp.join(new_data_path, "val_files.txt"))

        if osp.isdir(organised_target) and train_fl and val_fl:
            # Use StyleTransferDataset with organised layout
            new_train = StyleTransferDataset(root_dir=new_data_path,
                                             transform=common_transform,
                                             file_list=train_fl)
            new_val = StyleTransferDataset(root_dir=new_data_path,
                                           transform=common_transform,
                                           file_list=val_fl)
            print(f"[Curriculum] ImagePulse train={len(new_train)}, "
                  f"val={len(new_val)}")
        else:
            # Fallback: use ImagePulseDataset with raw extracted layout
            full_new = ImagePulseDataset(root_dir=new_data_path, transform=None)
            if len(full_new) > 0:
                n = len(full_new)
                val_n = max(int(n * 0.05), 1)
                tr_sub, va_sub = random_split(
                    full_new, [n - val_n, val_n],
                    generator=torch.Generator().manual_seed(42))
                new_train = SubsetWrapper(tr_sub, common_transform)
                new_val = SubsetWrapper(va_sub, common_transform)
                print(f"[Curriculum] ImagePulse (raw) train={len(new_train)}, "
                      f"val={len(new_val)}")
            else:
                print("[Curriculum] ImagePulse indexed 0 samples — skipped")
    else:
        print(f"[Curriculum] ImagePulse data NOT found at {new_data_path} — skipped")

    # ---- Combine ----
    train_set = CurriculumMixedDataset(
        old_dataset=old_train, new_dataset=new_train,
        new_data_ratio=start_ratio)

    # Validation: prefer new data if available, else old
    val_set = new_val if new_val is not None else old_val
    if val_set is None:
        raise RuntimeError("No validation data available from either dataset!")

    print(f"[Curriculum] Final  train_len={len(train_set)}, val_len={len(val_set)}")
    return 1000, train_set, val_set