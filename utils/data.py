import os
import os.path as osp
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
    """
    Dataset that reads <target> images and pairs them with their corresponding
    content and style images. Target filenames are expected to follow:
        {content_filename}&&{style_filename}.png
    where {style_filename} already contains its original extension (e.g. ".jpg.jpg").
    """
    def __init__(self, root_dir, transform=None, max_resample=10):
        self.root_dir = root_dir
        self.target_dir = osp.join(root_dir, 'target')
        self.style_dir = osp.join(root_dir, 'style')
        self.content_dir = osp.join(root_dir, 'content')
        self.transform = transform
        self.max_resample = max(1, int(max_resample))

        self.content_lookup = self._build_lookup(self.content_dir)
        self.style_lookup = self._build_lookup(self.style_dir)
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f'No valid triplets found under {root_dir}')

        print(f"[Dataset] Found {len(self.samples)} valid target/style/content triplets "
              f"(missing content: {self.missing_content}, missing style: {self.missing_style}).")

    @staticmethod
    def _build_lookup(directory):
        return {
            fname: osp.join(directory, fname)
            for fname in os.listdir(directory)
            if osp.isfile(osp.join(directory, fname))
        }

    def _build_samples(self):
        samples = []
        self.missing_content = 0
        self.missing_style = 0
        target_files = [
            f for f in os.listdir(self.target_dir)
            if '&&' in f and osp.isfile(osp.join(self.target_dir, f))
        ]
        target_files.sort()

        for target_file in target_files:
            content_name, style_token = target_file.split('&&', 1)
            style_name = style_token[:-4] if style_token.lower().endswith('.png') else style_token

            content_path = self.content_lookup.get(content_name)
            if content_path is None:
                self.missing_content += 1
                continue

            style_path = self.style_lookup.get(style_name)
            if style_path is None:
                self.missing_style += 1
                continue

            target_path = osp.join(self.target_dir, target_file)
            samples.append((target_path, style_path, content_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trials = 0
        cur_idx = idx % len(self.samples)
        last_error = None

        while trials < self.max_resample:
            target_path, style_path, content_path = self.samples[cur_idx]
            try:
                target_img = pil_loader(target_path)
                style_img = pil_loader(style_path)
                content_img = pil_loader(content_path)

                if self.transform:
                    target_img = self.transform(target_img)
                    style_img = self.transform(style_img)
                    content_img = self.transform(content_img)

                return target_img, style_img, content_img
            except Exception as exc:  # corrupted triplet
                last_error = exc
                print(f"[Dataset Warning] Skip corrupt triplet idx={cur_idx}: {exc}")
                trials += 1
                cur_idx = (cur_idx + 1) % len(self.samples)

        raise RuntimeError(f"Failed to fetch a valid triplet after {self.max_resample} attempts "
                           f"(last error: {last_error})")

class SubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        t, s, c = self.subset[idx] # These are PIL images because transform=None above
        return self.transform(t), self.transform(s), self.transform(c)

def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations similar to the vanilla VAR recipe
    mid_reso = round(mid_reso * final_reso)
    train_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())

    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]

    train_aug = transforms.Compose(train_aug)
    val_aug = transforms.Compose(val_aug)

    base_dataset = StyleTransferDataset(root_dir=data_path, transform=None)
    total_len = len(base_dataset)
    if total_len == 0:
        raise RuntimeError(f"No samples were found under {data_path} to build datasets.")

    from torch.utils.data import random_split, Subset

    if total_len == 1:
        train_subset = Subset(base_dataset, [0])
        val_subset = Subset(base_dataset, [])
    else:
        val_len = max(1, int(total_len * 0.05))
        val_len = min(val_len, total_len - 1)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(
            base_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

    train_set = SubsetWrapper(train_subset, train_aug)
    val_set = SubsetWrapper(val_subset, val_aug)

    num_classes = 1000
    print(f'[Dataset] len(train_set)={len(train_set)}, len(val_set)={len(val_set)}, num_classes={num_classes}')
    return num_classes, train_set, val_set
