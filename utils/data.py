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
    def __init__(self, root_dir, transform=None):
        """
        root_dir: /home/OmniStyle-150K/
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.target_dir = osp.join(root_dir, 'target')
        self.style_dir = osp.join(root_dir, 'style')
        self.content_dir = osp.join(root_dir, 'content')
        self.transform = transform
        
        # 1. Scan all target files
        # Filename format: "content_name.png&&style_name.jpg"
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

def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations (Same as vanilla VAR)
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    
    # Common transform for training
    train_aug = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), 
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ])
    
    if hflip:
        train_aug.transforms.insert(0, transforms.RandomHorizontalFlip())

    # Common transform for validation (CenterCrop)
    val_aug = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), 
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ])
    
    # Assume structure: /home/OmniStyle-150K/
    # We split the list of files manually for train/val since they are in the same folder structure
    full_dataset = StyleTransferDataset(root_dir=data_path, transform=None)
    
    # Split dataset: 95% train, 5% val (or whatever logic you prefer)
    total_len = len(full_dataset)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    
    # Assign specific transforms to subsets
    # Note: Dataset subsets don't easily support different transforms per subset. 
    # We wrap them in a helper class or just use train_aug for both if acceptable, 
    # BUT best practice is a wrapper.
    
    class SubsetWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            t, s, c = self.subset[idx] # These are PIL images because transform=None above
            return self.transform(t), self.transform(s), self.transform(c)

    train_set = SubsetWrapper(train_subset, train_aug)
    val_set = SubsetWrapper(val_subset, val_aug)
    
    num_classes = 1000 
    print(f'[Dataset] Train: {len(train_set)}, Val: {len(val_set)}')
    
    return num_classes, train_set, val_set