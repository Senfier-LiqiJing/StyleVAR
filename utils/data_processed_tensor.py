import os
import glob
import torch
from torch.utils.data import Dataset

class ShardedTensorDataset(Dataset):
    """
    A custom dataset to load pre-processed tensors stored in sharded .pth files.
    Assuming all shards EXCEPT the last one have 'shard_size' items.
    The last shard can have fewer items.
    """
    def __init__(self, root_dir, shard_size=1024):
        super().__init__()
        self.root_dir = root_dir
        self.shard_size = shard_size
        
        # 1. Find all shard files
        self.shard_paths = sorted(glob.glob(os.path.join(root_dir, "shard_*.pth")))
        
        if len(self.shard_paths) == 0:
            print(f"WARNING: No .pth shards found in {root_dir}.")
            self.total_len = 0
        else:
            n_full_shards = len(self.shard_paths) - 1
            
            try:
                last_shard_path = self.shard_paths[-1]
                last_shard_data = torch.load(last_shard_path, map_location='cpu')
                last_shard_size = last_shard_data['content'].shape[0]
            except Exception as e:
                print(f"Error reading last shard {self.shard_paths[-1]}: {e}")
                last_shard_size = 0
                
            self.total_len = n_full_shards * self.shard_size + last_shard_size
            
            print(f"Found {len(self.shard_paths)} shards in {root_dir}.")
            print(f"Last shard size: {last_shard_size}. Total samples: {self.total_len}")

        self.last_shard_idx = -1
        self.cached_shard_data = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 1. Map global index to shard index and local index
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        
        if shard_idx >= len(self.shard_paths):
            raise IndexError(f"Global index {idx} out of range.")

        # 2. Load shard (Cache check)
        if shard_idx != self.last_shard_idx:
            # print(f"[DEBUG] Loading Shard {shard_idx}...", flush=True)
            self.cached_shard_data = torch.load(self.shard_paths[shard_idx], map_location='cpu')
            self.last_shard_idx = shard_idx
        
        data_dict = self.cached_shard_data
        
        # 3. Extract tensors
        content_img = data_dict['content'][local_idx]
        style_img = data_dict['style'][local_idx]
        target_img = data_dict['target'][local_idx]
        
        return target_img, style_img, content_img


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # Note: final_reso, hflip, mid_reso are ignored because 
    # data is already pre-processed. We keep them in the signature 
    # to avoid breaking the function call in fine_tune.py.

    print(f'[build_dataset] Scanning data in {data_path}...')
    
    # Assume structure:
    # /home/OmniStyle-150k-tensor/train/*.pth
    # /home/OmniStyle-150k-tensor/valid/*.pth
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'valid') # User specified 'valid', not 'val'
    
    train_set = ShardedTensorDataset(train_dir, shard_size=1024)
    val_set = ShardedTensorDataset(val_dir, shard_size=1024)
    
    num_classes = 1000 # Dummy value, not used by StyleVAR but required by return signature
    
    print(f'[Dataset] Train samples: {len(train_set)}, Valid samples: {len(val_set)}')
    
    return num_classes, train_set, val_set