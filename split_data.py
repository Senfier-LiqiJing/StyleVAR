import os
import os.path as osp
import torch
from torch.utils.data import Dataset, random_split

# 配置路径 (根据你的 arg_util.py)
DATA_PATH = '/home/OmniStyle-150K'
OUTPUT_DIR = './split_info'  # 输出结果保存的文件夹

class FilenameDataset(Dataset):
    """
    这是一个轻量级的 Dataset 类，只负责读取文件名，不读取图片数据。
    逻辑严格复刻 utils/data.py 中的 StyleTransferDataset。
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.target_dir = osp.join(root_dir, 'target')
        
        if not os.path.exists(self.target_dir):
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        # --- 严格复刻 data.py 的扫描逻辑 ---
        # 原代码: self.target_files = [f for f in os.listdir(self.target_dir) if '&&' in f]
        # 注意：os.listdir 的顺序取决于文件系统，但在同一台机器未变动文件的情况下通常是稳定的。
        # 如果为了绝对严谨，原 data.py 应该加上 sorted()，但为了复现你现在的状态，这里必须保持原样不加 sort。
        self.target_files = [f for f in os.listdir(self.target_dir) if '&&' in f]
        
    def __len__(self):
        return len(self.target_files)
    
    def __getitem__(self, idx):
        # 只返回文件名
        return self.target_files[idx]

def export_splits():
    print(f"🔍 Scanning directory: {DATA_PATH} ...")
    
    # 1. 初始化数据集
    full_dataset = FilenameDataset(DATA_PATH)
    total_len = len(full_dataset)
    
    if total_len == 0:
        print("❌ Error: No files found!")
        return

    # 2. 复刻划分逻辑 (Copy from utils/data.py)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    
    print(f"📊 Total files: {total_len}")
    print(f"   Training set size:   {train_len}")
    print(f"   Validation set size: {val_len}")
    print(f"   Random Seed:         42")

    # 3. 执行随机划分
    # 这里必须使用和训练时完全一样的 Generator 和 Seed
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. 保存文件列表
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_to_file(subset, fname):
        output_path = os.path.join(OUTPUT_DIR, fname)
        print(f"💾 Saving {fname} ...", end="")
        with open(output_path, 'w', encoding='utf-8') as f:
            # subset.indices 包含了随机划分后的索引列表
            for idx in subset.indices:
                # 通过索引去原始数据集中拿文件名
                filename = full_dataset.target_files[idx]
                f.write(filename + '\n')
        print(f" Done! Saved to {output_path}")

    save_to_file(train_subset, "train_files.txt")
    save_to_file(val_subset, "val_files.txt")
    
    print("\n✅ All done.")

if __name__ == '__main__':
    export_splits()