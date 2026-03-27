import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def scan_and_verify_images(directory_path):
    """
    扫描目录下的图片文件，尝试用 PIL 打开并转换为 Tensor。
    如果失败则标记为 False，成功则标记为 True。
    """
    # 常见的图片后缀名，你可以根据需要增删
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # 转换为 Path 对象，方便操作
    target_dir = Path(directory_path)
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"错误: 目录 {directory_path} 不存在或不是一个文件夹。")
        return

    # 初始化转换器
    to_tensor = transforms.ToTensor()
    
    # 统计信息
    total_files = 0
    valid_files = 0
    corrupted_files = 0

    print(f"开始扫描目录: {target_dir.resolve()}")
    print("-" * 50)

    # rglob('*') 会递归扫描所有子目录。如果只需要扫当前目录，可以改为 glob('*')
    for file_path in target_dir.rglob('*'):
        # 只处理文件，且后缀名在我们的预期列表内
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            total_files += 1
            is_valid = False
            
            try:
                # 1. 尝试用 PIL 打开图片
                img = Image.open(file_path)
                
                # 2. 强制加载图片数据 (Image.open 是懒加载，调用 load 才能真正捕获文件损坏)
                img.load()
                
                # 有些模式（如 RGBA 或 P）在转换为 Tensor 时可能会出现警告或意外，
                # 如果你的模型只接受 RGB，建议在这里加上 img = img.convert('RGB')
                
                # 3. 尝试转换为 Tensor
                tensor = to_tensor(img)
                
                # 如果以上都没有报错，说明图片是正常的
                is_valid = True
                
            except Exception as e:
                # 捕获所有异常（如 UnidentifiedImageError, OSError, 或 Tensor 转换错误）
                is_valid = False
                error_msg = str(e)
            
            # 根据结果输出 True / False 并执行相应逻辑
            if is_valid:
                valid_files += 1
                print(f"[True]  有效图片: {file_path}")
            else:
                corrupted_files += 1
                print(f"[False] 损坏或无法加载: {file_path} (错误信息: {error_msg})")
                
                # =======================================================
                # 危险区域：实际删除文件的代码已注释
                # 确认输出无误后，可以取消下面这行代码的注释来执行物理删除
                # =======================================================
                file_path.unlink() 
                # =======================================================

    print("-" * 50)
    print("扫描完成！")
    print(f"总计检查图片: {total_files} 张")
    print(f"有效图片 (True): {valid_files} 张")
    print(f"无效图片 (False): {corrupted_files} 张")

if __name__ == "__main__":
    # 请将这里的路径替换为你需要扫描的实际目录路径
    # 例如: test_directory = "./my_dataset"
    test_directory = "/home/linux/StyleVAR/data/OmniStyle-150k/OmniStyle-150K" 
    
    scan_and_verify_images(test_directory)