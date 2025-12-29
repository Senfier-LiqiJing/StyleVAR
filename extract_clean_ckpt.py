import torch
import os
import datetime

now = datetime.datetime.now()
formatted_time = now.strftime("%m_%d_%H")

ckpt_path = "/home/PML-Project/local_output/ar-ckpt-best.pth"
save_path = "/home/PML-Project/checkpoints/style_var_d20_" + formatted_time + ".pth"

print(f"load file")
checkpoint = torch.load(ckpt_path, map_location='cpu')

if 'trainer' in checkpoint and 'var_wo_ddp' in checkpoint['trainer']:
    full_state_dict = checkpoint['trainer']['var_wo_ddp']
    print("✅ find model weight")
else:
    print("⚠️ not find model weight")
    print(checkpoint.keys())
    if 'trainer' in checkpoint:
        print("Trainer keys:", checkpoint['trainer'].keys())
    exit()

clean_state_dict = {}
# print("FP32->16")

for key, tensor in full_state_dict.items():
    clean_state_dict[key] = tensor

torch.save(clean_state_dict, save_path)

original_size = os.path.getsize(ckpt_path) / (1024**3)
new_size = os.path.getsize(save_path) / (1024**3)

print(f"\n🎉 Done")
print(f"Original: {original_size:.2f} GB")
print(f"New:   {new_size:.2f} GB")
print(f"Path: {save_path}")