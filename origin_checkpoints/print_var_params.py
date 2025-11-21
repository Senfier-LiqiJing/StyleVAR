#!/usr/bin/env python3
"""
Script to print all parameter names from original VAR checkpoint files
"""
import os
import torch
import sys

# Add project path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.style_var import StyleVAR


def extract_var_state_dict(raw_ckpt):
    """Extract VAR's state_dict from checkpoint using StyleVAR's static method"""
    return StyleVAR._extract_var_state_dict(raw_ckpt)


def print_params_from_ckpt(ckpt_path: str):
    """Load and print all parameter names from checkpoint file"""
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        return
    
    try:
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Extract VAR's state_dict
        var_state = extract_var_state_dict(ckpt)
        
        # Get all parameter names and sort them
        param_names = sorted(var_state.keys())
        
        print(f"Total number of parameters: {len(param_names)}\n")
        print("Parameter names:")
        print("-" * 80)
        
        for i, name in enumerate(param_names, 1):
            param = var_state[name]
            shape = tuple(param.shape) if hasattr(param, 'shape') else 'N/A'
            dtype = str(param.dtype) if hasattr(param, 'dtype') else 'N/A'
            numel = param.numel() if hasattr(param, 'numel') else 'N/A'
            print(f"{i:4d}. {name:60s} | shape: {str(shape):30s} | dtype: {dtype:10s} | numel: {numel}")
        
        print("-" * 80)
        print(f"\nTotal parameters: {len(param_names)}")
        
        # Calculate total parameter count
        total_numel = sum(p.numel() if hasattr(p, 'numel') else 0 for p in var_state.values())
        print(f"Total parameter count: {total_numel:,} ({total_numel/1e6:.2f}M)")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function: process all original VAR checkpoint files"""
    origin_checkpoints_dir = os.path.join(os.path.dirname(__file__), 'origin_checkpoints')
    
    # Find all VAR-related checkpoint files
    var_ckpt_files = []
    if os.path.exists(origin_checkpoints_dir):
        for fname in os.listdir(origin_checkpoints_dir):
            if fname.startswith('var_') and fname.endswith('.pth'):
                var_ckpt_files.append(os.path.join(origin_checkpoints_dir, fname))
    
    if not var_ckpt_files:
        print("No VAR checkpoint files found in origin_checkpoints directory.")
        print("Looking for files matching pattern: var_*.pth")
        return
    
    # Sort files
    var_ckpt_files.sort()
    
    # Process each checkpoint file
    for ckpt_path in var_ckpt_files:
        print_params_from_ckpt(ckpt_path)
        print("\n")


if __name__ == '__main__':
    main()

