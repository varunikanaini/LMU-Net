# /kaggle/working/LMU-Net/check_splits.py

import sys
import os
import numpy as np

# --- Setup Project Path ---
# This ensures the script can find your config and datasets modules
project_path = '/kaggle/working/LMU-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

try:
    import config
    from datasets import make_dataset
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure this script is in the same directory as config.py and datasets.py.")
    print(f"Details: {e}")
    sys.exit(1)

def analyze_splits():
    """
    Iterates through all datasets in the config and prints the train/val/test split counts.
    """
    print("=" * 70)
    print("Analyzing Dataset Splits based on your finalized datasets.py logic...")
    print("=" * 70)

    # Get the dataset configuration dictionary
    all_datasets = config.DATASET_CONFIG

    for name, dataset_config in all_datasets.items():
        root_path = dataset_config.get('path')
        
        if not os.path.exists(root_path):
            print(f"-> Dataset '{name}': SKIPPED (Path not found: {root_path})")
            continue

        # Use the make_dataset function to get the file lists for each split
        # This will automatically apply the splitting logic from your datasets.py
        train_files = make_dataset(root_path, name, split='train')
        val_files = make_dataset(root_path, name, split='val')
        test_files = make_dataset(root_path, name, split='test')

        # Get the counts
        num_train = len(train_files)
        num_val = len(val_files)
        num_test = len(test_files)
        total = num_train + num_val + num_test
        
        # In case the splitting logic doesn't produce an exact sum (e.g., for PRE_SPLIT)
        # we recalculate the total more robustly
        if dataset_config.get('structure') == 'PRE_SPLIT':
             # For pre-split, the total is just the sum of what's found
             total_found = len(make_dataset(root_path, name, split='train')) + len(make_dataset(root_path, name, split='test'))
             print(f"-> Dataset '{name}' (Pre-Split):")
             print(f"   Train: {num_train:<5} | Test: {num_test:<5} | Total Found: {total_found}")

        else: # For FLAT_SPLIT and MONTGOMERY
             print(f"-> Dataset '{name}' (Auto-Split):")
             print(f"   Train: {num_train:<5} | Val: {num_val:<5} | Test: {num_test:<5} | Total: {total}")
        
        print("-" * 70)

if __name__ == '__main__':
    analyze_splits()