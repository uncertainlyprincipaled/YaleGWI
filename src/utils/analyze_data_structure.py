import os
import json
from pathlib import Path
import numpy as np

def analyze_family_structure(root_dir):
    summary = {}
    root = Path(root_dir)
    for family in sorted(root.iterdir()):
        if not family.is_dir():
            continue
        fam_info = {'files': [], 'subdirs': {}}
        # Top-level files
        for f in family.glob('*.npy'):
            try:
                arr = np.load(f, mmap_mode='r')
                fam_info['files'].append({
                    'name': f.name,
                    'shape': arr.shape,
                    'dtype': str(arr.dtype)
                })
            except Exception as e:
                fam_info['files'].append({'name': f.name, 'error': str(e)})
        # Subdirectories
        for sub in family.iterdir():
            if sub.is_dir():
                sub_info = []
                for f in sub.glob('*.npy'):
                    try:
                        arr = np.load(f, mmap_mode='r')
                        sub_info.append({
                            'name': f.name,
                            'shape': arr.shape,
                            'dtype': str(arr.dtype)
                        })
                    except Exception as e:
                        sub_info.append({'name': f.name, 'error': str(e)})
                fam_info['subdirs'][sub.name] = sub_info
        summary[family.name] = fam_info
    return summary

def get_project_root():
    """Get the project root directory in a way that works in both notebooks and scripts."""
    try:
        # Try to get the path from __file__ (works in scripts)
        return Path(__file__).parent.parent.parent
    except NameError:
        # If __file__ is not defined (in notebooks), use the current working directory
        return Path(os.getcwd())

if __name__ == '__main__':
    # Get the project root directory
    project_root = get_project_root()
    root_dir = project_root / 'train_samples'
    
    # Ensure the directory exists
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    summary = analyze_family_structure(root_dir)
    # Print summary
    for fam, info in summary.items():
        print(f'Family: {fam}')
        print('  Top-level files:')
        for f in info['files']:
            print(f"    {f}")
        for sub, files in info['subdirs'].items():
            print(f'  Subdir: {sub}')
            for f in files:
                print(f"    {f}")
        print()
    # Save summary
    out_path = project_root / 'src' / 'utils' / 'data_structure_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {out_path}") 