import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
from src.core.config import CFG

def split_family_to_samples(family, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob('*.npy'))
    sample_count = 0
    manifest = []
    for f in tqdm(files, desc=f"Splitting {family}"):
        arr = np.load(f, mmap_mode='r')
        if arr.ndim == 4:
            n_samples = arr.shape[0]
            for i in range(n_samples):
                sample = arr[i]
                out_path = output_dir / f"sample_{sample_count:06d}.npy"
                np.save(out_path, sample)
                manifest.append(str(out_path))
                sample_count += 1
        elif arr.ndim == 3:
            out_path = output_dir / f"sample_{sample_count:06d}.npy"
            np.save(out_path, arr)
            manifest.append(str(out_path))
            sample_count += 1
        else:
            print(f"[!] Unexpected shape {arr.shape} in {f}")
    return manifest

def transfer_to_host23(local_dir, remote_user, remote_host, remote_path):
    """Transfer local_dir to remote_host:remote_path using rsync with progress and error handling."""
    local_dir = Path(local_dir)
    remote = f"{remote_user}@{remote_host}:{remote_path}"
    cmd = [
        "rsync", "-avz", "--progress", str(local_dir) + "/", remote + "/"
    ]
    print(f"Transferring {local_dir} to {remote} ...")
    try:
        subprocess.run(cmd, check=True)
        print("Transfer complete.")
    except subprocess.CalledProcessError as e:
        print(f"[!] Transfer failed: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Split large .npy files into per-sample files for distributed training and optionally transfer to another host.")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/mnt/waveform-inversion/fine/train_samples', help='Output directory for per-sample files')
    parser.add_argument('--transfer', action='store_true', help='Transfer output to remote host after splitting')
    parser.add_argument('--remote_user', type=str, default=None, help='Remote username for transfer')
    parser.add_argument('--remote_host', type=str, default=None, help='Remote host for transfer')
    parser.add_argument('--remote_path', type=str, default=None, help='Remote path for transfer')
    args = parser.parse_args()

    if not args.transfer:
        families = list(CFG.paths.families.keys())
        all_manifests = {}
        for family in families:
            input_dir = Path(args.input_root) / family
            output_dir = Path(args.output_root) / family
            manifest = split_family_to_samples(family, input_dir, output_dir)
            all_manifests[family] = manifest
            print(f"Family {family}: {len(manifest)} samples written to {output_dir}")
        # Write manifest files
        for family, manifest in all_manifests.items():
            manifest_path = Path(args.output_root) / family / 'manifest.txt'
            with open(manifest_path, 'w') as f:
                for line in manifest:
                    f.write(f"{line}\n")
        print("All families processed. Per-sample files and manifests created.")
    else:
        if not (args.remote_user and args.remote_host and args.remote_path):
            print("[!] Must specify --remote_user, --remote_host, and --remote_path for transfer.")
            exit(1)
        transfer_to_host23(args.output_root, args.remote_user, args.remote_host, args.remote_path)

if __name__ == "__main__":
    main() 