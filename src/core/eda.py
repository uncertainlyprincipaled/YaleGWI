import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import DataManager
from config import CFG
from pathlib import Path
from typing import Dict, List, Tuple
from src.core.setup import setup_environment  # Ensure correct paths
setup_environment()  # Set up paths before any EDA code

def summarize_array(arr):
    # Convert to float32 for statistics to avoid overflow
    arr_float32 = arr.astype(np.float32)
    stats = {
        'min': np.nanmin(arr_float32),
        'max': np.nanmax(arr_float32),
        'mean': np.nanmean(arr_float32),
        'std': np.nanstd(arr_float32),
        'nan_count': np.isnan(arr).sum(),
        'inf_count': np.isinf(arr).sum(),
        'shape': arr.shape
    }
    return stats

def eda_on_family(family, n_shape_samples=3):
    dm = DataManager()
    seis_files, vel_files, family_type = dm.list_family_files(family)
    seis_stats = []
    vel_stats = []
    seis_shapes = []
    vel_shapes = []
    for idx, (sfile, vfile) in enumerate(zip(seis_files, vel_files)):
        if idx >= 3:  # Only process first 3 files per family
            break
        sdata = np.load(sfile, mmap_mode='r')
        vdata = np.load(vfile, mmap_mode='r')
        # Handle both batched and per-sample files
        if sdata.ndim == 3:  # (sources, receivers, timesteps)
            seis_stats.append(summarize_array(sdata))
            vel_stats.append(summarize_array(vdata))
            seis_shapes.append(sdata.shape)
            vel_shapes.append(vdata.shape)
        elif sdata.ndim == 4:  # (batch, sources, receivers, timesteps)
            n_samples = sdata.shape[0]
            n_pick = min(10, n_samples)
            if n_samples > 1:
                pick_indices = np.linspace(0, n_samples - 1, n_pick, dtype=int)
            else:
                pick_indices = [0]
            for i in pick_indices:
                seis_stats.append(summarize_array(sdata[i]))
                vel_stats.append(summarize_array(vdata[i]))
                seis_shapes.append(sdata[i].shape)
                vel_shapes.append(vdata[i].shape)
        else:
            print(f"Unexpected shape for {sfile}: {sdata.shape}")
        # Print a few sample shapes for validation
        if idx < n_shape_samples:
            print(f"Sample {idx} - Seis shape: {sdata.shape}, Vel shape: {vdata.shape}")
    # Shape validation
    expected_seis_shapes = [(5, 72, 72), (500, 5, 72, 72)]  # Updated to match actual shapes
    expected_vel_shapes = [(1, 70, 70), (500, 1, 70, 70)]  # Updated to match actual shapes
    for shape in seis_shapes[:n_shape_samples]:
        if shape not in expected_seis_shapes:
            print(f"[!] Unexpected seis shape: {shape} in family {family}")
    for shape in vel_shapes[:n_shape_samples]:
        if shape not in expected_vel_shapes:
            print(f"[!] Unexpected vel shape: {shape} in family {family}")
    return seis_stats, vel_stats

def print_summary(stats, name):
    print(f"Summary for {name}:")
    for k in ['min', 'max', 'mean', 'std', 'nan_count', 'inf_count']:
        vals = [s[k] for s in stats]
        if k in ['nan_count', 'inf_count']:
            print(f"  {k}: min={np.min(vals)}, max={np.max(vals)}, mean={np.mean(vals)}, sum={np.sum(vals)}")
        else:
            # Use float32 for numerical stability
            vals = np.array(vals, dtype=np.float32)
            print(f"  {k}: min={vals.min()}, max={vals.max()}, mean={vals.mean()}, sum={vals.sum()}")
    print(f"  Total samples: {len(stats)}")

def plot_family_distributions(family_stats: Dict[str, Tuple[List, List]]):
    """Plot distributions of key statistics across families."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Statistics Across Families')
    
    # Velocity ranges
    vel_ranges = []
    family_names = []
    for family, (_, vel_stats) in family_stats.items():
        vel_mins = [s['min'] for s in vel_stats]
        vel_maxs = [s['max'] for s in vel_stats]
        vel_ranges.append((np.mean(vel_mins), np.mean(vel_maxs)))
        family_names.append(family)
    
    # Plot velocity ranges
    vel_mins, vel_maxs = zip(*vel_ranges)
    axes[0,0].bar(family_names, vel_mins, label='Min Velocity')
    axes[0,0].bar(family_names, vel_maxs, bottom=vel_mins, label='Max Velocity')
    axes[0,0].set_title('Velocity Ranges by Family')
    axes[0,0].set_ylabel('Velocity (m/s)')
    axes[0,0].legend()
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
    
    # Seismic amplitude ranges
    seis_ranges = []
    for family, (seis_stats, _) in family_stats.items():
        seis_mins = [s['min'] for s in seis_stats]
        seis_maxs = [s['max'] for s in seis_stats]
        seis_ranges.append((np.mean(seis_mins), np.mean(seis_maxs)))
    
    # Plot seismic ranges
    seis_mins, seis_maxs = zip(*seis_ranges)
    axes[0,1].bar(family_names, seis_mins, label='Min Amplitude')
    axes[0,1].bar(family_names, seis_maxs, bottom=seis_mins, label='Max Amplitude')
    axes[0,1].set_title('Seismic Amplitude Ranges by Family')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].legend()
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
    
    # Velocity standard deviations
    vel_stds = []
    for family, (_, vel_stats) in family_stats.items():
        stds = [s['std'] for s in vel_stats]
        vel_stds.append(np.mean(stds))
    
    axes[1,0].bar(family_names, vel_stds)
    axes[1,0].set_title('Average Velocity Standard Deviation by Family')
    axes[1,0].set_ylabel('Standard Deviation (m/s)')
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
    
    # Sample counts
    sample_counts = []
    for family, (seis_stats, _) in family_stats.items():
        sample_counts.append(len(seis_stats))
    
    axes[1,1].bar(family_names, sample_counts)
    axes[1,1].set_title('Number of Samples by Family')
    axes[1,1].set_ylabel('Sample Count')
    plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    return fig

def analyze_family_correlations(family_stats: Dict[str, Tuple[List, List]]):
    """Analyze correlations between seismic and velocity statistics."""
    correlations = {}
    for family, (seis_stats, vel_stats) in family_stats.items():
        # Extract statistics
        seis_means = [s['mean'] for s in seis_stats]
        seis_stds = [s['std'] for s in seis_stats]
        vel_means = [v['mean'] for v in vel_stats]
        vel_stds = [v['std'] for v in vel_stats]
        
        # Calculate correlations
        correlations[family] = {
            'mean_corr': np.corrcoef(seis_means, vel_means)[0,1],
            'std_corr': np.corrcoef(seis_stds, vel_stds)[0,1]
        }
    
    return correlations

def extract_and_plot_geometry(family_stats: Dict[str, Tuple[List, List]]):
    """
    Attempt to extract and plot receiver/source geometry for a few representative families only.
    Assumes seismic data shape is (sources, receivers, timesteps) or (sources, timesteps, receivers).
    """
    print("\nReceiver/Source Geometry per Family:")
    # Only plot for the first 2 families
    max_plots = 2
    plotted = 0
    for family, (seis_stats, _) in family_stats.items():
        if plotted >= max_plots:
            print(f"(Skipping plots for remaining families...)")
            break
        if len(seis_stats) == 0:
            print(f"{family}: No samples found.")
            continue
        shape = seis_stats[0]['shape']
        if len(shape) == 3:
            n_sources, n_receivers, n_timesteps = shape
            print(f"{family}: sources={n_sources}, receivers={n_receivers}, timesteps={n_timesteps}")
            plt.figure()
            plt.title(f"{family} - Receiver/Source Geometry")
            plt.scatter(range(n_receivers), [0]*n_receivers, label='Receivers', marker='x')
            plt.scatter([0]*n_sources, range(n_sources), label='Sources', marker='o')
            plt.xlabel('Receiver Index')
            plt.ylabel('Source Index')
            plt.legend()
            plt.show()
            plt.close()
            plotted += 1
        elif len(shape) == 2:
            n1, n2 = shape
            print(f"{family}: 2D seismic shape: {shape}")
        else:
            print(f"{family}: Unexpected seismic shape: {shape}")


def summarize_array_shapes(family_stats: Dict[str, Tuple[List, List]]):
    """
    Summarize array shapes for seismic and velocity data per family.
    """
    print("\nArray Shape Summary per Family:")
    for family, (seis_stats, vel_stats) in family_stats.items():
        seis_shapes = [tuple(s['shape']) for s in seis_stats]
        vel_shapes = [tuple(s['shape']) for s in vel_stats]
        unique_seis_shapes = set(seis_shapes)
        unique_vel_shapes = set(vel_shapes)
        print(f"{family}: Seismic shapes: {unique_seis_shapes}")
        print(f"{family}: Velocity shapes: {unique_vel_shapes}")
        if len(unique_seis_shapes) > 1 or len(unique_vel_shapes) > 1:
            print(f"  [!] Shape inconsistency detected in {family}")


def summarize_family_sizes(family_stats: Dict[str, Tuple[List, List]]):
    """
    Print a summary of family sizes and highlight imbalances.
    """
    print("\nFamily Size Summary:")
    sizes = {family: len(seis_stats) for family, (seis_stats, _) in family_stats.items()}
    for family, size in sizes.items():
        print(f"{family}: {size} samples")
    min_size = min(sizes.values())
    max_size = max(sizes.values())
    print(f"\nSmallest family: {min_size} samples, Largest family: {max_size} samples")
    print("Families with < 10 samples:")
    for family, size in sizes.items():
        if size < 10:
            print(f"  [!] {family}: {size} samples (very small)")

# --- New: Supplement/Downsample Check ---
def check_balancing_requirements(target_count=1000):
    """
    For each family, print base and OpenFWI sample counts, and how many to supplement or downsample.
    """
    print("\n=== Data Balancing Requirements ===")
    from config import CFG
    import os
    # Detect OpenFWI path
    openfwi_path = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')
    base_families = list(CFG.paths.families.keys())
    # If OpenFWI is not available, skip
    if not openfwi_path.exists():
        print("OpenFWI dataset not found. Skipping balancing check.")
        return
    print(f"Target samples per family: {target_count}")
    print(f"{'Family':<15} {'Base':>6} {'OpenFWI':>8} {'To Add':>8} {'To Down':>8}")
    print("-"*50)
    for family in base_families:
        # Count base samples
        dm = DataManager()
        base_seis, base_vel, _ = dm.list_family_files(family)
        base_count = len(base_seis)
        # Count OpenFWI samples
        openfwi_family = openfwi_path / family
        openfwi_seis = sorted(openfwi_family.glob('seis*.npy')) if openfwi_family.exists() else []
        openfwi_count = len(openfwi_seis)
        # Compute how many to add or downsample
        to_add = max(0, target_count - base_count)
        to_down = max(0, (base_count + openfwi_count) - target_count) if (base_count + openfwi_count) > target_count else 0
        print(f"{family:<15} {base_count:>6} {openfwi_count:>8} {to_add:>8} {to_down:>8}")
    print("-"*50)

def count_samples_base(base_path, family):
    fam_dir = base_path / family
    # Check for data/ subfolder
    data_dir = fam_dir / 'data' if (fam_dir / 'data').exists() else fam_dir
    seis_files = sorted(data_dir.glob('seis*.npy'))
    total_samples = 0
    for f in seis_files:
        arr = np.load(f, mmap_mode='r')
        # If batched, count first dimension; else count as 1
        n = arr.shape[0] if arr.ndim > 2 else 1
        total_samples += n
    return total_samples

def count_samples_openfwi(openfwi_path, family):
    fam_dir = openfwi_path / family
    seis_files = sorted(fam_dir.glob('seis*.npy'))
    return len(seis_files)

def main():
    from config import CFG
    families = list(CFG.paths.families.keys())
    family_stats = {}
    
    for family in families:
        print(f"\n--- EDA for family: {family} ---")
        seis_stats, vel_stats = eda_on_family(family)
        family_stats[family] = (seis_stats, vel_stats)
        print_summary(seis_stats, f"{family} seis")
        print_summary(vel_stats, f"{family} vel")
    
    # Generate distribution plots
    fig = plot_family_distributions(family_stats)
    plt.show()
    
    # Analyze correlations
    correlations = analyze_family_correlations(family_stats)
    print("\nCorrelations between seismic and velocity statistics:")
    for family, corrs in correlations.items():
        print(f"{family}:")
        print(f"  Mean correlation: {corrs['mean_corr']:.3f}")
        print(f"  Std correlation: {corrs['std_corr']:.3f}")

    # New EDA: Geometry, shape, and size summaries
    extract_and_plot_geometry(family_stats)
    summarize_array_shapes(family_stats)
    summarize_family_sizes(family_stats)

    # --- Print explicit sample counts for each family ---
    print("\nSample counts per family (Base and OpenFWI):")
    base_path = CFG.paths.train.parent  # This should point to 'train_samples'
    openfwi_path = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')
    print(f"{'Family':<15} {'Base':>6} {'OpenFWI':>8}")
    print("-"*32)
    for fam in families:
        base_count = count_samples_base(base_path, fam)
        openfwi_count = count_samples_openfwi(openfwi_path, fam)
        print(f"{fam:<15} {base_count:>6} {openfwi_count:>8}")
    print("-"*32)

    # Check balancing requirements
    check_balancing_requirements()

if __name__ == "__main__":
    main() 