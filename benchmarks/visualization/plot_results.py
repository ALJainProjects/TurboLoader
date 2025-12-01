#!/usr/bin/env python3
"""
Benchmark Visualization

Generates charts and visualizations from benchmark results.
Requires: matplotlib, pandas, seaborn
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Color scheme
COLORS = {
    'turboloader': '#2ecc71',  # Green
    'turboloader_pipe': '#27ae60',  # Darker green
    'pytorch': '#3498db',  # Blue
    'pytorch_cached': '#2980b9',  # Darker blue
    'webdataset': '#e74c3c',  # Red
    'torchvision': '#9b59b6',  # Purple
    'albumentations': '#f39c12',  # Orange
}


def load_json(path: str) -> Any:
    """Load JSON file"""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def plot_throughput_comparison(results_dir: str, output_path: str):
    """Plot throughput comparison bar chart"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plots")
        return

    # Load results
    turboloader_path = os.path.join(results_dir, 'throughput/turboloader.json')
    pytorch_path = os.path.join(results_dir, 'throughput/pytorch.json')

    turboloader_data = load_json(turboloader_path) or []
    pytorch_data = load_json(pytorch_path) or []

    # Extract throughput values
    data = []
    for entry in turboloader_data:
        if entry.get('metric') == 'throughput':
            data.append({
                'Library': 'TurboLoader',
                'Batch Size': entry['config']['batch_size'],
                'Workers': entry['config']['num_workers'],
                'Throughput': entry['value']
            })

    for entry in pytorch_data:
        if entry.get('metric') == 'throughput':
            lib = 'PyTorch (cached)' if entry['config'].get('cached') else 'PyTorch'
            data.append({
                'Library': lib,
                'Batch Size': entry['config']['batch_size'],
                'Workers': entry['config']['num_workers'],
                'Throughput': entry['value']
            })

    if not data:
        print("No throughput data found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by batch size and workers
    if HAS_PANDAS:
        df = pd.DataFrame(data)

        # Get unique configurations
        configs = df.groupby(['Batch Size', 'Workers']).groups.keys()

        # For each config, plot bars
        x_labels = []
        turbo_vals = []
        pytorch_vals = []

        for batch, workers in sorted(configs):
            x_labels.append(f"B{batch}/W{workers}")

            turbo = df[(df['Library'] == 'TurboLoader') &
                      (df['Batch Size'] == batch) &
                      (df['Workers'] == workers)]['Throughput'].values
            turbo_vals.append(turbo[0] if len(turbo) > 0 else 0)

            pytorch = df[(df['Library'].str.contains('PyTorch')) &
                        (df['Batch Size'] == batch) &
                        (df['Workers'] == workers)]['Throughput'].values
            pytorch_vals.append(pytorch[0] if len(pytorch) > 0 else 0)

        x = range(len(x_labels))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], turbo_vals, width,
                       label='TurboLoader', color=COLORS['turboloader'])
        bars2 = ax.bar([i + width/2 for i in x], pytorch_vals, width,
                       label='PyTorch', color=COLORS['pytorch'])

        ax.set_xlabel('Configuration (Batch/Workers)')
        ax.set_ylabel('Throughput (images/sec)')
        ax.set_title('TurboLoader vs PyTorch DataLoader Throughput')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_transform_comparison(results_dir: str, output_path: str):
    """Plot transform performance comparison"""
    if not HAS_MATPLOTLIB:
        return

    transform_path = os.path.join(results_dir, 'transforms/transforms.json')
    data = load_json(transform_path)

    if not data:
        print("No transform data found")
        return

    # Group by transform
    transforms = {}
    for entry in data:
        name = entry['transform']
        lib = entry['library']
        time_ms = entry['time_per_image_ms']

        if name not in transforms:
            transforms[name] = {}
        transforms[name][lib] = time_ms

    # Filter to transforms that have both TurboLoader and torchvision
    common_transforms = {k: v for k, v in transforms.items()
                         if 'turboloader' in v and 'torchvision' in v}

    if not common_transforms:
        print("No common transforms to compare")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    names = list(common_transforms.keys())
    turbo_times = [common_transforms[n].get('turboloader', 0) for n in names]
    torch_times = [common_transforms[n].get('torchvision', 0) for n in names]

    x = range(len(names))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], turbo_times, width,
                   label='TurboLoader', color=COLORS['turboloader'])
    bars2 = ax.bar([i + width/2 for i in x], torch_times, width,
                   label='torchvision', color=COLORS['torchvision'])

    ax.set_xlabel('Transform')
    ax.set_ylabel('Time per Image (ms)')
    ax.set_title('Transform Performance: TurboLoader vs torchvision')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_chart(results_dir: str, output_path: str):
    """Plot speedup ratio chart"""
    if not HAS_MATPLOTLIB:
        return

    transform_path = os.path.join(results_dir, 'transforms/transforms.json')
    data = load_json(transform_path)

    if not data:
        return

    # Calculate speedups
    transforms = {}
    for entry in data:
        name = entry['transform']
        lib = entry['library']
        time_ms = entry['time_per_image_ms']

        if name not in transforms:
            transforms[name] = {}
        transforms[name][lib] = time_ms

    speedups = {}
    for name, libs in transforms.items():
        if 'turboloader' in libs and 'torchvision' in libs:
            turbo = libs['turboloader']
            torch = libs['torchvision']
            if turbo > 0:
                speedups[name] = torch / turbo

    if not speedups:
        return

    # Sort by speedup
    sorted_speedups = sorted(speedups.items(), key=lambda x: x[1], reverse=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [s[0] for s in sorted_speedups]
    values = [s[1] for s in sorted_speedups]

    colors = [COLORS['turboloader'] if v > 1 else '#e74c3c' for v in values]
    bars = ax.barh(names, values, color=colors)

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Speedup (x times faster)')
    ax.set_ylabel('Transform')
    ax.set_title('TurboLoader Speedup vs torchvision')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.1f}x',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_comparison(results_dir: str, output_path: str):
    """Plot memory usage comparison"""
    if not HAS_MATPLOTLIB:
        return

    memory_path = os.path.join(results_dir, 'memory/memory.json')
    data = load_json(memory_path)

    if not data:
        print("No memory data found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    libraries = []
    peak_memory = []
    delta_memory = []

    for entry in data:
        lib = entry['library']
        libraries.append(f"{lib} (B{entry['config']['batch_size']})")
        peak_memory.append(entry['peak_mb'])
        delta_memory.append(entry['delta_mb'])

    x = range(len(libraries))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], peak_memory, width,
                   label='Peak Memory', color=COLORS['turboloader'])
    bars2 = ax.bar([i + width/2 for i in x], delta_memory, width,
                   label='Delta Memory', color=COLORS['pytorch'])

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(libraries, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_scalability(results_dir: str, output_path: str):
    """Plot scalability chart (throughput vs workers)"""
    if not HAS_MATPLOTLIB:
        return

    all_results_path = os.path.join(results_dir, 'all_results.json')
    data = load_json(all_results_path)

    if not data or 'benchmarks' not in data:
        print("No scalability data found")
        return

    scalability = data['benchmarks'].get('Scalability', {})
    if not scalability:
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    workers = sorted([int(w) for w in scalability.keys()])
    throughputs = [scalability[str(w)] for w in workers]

    ax.plot(workers, throughputs, 'o-', color=COLORS['turboloader'],
            linewidth=2, markersize=10, label='TurboLoader')

    # Add ideal scaling line
    if throughputs:
        base = throughputs[0]
        ideal = [base * (w / workers[0]) for w in workers]
        ax.plot(workers, ideal, '--', color='gray', label='Ideal scaling')

    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput (images/sec)')
    ax.set_title('TurboLoader Scalability')
    ax.set_xticks(workers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(results_dir: str, output_dir: str):
    """Generate all visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    plot_throughput_comparison(
        results_dir,
        os.path.join(output_dir, 'throughput_comparison.png')
    )

    plot_transform_comparison(
        results_dir,
        os.path.join(output_dir, 'transform_comparison.png')
    )

    plot_speedup_chart(
        results_dir,
        os.path.join(output_dir, 'speedup_chart.png')
    )

    plot_memory_comparison(
        results_dir,
        os.path.join(output_dir, 'memory_comparison.png')
    )

    plot_scalability(
        results_dir,
        os.path.join(output_dir, 'scalability.png')
    )

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark visualizations')
    parser.add_argument('--results-dir', type=str, default='benchmarks/results',
                        help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default='benchmarks/results/plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualizations")
        print("Install with: pip install matplotlib pandas seaborn")
        sys.exit(1)

    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
