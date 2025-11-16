#!/usr/bin/env python3
"""
Generate Beautiful Plots from TurboLoader Benchmark Results

Creates publication-ready figures:
- Throughput comparison bar charts
- Speedup charts
- Per-data-type breakdown
- Time series performance
- CPU utilization plots
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Set publication-quality defaults
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5

# Color scheme
TURBO_COLOR = '#FF6B35'  # Orange for TurboLoader
PYTORCH_COLOR = '#4A90E2'  # Blue for PyTorch
ACCENT_COLOR = '#50E3C2'  # Teal for highlights

def load_results(results_file):
    """Load benchmark results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_throughput_comparison(results, output_dir):
    """Create throughput comparison bar chart"""
    print("Creating throughput comparison plot...")

    datasets = list(results.keys())
    turbo_throughput = [results[d]['turboloader']['throughput'] for d in datasets]
    pytorch_throughput = [results[d]['pytorch']['throughput'] for d in datasets]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, turbo_throughput, width, label='TurboLoader',
                   color=TURBO_COLOR, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, pytorch_throughput, width, label='PyTorch',
                   color=PYTORCH_COLOR, edgecolor='black', linewidth=1.5)

    # Customize
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (samples/s)', fontsize=12, fontweight='bold')
    ax.set_title('TurboLoader vs PyTorch DataLoader: Throughput Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(' (', '\n(') for d in datasets], rotation=0, ha='center')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    output_path = output_dir / 'throughput_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'throughput_comparison.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_speedup_chart(results, output_dir):
    """Create speedup bar chart"""
    print("Creating speedup chart...")

    datasets = list(results.keys())
    speedups = [results[d]['speedup'] for d in datasets]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(datasets, speedups, color=TURBO_COLOR,
                  edgecolor='black', linewidth=1.5)

    # Add reference line at 1x
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (1x)')

    # Customize
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('TurboLoader Speedup Over PyTorch DataLoader',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels([d.replace(' (', '\n(') for d in datasets], rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}×',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # Add average line
    avg_speedup = np.mean(speedups)
    ax.axhline(y=avg_speedup, color=ACCENT_COLOR, linestyle='-', linewidth=2,
               label=f'Average: {avg_speedup:.1f}×')

    ax.legend(fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    output_path = output_dir / 'speedup_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'speedup_chart.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_category_breakdown(results, output_dir):
    """Create breakdown by data type category"""
    print("Creating category breakdown...")

    # Group by category
    categories = {}
    for dataset, data in results.items():
        if 'Images' in dataset:
            category = 'Images'
        elif 'Text' in dataset:
            category = 'Text'
        elif 'Audio' in dataset:
            category = 'Audio'
        else:
            category = 'Other'

        if category not in categories:
            categories[category] = []

        categories[category].append({
            'name': dataset,
            'speedup': data['speedup'],
            'turbo': data['turboloader']['throughput'],
            'pytorch': data['pytorch']['throughput']
        })

    # Create subplots
    n_categories = len(categories)
    fig, axes = plt.subplots(1, n_categories, figsize=(5*n_categories, 6))

    if n_categories == 1:
        axes = [axes]

    for idx, (category, datasets) in enumerate(categories.items()):
        ax = axes[idx]

        names = [d['name'].split('(')[0].strip() for d in datasets]
        speedups = [d['speedup'] for d in datasets]

        bars = ax.bar(names, speedups, color=TURBO_COLOR,
                     edgecolor='black', linewidth=1.5)

        ax.set_title(f'{category}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Speedup (×)', fontsize=11, fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}×',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

    plt.suptitle('TurboLoader Speedup by Data Type',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'category_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_breakdown.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_batch_time_comparison(results, output_dir):
    """Create batch time comparison"""
    print("Creating batch time comparison...")

    datasets = list(results.keys())
    turbo_batch_time = [results[d]['turboloader']['avg_batch_time_ms'] for d in datasets]
    pytorch_batch_time = [results[d]['pytorch']['avg_batch_time_ms'] for d in datasets]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, turbo_batch_time, width, label='TurboLoader',
                   color=TURBO_COLOR, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, pytorch_batch_time, width, label='PyTorch',
                   color=PYTORCH_COLOR, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Batch (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Average Batch Processing Time',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(' (', '\n(') for d in datasets], rotation=0, ha='center')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    output_path = output_dir / 'batch_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'batch_time_comparison.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_summary_infographic(results, output_dir):
    """Create summary infographic"""
    print("Creating summary infographic...")

    speedups = [results[d]['speedup'] for d in results.keys()]
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)
    min_speedup = np.min(speedups)

    # Count categories
    n_image = sum(1 for d in results.keys() if 'Images' in d)
    n_text = sum(1 for d in results.keys() if 'Text' in d)
    n_audio = sum(1 for d in results.keys() if 'Audio' in d)

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('TurboLoader Performance Summary', fontsize=20, fontweight='bold', y=0.98)

    # Box 1: Average Speedup
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, f'{avg_speedup:.1f}×', ha='center', va='center',
             fontsize=60, fontweight='bold', color=TURBO_COLOR)
    ax1.text(0.5, 0.15, 'Average Speedup', ha='center', va='center',
             fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(mpatches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                     edgecolor='black', linewidth=3))

    # Box 2: Max Speedup
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, f'{max_speedup:.1f}×', ha='center', va='center',
             fontsize=60, fontweight='bold', color=ACCENT_COLOR)
    ax2.text(0.5, 0.15, 'Peak Speedup', ha='center', va='center',
             fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(mpatches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                     edgecolor='black', linewidth=3))

    # Box 3: Datasets Tested
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, f'{len(results)}', ha='center', va='center',
             fontsize=60, fontweight='bold', color=PYTORCH_COLOR)
    ax3.text(0.5, 0.15, 'Datasets Tested', ha='center', va='center',
             fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.add_patch(mpatches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                     edgecolor='black', linewidth=3))

    # Bottom: Speedup distribution
    ax4 = fig.add_subplot(gs[1, :])
    datasets = list(results.keys())
    speedups_list = [results[d]['speedup'] for d in datasets]

    bars = ax4.barh(datasets, speedups_list, color=TURBO_COLOR,
                    edgecolor='black', linewidth=1.5)

    # Color bars by category
    for idx, (bar, dataset) in enumerate(zip(bars, datasets)):
        if 'Images' in dataset:
            bar.set_color('#FF6B35')
        elif 'Text' in dataset:
            bar.set_color('#4ECDC4')
        elif 'Audio' in dataset:
            bar.set_color('#FFB74D')

    ax4.axvline(x=avg_speedup, color='red', linestyle='--', linewidth=2,
                label=f'Average: {avg_speedup:.1f}×')
    ax4.set_xlabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax4.set_title('Speedup by Dataset', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    # Add value labels
    for idx, (bar, speedup) in enumerate(zip(bars, speedups_list)):
        ax4.text(speedup + 0.5, idx, f'{speedup:.1f}×',
                va='center', fontsize=10, fontweight='bold')

    # Legend
    image_patch = mpatches.Patch(color='#FF6B35', label='Images')
    text_patch = mpatches.Patch(color='#4ECDC4', label='Text')
    audio_patch = mpatches.Patch(color='#FFB74D', label='Audio')
    ax4.legend(handles=[image_patch, text_patch, audio_patch],
               loc='lower right', fontsize=10, frameon=True, shadow=True)

    output_path = output_dir / 'summary_infographic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_infographic.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_readme_hero_image(results, output_dir):
    """Create hero image for README"""
    print("Creating README hero image...")

    # Calculate stats
    speedups = [results[d]['speedup'] for d in results.keys()]
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Left side: Big speedup number
    ax.text(0.25, 0.5, f'{avg_speedup:.0f}×', ha='center', va='center',
            fontsize=120, fontweight='bold', color=TURBO_COLOR,
            transform=ax.transAxes)

    ax.text(0.25, 0.25, 'FASTER', ha='center', va='center',
            fontsize=36, fontweight='bold', color='black',
            transform=ax.transAxes)

    # Right side: Mini bar chart
    datasets = list(results.keys())[:4]  # Top 4
    speedups_list = [results[d]['speedup'] for d in datasets]
    short_names = [d.split('(')[0].strip()[:15] for d in datasets]

    # Inset axes for bar chart
    ax_inset = fig.add_axes([0.55, 0.2, 0.4, 0.6])
    bars = ax_inset.barh(short_names, speedups_list, color=TURBO_COLOR,
                         edgecolor='black', linewidth=2)

    ax_inset.set_xlabel('Speedup (×)', fontsize=14, fontweight='bold')
    ax_inset.set_title('TurboLoader vs PyTorch DataLoader',
                       fontsize=16, fontweight='bold')
    ax_inset.grid(axis='x', alpha=0.3, linestyle='--')

    # Add labels
    for idx, (bar, speedup) in enumerate(zip(bars, speedups_list)):
        ax_inset.text(speedup + 0.5, idx, f'{speedup:.1f}×',
                     va='center', fontsize=12, fontweight='bold')

    ax.axis('off')

    output_path = output_dir / 'readme_hero.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from benchmark results')
    parser.add_argument('--results', default='benchmark_results/comprehensive_benchmark.json',
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', default='benchmark_results/plots',
                       help='Output directory for plots')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)

    print(f"\nGenerating {len(results)} dataset plots...")

    # Generate all plots
    create_throughput_comparison(results, output_dir)
    create_speedup_chart(results, output_dir)
    create_category_breakdown(results, output_dir)
    create_batch_time_comparison(results, output_dir)
    create_summary_infographic(results, output_dir)
    create_readme_hero_image(results, output_dir)

    print(f"\n✅ All plots saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - throughput_comparison.png/pdf")
    print("  - speedup_chart.png/pdf")
    print("  - category_breakdown.png/pdf")
    print("  - batch_time_comparison.png/pdf")
    print("  - summary_infographic.png/pdf")
    print("  - readme_hero.png")

if __name__ == '__main__':
    main()
