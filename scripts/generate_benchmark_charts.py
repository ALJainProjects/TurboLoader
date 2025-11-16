#!/usr/bin/env python3
"""
Generate Benchmark Visualization Charts

Creates visual charts from benchmark JSON results:
- Throughput comparison bar chart
- Epoch time comparison
- Memory usage comparison
- Stability analysis (std deviation)
- Performance speedup ratios

Outputs both ASCII art charts (for terminal) and can optionally
generate matplotlib charts if available.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_benchmark_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all benchmark JSON results"""
    benchmarks = {}

    benchmark_files = {
        'PIL Baseline': '01_pil_baseline.json',
        'PyTorch Naive': '02_pytorch_naive.json',
        'PyTorch Optimized': '03_pytorch_optimized.json',
        'PyTorch Cached': '04_pytorch_cached.json',
        'TurboLoader': '05_turboloader.json',
        'FFCV': '06_ffcv.json',
        'NVIDIA DALI': '07_dali.json',
        'TensorFlow': '08_tensorflow.json',
        'ResNet-50 Training': '09_resnet50_training.json'
    }

    for name, filename in benchmark_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                benchmarks[name] = json.load(f)
        else:
            print(f"Warning: {filename} not found, skipping {name}", file=sys.stderr)

    return benchmarks


def create_ascii_bar_chart(data: List[Tuple[str, float]], title: str,
                          unit: str = '', max_width: int = 60) -> str:
    """Create ASCII bar chart"""
    if not data:
        return f"{title}\n(No data available)\n"

    chart = []
    chart.append("=" * 80)
    chart.append(title)
    chart.append("=" * 80)

    # Find max value for scaling
    max_val = max(val for _, val in data)
    if max_val == 0:
        max_val = 1

    for name, value in data:
        # Calculate bar width
        bar_width = int((value / max_val) * max_width)
        bar = '█' * bar_width

        # Format value
        if unit == 'img/s':
            val_str = f"{value:,.0f}"
        elif unit == 's':
            val_str = f"{value:.2f}"
        elif unit == 'MB':
            val_str = f"{value:.1f}"
        elif unit == '%':
            val_str = f"{value:.1f}"
        else:
            val_str = f"{value:.2f}"

        chart.append(f"{name:25s} {bar} {val_str} {unit}")

    chart.append("=" * 80)
    chart.append("")

    return '\n'.join(chart)


def generate_throughput_chart(benchmarks: Dict[str, Dict]) -> str:
    """Generate throughput comparison chart"""
    data = []

    for name, results in benchmarks.items():
        if name == 'ResNet-50 Training':
            continue  # Skip training pipeline for throughput

        if 'throughput' in results:
            data.append((name, results['throughput']))

    # Sort by throughput (descending)
    data.sort(key=lambda x: x[1], reverse=True)

    return create_ascii_bar_chart(
        data,
        "Throughput Comparison (images/second)",
        unit="img/s"
    )


def generate_epoch_time_chart(benchmarks: Dict[str, Dict]) -> str:
    """Generate average epoch time comparison chart"""
    data = []

    for name, results in benchmarks.items():
        if name == 'ResNet-50 Training':
            continue  # Skip training pipeline

        if 'avg_epoch_time' in results:
            data.append((name, results['avg_epoch_time']))

    # Sort by time (ascending - faster is better)
    data.sort(key=lambda x: x[1])

    return create_ascii_bar_chart(
        data,
        "Average Epoch Time (seconds)",
        unit="s"
    )


def generate_memory_chart(benchmarks: Dict[str, Dict]) -> str:
    """Generate memory usage comparison chart"""
    data = []

    for name, results in benchmarks.items():
        if name == 'ResNet-50 Training':
            continue  # Skip training pipeline

        if 'peak_memory_mb' in results:
            data.append((name, results['peak_memory_mb']))

    # Sort by memory (ascending - less is better)
    data.sort(key=lambda x: x[1])

    return create_ascii_bar_chart(
        data,
        "Peak Memory Usage (MB)",
        unit="MB"
    )


def generate_stability_chart(benchmarks: Dict[str, Dict]) -> str:
    """Generate stability comparison (std deviation)"""
    data = []

    for name, results in benchmarks.items():
        if name == 'ResNet-50 Training':
            continue  # Skip training pipeline

        if 'std_epoch_time' in results and 'avg_epoch_time' in results:
            # Calculate coefficient of variation (std/mean * 100)
            cv = (results['std_epoch_time'] / results['avg_epoch_time']) * 100
            data.append((name, cv))

    # Sort by CV (ascending - more stable is better)
    data.sort(key=lambda x: x[1])

    return create_ascii_bar_chart(
        data,
        "Stability (Coefficient of Variation %)",
        unit="%"
    )


def generate_speedup_chart(benchmarks: Dict[str, Dict]) -> str:
    """Generate speedup comparison vs TurboLoader"""
    if 'TurboLoader' not in benchmarks:
        return "Speedup Chart\n(TurboLoader results not found)\n"

    turbo_throughput = benchmarks['TurboLoader'].get('throughput', 0)
    if turbo_throughput == 0:
        return "Speedup Chart\n(Invalid TurboLoader throughput)\n"

    data = []

    for name, results in benchmarks.items():
        if name == 'ResNet-50 Training' or name == 'TurboLoader':
            continue

        if 'throughput' in results:
            speedup = results['throughput'] / turbo_throughput
            data.append((name, speedup))

    # Sort by speedup (descending)
    data.sort(key=lambda x: x[1], reverse=True)

    return create_ascii_bar_chart(
        data,
        "Speedup vs TurboLoader (higher is faster)",
        unit="x"
    )


def generate_training_summary(benchmarks: Dict[str, Dict]) -> str:
    """Generate ResNet-50 training summary"""
    if 'ResNet-50 Training' not in benchmarks:
        return "ResNet-50 Training Results\n(Not available)\n"

    results = benchmarks['ResNet-50 Training']

    summary = []
    summary.append("=" * 80)
    summary.append("ResNet-50 Training Pipeline Results")
    summary.append("=" * 80)
    summary.append(f"Model: {results.get('model', 'N/A')}")
    summary.append(f"Total parameters: {results.get('total_parameters', 0):,}")
    summary.append(f"Trainable parameters: {results.get('trainable_parameters', 0):,}")
    summary.append("")
    summary.append(f"Optimizer: {results.get('optimizer', 'N/A')}")
    summary.append(f"Scheduler: {results.get('scheduler', 'N/A')}")
    summary.append(f"Loss function: {results.get('loss_function', 'N/A')}")
    summary.append("")
    summary.append(f"Epochs: {results.get('num_epochs', 0)}")
    summary.append(f"Batch size: {results.get('batch_size', 0)}")
    summary.append(f"Workers: {results.get('num_workers', 0)}")
    summary.append(f"Device: {results.get('device', 'N/A')}")
    summary.append("")
    summary.append(f"Total time: {results.get('total_time', 0):.2f}s")
    summary.append(f"Avg epoch time: {results.get('avg_epoch_time', 0):.2f}s ± {results.get('std_epoch_time', 0):.2f}s")
    summary.append("")
    summary.append(f"Final loss: {results.get('final_loss', 0):.4f}")
    summary.append(f"Final accuracy: {results.get('final_accuracy', 0):.2f}%")
    summary.append(f"Best accuracy: {results.get('best_accuracy', 0):.2f}%")

    # Epoch-by-epoch results
    if 'epoch_times' in results and 'losses_per_epoch' in results:
        summary.append("")
        summary.append("Epoch-by-Epoch Results:")
        summary.append("-" * 80)
        summary.append(f"{'Epoch':<8} {'Time (s)':<12} {'Loss':<12} {'Accuracy (%)':<15}")
        summary.append("-" * 80)

        for i, (time, loss, acc) in enumerate(zip(
            results['epoch_times'],
            results['losses_per_epoch'],
            results['accuracies_per_epoch']
        ), 1):
            summary.append(f"{i:<8} {time:<12.2f} {loss:<12.4f} {acc:<15.2f}")

    summary.append("=" * 80)
    summary.append("")

    return '\n'.join(summary)


def generate_comparison_table(benchmarks: Dict[str, Dict]) -> str:
    """Generate comprehensive comparison table"""
    table = []
    table.append("=" * 100)
    table.append("Comprehensive Benchmark Comparison")
    table.append("=" * 100)
    table.append(f"{'Framework':<25} {'Throughput':<15} {'Epoch Time':<15} {'Memory (MB)':<15} {'Stability':<15}")
    table.append("-" * 100)

    # Get TurboLoader throughput for comparison
    turbo_throughput = benchmarks.get('TurboLoader', {}).get('throughput', 0)

    for name, results in sorted(benchmarks.items()):
        if name == 'ResNet-50 Training':
            continue

        throughput = results.get('throughput', 0)
        epoch_time = results.get('avg_epoch_time', 0)
        memory = results.get('peak_memory_mb', 0)
        std_time = results.get('std_epoch_time', 0)

        # Format throughput with speedup
        if turbo_throughput > 0:
            speedup = throughput / turbo_throughput
            throughput_str = f"{throughput:,.0f} ({speedup:.2f}x)"
        else:
            throughput_str = f"{throughput:,.0f}"

        # Format epoch time
        epoch_time_str = f"{epoch_time:.2f}s"

        # Format memory
        memory_str = f"{memory:.1f}"

        # Format stability
        stability_str = f"±{std_time:.3f}s"

        table.append(f"{name:<25} {throughput_str:<15} {epoch_time_str:<15} {memory_str:<15} {stability_str:<15}")

    table.append("=" * 100)
    table.append("")

    return '\n'.join(table)


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark visualization charts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str,
                       default='benchmark_results',
                       help='Directory containing benchmark JSON results')
    parser.add_argument('--output', type=str,
                       default='BENCHMARK_CHARTS.txt',
                       help='Output file for ASCII charts')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("Loading benchmark results...")
    benchmarks = load_benchmark_results(results_dir)

    if not benchmarks:
        print("Error: No benchmark results found")
        sys.exit(1)

    print(f"Loaded {len(benchmarks)} benchmark results")
    print()

    # Generate all charts
    charts = []

    charts.append(generate_comparison_table(benchmarks))
    charts.append(generate_throughput_chart(benchmarks))
    charts.append(generate_epoch_time_chart(benchmarks))
    charts.append(generate_memory_chart(benchmarks))
    charts.append(generate_stability_chart(benchmarks))
    charts.append(generate_speedup_chart(benchmarks))
    charts.append(generate_training_summary(benchmarks))

    # Combine all charts
    output_text = '\n'.join(charts)

    # Print to terminal
    print(output_text)

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write(output_text)

    print(f"\nCharts saved to: {output_path}")


if __name__ == '__main__':
    main()
