#!/usr/bin/env python3
"""
Generate Comprehensive Benchmark Report with Plotly Charts

Creates an interactive HTML report from benchmark JSON results:
- Throughput comparison (bar chart)
- Epoch time comparison (bar chart)
- Memory usage comparison (bar chart)
- Stability analysis (scatter/bar chart)
- Speedup ratio chart
- Detailed comparison table

Outputs interactive HTML report using Plotly.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("Error: plotly is required for this script")
    print("Install with: pip install plotly")
    sys.exit(1)


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
    }

    for name, filename in benchmark_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                benchmarks[name] = json.load(f)
        else:
            print(f"Warning: {filename} not found, skipping {name}", file=sys.stderr)

    return benchmarks


def create_throughput_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create throughput comparison bar chart"""
    data = []

    for name, results in benchmarks.items():
        if 'throughput' in results:
            data.append((name, results['throughput']))

    # Sort by throughput (descending)
    data.sort(key=lambda x: x[1], reverse=True)

    names = [d[0] for d in data]
    values = [d[1] for d in data]

    # Create bar chart with color scale
    colors = px.colors.sequential.Viridis

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            text=[f'{v:,.0f}' for v in values],
            textposition='outside',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Images/sec")
            ),
            hovertemplate='<b>%{x}</b><br>Throughput: %{y:,.0f} img/s<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text="Throughput Comparison (Images/Second)",
            font=dict(size=20)
        ),
        xaxis_title="Framework",
        yaxis_title="Throughput (images/second)",
        yaxis=dict(type='log'),
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


def create_epoch_time_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create epoch time comparison bar chart"""
    data = []

    for name, results in benchmarks.items():
        if 'avg_epoch_time' in results:
            avg = results['avg_epoch_time']
            std = results.get('std_epoch_time', 0)
            data.append((name, avg, std))

    # Sort by time (ascending - faster is better)
    data.sort(key=lambda x: x[1])

    names = [d[0] for d in data]
    values = [d[1] for d in data]
    errors = [d[2] for d in data]

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            error_y=dict(type='data', array=errors),
            text=[f'{v:.2f}s' for v in values],
            textposition='outside',
            marker=dict(
                color=values,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Seconds")
            ),
            hovertemplate='<b>%{x}</b><br>Avg Time: %{y:.2f}s<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text="Average Epoch Time (Lower is Better)",
            font=dict(size=20)
        ),
        xaxis_title="Framework",
        yaxis_title="Time (seconds)",
        yaxis=dict(type='log'),
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


def create_memory_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create memory usage comparison bar chart"""
    data = []

    for name, results in benchmarks.items():
        if 'peak_memory_mb' in results:
            peak = results['peak_memory_mb']
            avg = results.get('avg_memory_mb', peak)
            data.append((name, peak, avg))

    # Sort by peak memory (ascending - less is better)
    data.sort(key=lambda x: x[1])

    names = [d[0] for d in data]
    peak_values = [d[1] for d in data]
    avg_values = [d[2] for d in data]

    fig = go.Figure(data=[
        go.Bar(
            name='Peak Memory',
            x=names,
            y=peak_values,
            text=[f'{v:.1f} MB' for v in peak_values],
            textposition='outside',
            marker=dict(color='indianred'),
            hovertemplate='<b>%{x}</b><br>Peak: %{y:.1f} MB<extra></extra>'
        ),
        go.Bar(
            name='Avg Memory',
            x=names,
            y=avg_values,
            text=[f'{v:.1f} MB' for v in avg_values],
            textposition='inside',
            marker=dict(color='lightsalmon'),
            hovertemplate='<b>%{x}</b><br>Avg: %{y:.1f} MB<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text="Memory Usage (Lower is Better)",
            font=dict(size=20)
        ),
        xaxis_title="Framework",
        yaxis_title="Memory (MB)",
        barmode='group',
        template='plotly_white',
        height=500
    )

    return fig


def create_stability_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create stability analysis scatter chart"""
    data = []

    for name, results in benchmarks.items():
        if 'std_epoch_time' in results and 'avg_epoch_time' in results:
            std = results['std_epoch_time']
            avg = results['avg_epoch_time']
            # Coefficient of variation (%)
            cv = (std / avg) * 100 if avg > 0 else 0
            data.append((name, cv, avg))

    # Sort by CV (ascending - more stable is better)
    data.sort(key=lambda x: x[1])

    names = [d[0] for d in data]
    cv_values = [d[1] for d in data]
    avg_values = [d[2] for d in data]

    # Create scatter plot with size based on epoch time
    fig = go.Figure(data=[
        go.Scatter(
            x=names,
            y=cv_values,
            mode='markers+text',
            marker=dict(
                size=[a * 20 for a in avg_values],  # Scale by epoch time
                color=cv_values,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="CV %"),
                line=dict(width=2, color='DarkSlateGray')
            ),
            text=[f'{v:.2f}%' for v in cv_values],
            textposition='top center',
            hovertemplate='<b>%{x}</b><br>CV: %{y:.2f}%<br>Avg Time: %{customdata:.2f}s<extra></extra>',
            customdata=avg_values
        )
    ])

    fig.update_layout(
        title=dict(
            text="Stability Analysis (Coefficient of Variation %)<br><sub>Lower is more stable, bubble size = epoch time</sub>",
            font=dict(size=20)
        ),
        xaxis_title="Framework",
        yaxis_title="Coefficient of Variation (%)",
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


def create_speedup_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create speedup comparison vs TurboLoader"""
    if 'TurboLoader' not in benchmarks:
        return None

    turbo_throughput = benchmarks['TurboLoader'].get('throughput', 0)
    if turbo_throughput == 0:
        return None

    data = []

    for name, results in benchmarks.items():
        if name == 'TurboLoader':
            continue
        if 'throughput' in results:
            speedup = results['throughput'] / turbo_throughput
            data.append((name, speedup))

    # Sort by speedup (descending)
    data.sort(key=lambda x: x[1], reverse=True)

    names = [d[0] for d in data]
    values = [d[1] for d in data]

    # Color based on speedup
    colors = ['green' if v >= 0.5 else 'orange' if v >= 0.2 else 'red' for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            text=[f'{v:.2f}x' for v in values],
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>Speedup: %{y:.2f}x vs TurboLoader<extra></extra>'
        )
    ])

    # Add reference line at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        annotation_text="TurboLoader baseline (1.0x)",
        annotation_position="right"
    )

    fig.update_layout(
        title=dict(
            text="Relative Performance vs TurboLoader",
            font=dict(size=20)
        ),
        xaxis_title="Framework",
        yaxis_title="Speedup Ratio (vs TurboLoader)",
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


def create_comparison_table(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create comprehensive comparison table"""
    # Get TurboLoader throughput for speedup calculation
    turbo_throughput = benchmarks.get('TurboLoader', {}).get('throughput', 0)

    headers = ['Framework', 'Throughput<br>(img/s)', 'Speedup<br>vs Turbo',
               'Epoch Time<br>(s)', 'Memory<br>(MB)', 'Stability<br>(CV %)']

    rows = []
    for name, results in sorted(benchmarks.items()):
        throughput = results.get('throughput', 0)
        speedup = throughput / turbo_throughput if turbo_throughput > 0 else 0
        epoch_time = results.get('avg_epoch_time', 0)
        memory = results.get('peak_memory_mb', 0)
        std_time = results.get('std_epoch_time', 0)
        cv = (std_time / epoch_time * 100) if epoch_time > 0 else 0

        rows.append([
            name,
            f"{throughput:,.0f}",
            f"{speedup:.2f}x",
            f"{epoch_time:.2f}",
            f"{memory:.1f}",
            f"{cv:.2f}%"
        ])

    # Transpose for table format
    cell_values = list(zip(*rows))

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=cell_values,
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    )])

    fig.update_layout(
        title=dict(
            text="Comprehensive Benchmark Comparison",
            font=dict(size=20)
        ),
        height=400
    )

    return fig


def create_epoch_timeseries_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create epoch-by-epoch time series line chart"""
    fig = go.Figure()

    for name, results in benchmarks.items():
        if 'epoch_times' in results and len(results['epoch_times']) > 0:
            epoch_times = results['epoch_times']
            epochs = list(range(1, len(epoch_times) + 1))

            fig.add_trace(go.Scatter(
                x=epochs,
                y=epoch_times,
                mode='lines+markers',
                name=name,
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate=f'<b>{name}</b><br>Epoch: %{{x}}<br>Time: %{{y:.3f}}s<extra></extra>'
            ))

    fig.update_layout(
        title=dict(
            text="Epoch-by-Epoch Performance Over Time",
            font=dict(size=20)
        ),
        xaxis_title="Epoch",
        yaxis_title="Time (seconds)",
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_throughput_timeseries_chart(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create throughput time series calculated from epoch times"""
    fig = go.Figure()

    for name, results in benchmarks.items():
        if 'epoch_times' in results and len(results['epoch_times']) > 0:
            epoch_times = results['epoch_times']
            epochs = list(range(1, len(epoch_times) + 1))

            # Calculate throughput for each epoch (images/time)
            # Assuming constant dataset size across epochs
            if 'throughput' in results and 'avg_epoch_time' in results:
                avg_epoch_time = results['avg_epoch_time']
                avg_throughput = results['throughput']
                # Calculate images per epoch
                images_per_epoch = avg_throughput * avg_epoch_time

                # Calculate throughput for each epoch
                throughputs = [images_per_epoch / t for t in epoch_times]

                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=throughputs,
                    mode='lines+markers',
                    name=name,
                    line=dict(width=2),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name}</b><br>Epoch: %{{x}}<br>Throughput: %{{y:,.0f}} img/s<extra></extra>'
                ))

    fig.update_layout(
        title=dict(
            text="Throughput Evolution Across Epochs",
            font=dict(size=20)
        ),
        xaxis_title="Epoch",
        yaxis_title="Throughput (images/second)",
        yaxis=dict(type='log'),
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_batch_time_distribution(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create batch time distribution comparison"""
    fig = go.Figure()

    for name, results in benchmarks.items():
        if 'avg_batch_time' in results and 'std_batch_time' in results:
            avg = results['avg_batch_time']
            std = results['std_batch_time']

            # Create violin plot showing distribution
            fig.add_trace(go.Box(
                y=[avg],
                name=name,
                boxmean='sd',
                marker=dict(size=8),
                hovertemplate=f'<b>{name}</b><br>Avg: {avg*1000:.2f}ms<br>Std: {std*1000:.2f}ms<extra></extra>'
            ))

    fig.update_layout(
        title=dict(
            text="Batch Processing Time Distribution",
            font=dict(size=20)
        ),
        yaxis_title="Time per Batch (seconds)",
        yaxis=dict(type='log'),
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def create_efficiency_heatmap(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create efficiency heatmap comparing multiple metrics"""
    frameworks = []
    throughput_scores = []
    memory_scores = []
    stability_scores = []

    # Get max values for normalization
    max_throughput = max((b.get('throughput', 0) for b in benchmarks.values()), default=1)
    max_memory = max((b.get('peak_memory_mb', 0) for b in benchmarks.values()), default=1)

    for name, results in sorted(benchmarks.items()):
        frameworks.append(name)

        # Normalize throughput (higher is better, so keep as-is)
        throughput = results.get('throughput', 0)
        throughput_scores.append(throughput / max_throughput * 100)

        # Normalize memory (lower is better, so invert)
        memory = results.get('peak_memory_mb', 0)
        memory_scores.append((1 - memory / max_memory) * 100 if max_memory > 0 else 0)

        # Stability score (lower CV is better, so invert)
        std_time = results.get('std_epoch_time', 0)
        avg_time = results.get('avg_epoch_time', 1)
        cv = (std_time / avg_time * 100) if avg_time > 0 else 0
        stability_scores.append(max(0, 100 - cv * 10))  # Scale CV for visibility

    # Create heatmap
    z_data = [throughput_scores, memory_scores, stability_scores]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=frameworks,
        y=['Throughput', 'Memory Efficiency', 'Stability'],
        colorscale='RdYlGn',
        text=[[f'{val:.1f}' for val in row] for row in z_data],
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{y}</b><br>%{x}<br>Score: %{z:.1f}/100<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Overall Efficiency Heatmap (Normalized Scores)",
            font=dict(size=20)
        ),
        template='plotly_white',
        height=400
    )

    return fig


def create_performance_radar(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create radar chart comparing top frameworks"""
    # Select top 4 frameworks by throughput
    top_frameworks = sorted(
        benchmarks.items(),
        key=lambda x: x[1].get('throughput', 0),
        reverse=True
    )[:4]

    fig = go.Figure()

    categories = ['Throughput', 'Memory\nEfficiency', 'Stability', 'Speed']

    for name, results in top_frameworks:
        # Normalize metrics to 0-100 scale
        max_throughput = max((b.get('throughput', 0) for b in benchmarks.values()), default=1)
        max_memory = max((b.get('peak_memory_mb', 0) for b in benchmarks.values()), default=1)

        throughput = results.get('throughput', 0)
        memory = results.get('peak_memory_mb', 0)
        std_time = results.get('std_epoch_time', 0)
        avg_time = results.get('avg_epoch_time', 1)

        # Calculate scores
        throughput_score = (throughput / max_throughput) * 100
        memory_score = (1 - memory / max_memory) * 100 if max_memory > 0 else 0
        cv = (std_time / avg_time * 100) if avg_time > 0 else 0
        stability_score = max(0, 100 - cv * 10)
        speed_score = (1 / avg_time) * 10 if avg_time > 0 else 0  # Inverse time
        speed_score = min(100, speed_score)  # Cap at 100

        values = [throughput_score, memory_score, stability_score, speed_score]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=dict(
            text="Performance Radar: Top 4 Frameworks",
            font=dict(size=20)
        ),
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def create_cost_benefit_scatter(benchmarks: Dict[str, Dict]) -> go.Figure:
    """Create scatter plot: throughput vs memory usage"""
    frameworks = []
    throughputs = []
    memories = []
    sizes = []

    for name, results in benchmarks.items():
        throughput = results.get('throughput', 0)
        memory = results.get('peak_memory_mb', 0)
        avg_time = results.get('avg_epoch_time', 1)

        frameworks.append(name)
        throughputs.append(throughput)
        memories.append(memory)
        sizes.append(max(10, avg_time * 100))  # Size based on epoch time

    fig = go.Figure(data=go.Scatter(
        x=memories,
        y=throughputs,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=throughputs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Throughput"),
            line=dict(width=2, color='DarkSlateGray')
        ),
        text=frameworks,
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Throughput: %{y:,.0f} img/s<br>Memory: %{x:.1f} MB<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Cost-Benefit Analysis: Throughput vs Memory<br><sub>Bubble size = epoch time</sub>",
            font=dict(size=20)
        ),
        xaxis_title="Peak Memory Usage (MB)",
        yaxis_title="Throughput (images/second)",
        yaxis=dict(type='log'),
        template='plotly_white',
        height=500
    )

    return fig


def generate_html_report(benchmarks: Dict[str, Dict], output_path: Path):
    """Generate comprehensive HTML report with all charts"""

    # Create all charts
    charts = []

    # Comparison table
    table_fig = create_comparison_table(benchmarks)
    if table_fig:
        charts.append(table_fig)

    # Throughput chart
    throughput_fig = create_throughput_chart(benchmarks)
    if throughput_fig:
        charts.append(throughput_fig)

    # Speedup chart
    speedup_fig = create_speedup_chart(benchmarks)
    if speedup_fig:
        charts.append(speedup_fig)

    # TIME SERIES CHARTS
    # Epoch-by-epoch time series
    epoch_ts_fig = create_epoch_timeseries_chart(benchmarks)
    if epoch_ts_fig:
        charts.append(epoch_ts_fig)

    # Throughput time series
    throughput_ts_fig = create_throughput_timeseries_chart(benchmarks)
    if throughput_ts_fig:
        charts.append(throughput_ts_fig)

    # NEW ADVANCED CHARTS
    # Efficiency heatmap
    heatmap_fig = create_efficiency_heatmap(benchmarks)
    if heatmap_fig:
        charts.append(heatmap_fig)

    # Performance radar
    radar_fig = create_performance_radar(benchmarks)
    if radar_fig:
        charts.append(radar_fig)

    # Cost-benefit scatter
    scatter_fig = create_cost_benefit_scatter(benchmarks)
    if scatter_fig:
        charts.append(scatter_fig)

    # Epoch time chart
    epoch_fig = create_epoch_time_chart(benchmarks)
    if epoch_fig:
        charts.append(epoch_fig)

    # Batch time distribution
    batch_fig = create_batch_time_distribution(benchmarks)
    if batch_fig:
        charts.append(batch_fig)

    # Memory chart
    memory_fig = create_memory_chart(benchmarks)
    if memory_fig:
        charts.append(memory_fig)

    # Stability chart
    stability_fig = create_stability_chart(benchmarks)
    if stability_fig:
        charts.append(stability_fig)

    # Generate HTML
    html_parts = [
        '<html>',
        '<head>',
        '<title>TurboLoader Benchmark Report</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
        'h1 { color: #333; text-align: center; }',
        '.chart-container { background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '</style>',
        '</head>',
        '<body>',
        '<h1>TurboLoader Benchmark Report</h1>',
        f'<p style="text-align: center; color: #666;">Generated from {len(benchmarks)} benchmark results</p>',
    ]

    # Add each chart
    for fig in charts:
        html_parts.append('<div class="chart-container">')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        html_parts.append('</div>')

    html_parts.extend(['</body>', '</html>'])

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive benchmark report with Plotly charts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str,
                       default='benchmark_results',
                       help='Directory containing benchmark JSON results')
    parser.add_argument('--output', type=str,
                       default='BENCHMARK_REPORT.html',
                       help='Output HTML file for report')

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

    print(f"Loaded {len(benchmarks)} benchmark results:")
    for name in benchmarks.keys():
        print(f"  - {name}")
    print()

    print("Generating interactive HTML report...")
    output_path = Path(args.output)
    generate_html_report(benchmarks, output_path)

    print(f"\n✓ Report generated successfully!")
    print(f"  File: {output_path.absolute()}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
