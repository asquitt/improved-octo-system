#!/usr/bin/env python3
"""
Visualization generator for benchmark results

Generates ASCII charts and optionally matplotlib charts if available.
Works with minimal dependencies for maximum compatibility.
"""

import json
import sys
from typing import Dict, List, Tuple


def load_results(filepath: str = "benchmark_results.json") -> dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        print("Run benchmark_suite.py first to generate results")
        sys.exit(1)


def create_ascii_bar_chart(data: List[Tuple[str, float]],
                           title: str,
                           max_width: int = 60,
                           unit: str = "") -> str:
    """Create ASCII bar chart

    Args:
        data: List of (label, value) tuples
        title: Chart title
        max_width: Maximum width of bars
        unit: Unit to display

    Returns:
        ASCII bar chart as string
    """
    if not data:
        return "No data to display"

    max_value = max(v for _, v in data)
    max_label_len = max(len(label) for label, _ in data)

    lines = []
    lines.append("\n" + "="*80)
    lines.append(title)
    lines.append("="*80 + "\n")

    for label, value in data:
        # Calculate bar length
        if max_value > 0:
            bar_len = int((value / max_value) * max_width)
        else:
            bar_len = 0

        # Create bar
        bar = "█" * bar_len

        # Format value
        if unit == "x":
            value_str = f"{value:.2f}{unit}"
        elif unit == "MB":
            value_str = f"{value:,.0f} {unit}"
        elif unit == "ms":
            value_str = f"{value:.2f} {unit}"
        elif unit == "s/s":
            value_str = f"{value:,.1f} {unit}"
        else:
            value_str = f"{value:,.2f}"

        # Format line
        line = f"{label:<{max_label_len}} │ {bar} {value_str}"
        lines.append(line)

    return "\n".join(lines)


def create_speedup_chart(results: List[dict]) -> str:
    """Create speedup comparison chart"""
    data = [(r['config']['name'], r['speedup_vs_baseline'])
            for r in results]
    return create_ascii_bar_chart(data, "SPEEDUP vs BASELINE", unit="x")


def create_throughput_chart(results: List[dict]) -> str:
    """Create throughput comparison chart"""
    data = [(r['config']['name'], r['throughput_samples_per_sec'])
            for r in results]
    return create_ascii_bar_chart(data, "THROUGHPUT (samples/sec)", unit="s/s")


def create_memory_chart(results: List[dict]) -> str:
    """Create memory usage comparison chart"""
    data = [(r['config']['name'], r['memory_mb'])
            for r in results]
    return create_ascii_bar_chart(data, "MEMORY USAGE", unit="MB")


def create_step_time_chart(results: List[dict]) -> str:
    """Create step time comparison chart"""
    data = [(r['config']['name'], r['step_time_ms'])
            for r in results]
    return create_ascii_bar_chart(data, "STEP TIME (lower is better)", unit="ms")


def create_efficiency_chart(results: List[dict]) -> str:
    """Create multi-GPU efficiency chart"""
    # Filter for multi-GPU results
    multi_gpu = [r for r in results if r['config']['num_gpus'] > 1]

    if not multi_gpu:
        return "\nNo multi-GPU results available"

    data = [(f"{r['config']['num_gpus']} GPUs", r['efficiency_percent'])
            for r in multi_gpu]
    return create_ascii_bar_chart(data, "MULTI-GPU SCALING EFFICIENCY", unit="%")


def create_optimization_breakdown(results: List[dict]) -> str:
    """Create breakdown of optimization contributions"""
    if len(results) < 5:
        return "\nInsufficient results for optimization breakdown"

    lines = []
    lines.append("\n" + "="*80)
    lines.append("OPTIMIZATION CONTRIBUTION ANALYSIS")
    lines.append("="*80 + "\n")

    baseline = results[0]

    # Find individual optimization results
    compile_result = next((r for r in results if r['config']['use_compile'] and
                          not r['config']['use_mixed_precision'] and
                          not r['config']['use_fsdp2']), None)

    mp_result = next((r for r in results if r['config']['use_mixed_precision'] and
                     not r['config']['use_compile'] and
                     not r['config']['use_fsdp2']), None)

    fsdp2_result = next((r for r in results if r['config']['use_fsdp2'] and
                        not r['config']['use_compile'] and
                        not r['config']['use_mixed_precision']), None)

    full_result = next((r for r in results if r['config']['use_compile'] and
                       r['config']['use_mixed_precision'] and
                       r['config']['use_fsdp2']), None)

    if compile_result:
        speedup = compile_result['speedup_vs_baseline']
        lines.append(f"torch.compile:      {speedup:.2f}x speedup (+{(speedup-1)*100:.0f}%)")

    if mp_result:
        speedup = mp_result['speedup_vs_baseline']
        lines.append(f"Mixed Precision:    {speedup:.2f}x speedup (+{(speedup-1)*100:.0f}%)")

    if fsdp2_result:
        speedup = fsdp2_result['speedup_vs_baseline']
        memory_reduction = ((baseline['memory_mb'] - fsdp2_result['memory_mb']) /
                           baseline['memory_mb'] * 100)
        lines.append(f"FSDP2:              {speedup:.2f}x speedup (+{(speedup-1)*100:.0f}%), "
                    f"{memory_reduction:.0f}% memory reduction")

    if full_result:
        speedup = full_result['speedup_vs_baseline']
        memory_reduction = ((baseline['memory_mb'] - full_result['memory_mb']) /
                           baseline['memory_mb'] * 100)
        lines.append(f"\nFull Stack:         {speedup:.2f}x speedup (+{(speedup-1)*100:.0f}%), "
                    f"{memory_reduction:.0f}% memory reduction")

    return "\n".join(lines)


def create_scaling_analysis(results: List[dict]) -> str:
    """Create multi-GPU scaling analysis"""
    multi_gpu = sorted([r for r in results if r['config']['num_gpus'] > 1],
                      key=lambda x: x['config']['num_gpus'])

    if not multi_gpu:
        return "\nNo multi-GPU results available"

    lines = []
    lines.append("\n" + "="*80)
    lines.append("MULTI-GPU SCALING ANALYSIS")
    lines.append("="*80 + "\n")

    single_gpu = results[0]

    lines.append(f"{'GPUs':<8} {'Throughput':<20} {'Speedup':<15} {'Efficiency':<15}")
    lines.append("-" * 60)

    lines.append(f"{'1':<8} {single_gpu['throughput_samples_per_sec']:>15,.1f} s/s  "
                f"{'1.00x':<15} {'100.0%':<15}")

    for r in multi_gpu:
        gpus = r['config']['num_gpus']
        throughput = r['throughput_samples_per_sec']
        speedup = throughput / single_gpu['throughput_samples_per_sec']
        efficiency = r['efficiency_percent']

        lines.append(f"{gpus:<8} {throughput:>15,.1f} s/s  "
                    f"{speedup:<15.2f}x {efficiency:<15.1f}%")

    return "\n".join(lines)


def create_summary_table(results: List[dict]) -> str:
    """Create comprehensive summary table"""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("COMPREHENSIVE BENCHMARK RESULTS")
    lines.append("="*100 + "\n")

    lines.append(f"{'Configuration':<35} {'Throughput':<18} {'Memory':<15} {'Step Time':<12} {'Speedup':<10}")
    lines.append("-" * 100)

    for r in results:
        name = r['config']['name']
        if len(name) > 33:
            name = name[:30] + "..."

        throughput = f"{r['throughput_samples_per_sec']:,.1f} s/s"
        memory = f"{r['memory_mb']:,.0f} MB"
        step_time = f"{r['step_time_ms']:.2f} ms"
        speedup = f"{r['speedup_vs_baseline']:.2f}x"

        lines.append(f"{name:<35} {throughput:<18} {memory:<15} {step_time:<12} {speedup:<10}")

    return "\n".join(lines)


def create_text_visualizations(results_file: str = "benchmark_results.json"):
    """Create all text-based visualizations"""
    data = load_results(results_file)
    results = data['results']
    metadata = data['metadata']

    print("\n" + "="*100)
    print("DISTRIBUTED TRAINING FRAMEWORK - PERFORMANCE VISUALIZATION")
    print("="*100)
    print(f"\nMode: {metadata['mode'].upper()}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Number of benchmarks: {metadata['num_benchmarks']}")

    # Summary table
    print(create_summary_table(results))

    # Individual charts
    print(create_speedup_chart(results))
    print(create_throughput_chart(results))
    print(create_memory_chart(results))
    print(create_step_time_chart(results))

    # Analysis
    print(create_optimization_breakdown(results))
    print(create_scaling_analysis(results))
    print(create_efficiency_chart(results))

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")

    baseline = results[0]
    best_single = max((r for r in results if r['config']['num_gpus'] == 1),
                     key=lambda x: x['throughput_samples_per_sec'])

    print(f"✓ Best single-GPU configuration: {best_single['config']['name']}")
    print(f"  - Throughput: {best_single['throughput_samples_per_sec']:,.1f} samples/sec")
    print(f"  - Speedup: {best_single['speedup_vs_baseline']:.2f}x over baseline")
    print(f"  - Memory: {best_single['memory_mb']:,.0f} MB "
          f"({((baseline['memory_mb'] - best_single['memory_mb']) / baseline['memory_mb'] * 100):.0f}% reduction)")

    multi_gpu = [r for r in results if r['config']['num_gpus'] > 1]
    if multi_gpu:
        best_multi = max(multi_gpu, key=lambda x: x['throughput_samples_per_sec'])
        print(f"\n✓ Best multi-GPU configuration: {best_multi['config']['name']}")
        print(f"  - Throughput: {best_multi['throughput_samples_per_sec']:,.1f} samples/sec")
        print(f"  - Speedup: {best_multi['speedup_vs_baseline']:.2f}x over baseline")
        print(f"  - Efficiency: {best_multi['efficiency_percent']:.1f}%")

    print("\n" + "="*80)


def try_matplotlib_visualization(results_file: str = "benchmark_results.json"):
    """Try to create matplotlib visualizations if available"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nNote: matplotlib not available, skipping graphical visualizations")
        print("Install with: pip install matplotlib")
        return

    data = load_results(results_file)
    results = data['results']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distributed Training Performance Benchmarks', fontsize=16, fontweight='bold')

    # Speedup chart
    ax = axes[0, 0]
    names = [r['config']['name'] for r in results]
    speedups = [r['speedup_vs_baseline'] for r in results]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, speedups, color='#2ecc71')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Speedup vs Baseline', fontweight='bold')
    ax.set_title('Performance Speedup', fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Baseline')
    ax.legend()

    # Add value labels
    for i, v in enumerate(speedups):
        ax.text(v + 0.1, i, f'{v:.2f}x', va='center', fontweight='bold')

    # Throughput chart
    ax = axes[0, 1]
    throughputs = [r['throughput_samples_per_sec'] for r in results]
    bars = ax.barh(y_pos, throughputs, color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Samples per Second', fontweight='bold')
    ax.set_title('Training Throughput', fontweight='bold')

    # Memory usage chart
    ax = axes[1, 0]
    memory = [r['memory_mb'] for r in results]
    bars = ax.barh(y_pos, memory, color='#e74c3c')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Memory (MB)', fontweight='bold')
    ax.set_title('Memory Usage', fontweight='bold')

    # Multi-GPU scaling
    ax = axes[1, 1]
    multi_gpu = sorted([r for r in results if r['config']['num_gpus'] > 1],
                      key=lambda x: x['config']['num_gpus'])

    if multi_gpu:
        gpus = [r['config']['num_gpus'] for r in multi_gpu]
        efficiencies = [r['efficiency_percent'] for r in multi_gpu]

        ax.plot(gpus, efficiencies, marker='o', linewidth=2, markersize=8, color='#9b59b6')
        ax.set_xlabel('Number of GPUs', fontweight='bold')
        ax.set_ylabel('Scaling Efficiency (%)', fontweight='bold')
        ax.set_title('Multi-GPU Scaling Efficiency', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Ideal (100%)')
        ax.axhline(y=90, color='orange', linestyle='--', linewidth=1, label='Good (90%)')
        ax.legend()

        # Add value labels
        for i, (g, e) in enumerate(zip(gpus, efficiencies)):
            ax.text(g, e + 2, f'{e:.1f}%', ha='center', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No multi-GPU results available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Save figure
    output_file = 'benchmark_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphical visualization saved to {output_file}")

    plt.close()


def main():
    """Main visualization function"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", default="benchmark_results.json",
                       help="Input results file")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip matplotlib plots")

    args = parser.parse_args()

    # Create text visualizations
    create_text_visualizations(args.input)

    # Try matplotlib if not disabled
    if not args.no_plots:
        try_matplotlib_visualization(args.input)

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
