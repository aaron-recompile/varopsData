#!/usr/bin/env python3

import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCHNORR_BASELINE = 80000
AVERAGE_CUTOFF = SCHNORR_BASELINE / 2


def extract_machine_info(filepath: str) -> dict:
    info = {'cpu': 'Unknown', 'arch': 'Unknown', 'file': Path(filepath).name}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'CPU:' in line:
                info['cpu'] = line.split('CPU:')[1].strip()
            elif 'Architecture:' in line:
                info['arch'] = line.split('Architecture:')[1].strip()
    return info


def parse_csv(filepath: str) -> tuple[list[dict], dict]:
    results = []
    machine_info = extract_machine_info(filepath)
    
    with open(filepath, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    
    reader = csv.DictReader(lines)
    for row in reader:
        results.append({
            'rank': int(row['Rank']),
            'name': row['Name'],
            'seconds': float(row['Seconds']),
            'schnorr_equivalents': float(row['Schnorr_Equivalents']),
            'varops_percentage': float(row['Varops_Percentage']),
            'is_gsr_only': row['Is_GSR_Only'].lower() == 'true'
        })
    return results, machine_info


def parse_multiple_csvs(filepaths: list[str]) -> tuple[list[dict], list[dict]]:
    all_machine_data = []
    benchmark_data = defaultdict(lambda: {'seconds': [], 'schnorr_equivalents': [], 'varops_percentage': [], 'is_gsr_only': None})
    
    for filepath in filepaths:
        results, machine_info = parse_csv(filepath)
        all_machine_data.append((results, machine_info))
        
        for r in results:
            name = r['name']
            benchmark_data[name]['seconds'].append(r['seconds'])
            benchmark_data[name]['schnorr_equivalents'].append(r['schnorr_equivalents'])
            benchmark_data[name]['varops_percentage'].append(r['varops_percentage'])
            if benchmark_data[name]['is_gsr_only'] is None:
                benchmark_data[name]['is_gsr_only'] = r['is_gsr_only']
    
    averaged_results = []
    for name, data in benchmark_data.items():
        n = len(data['seconds'])
        avg_seconds = sum(data['seconds']) / n
        std_seconds = statistics.stdev(data['seconds']) if n > 1 else 0

        averaged_results.append({
            'name': name,
            'seconds': avg_seconds,
            'seconds_std': std_seconds,
            'schnorr_equivalents': sum(data['schnorr_equivalents']) / n,
            'varops_percentage': sum(data['varops_percentage']) / n,
            'is_gsr_only': data['is_gsr_only'],
            'seconds_all': data['seconds'],
            'schnorr_equivalents_all': data['schnorr_equivalents'],
        })
    
    return averaged_results, all_machine_data


def get_schnorr_baseline(results: list[dict]) -> dict | None:
    for r in results:
        if r['name'] == 'Schnorr signature validation':
            return r
    return None


def analyze_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    current_script = []
    gsr_added = []
    
    for r in results:
        if r['name'] == 'Schnorr signature validation':
            continue
        if r['is_gsr_only']:
            gsr_added.append(r)
        else:
            current_script.append(r)
    
    current_script.sort(key=lambda x: x['seconds'], reverse=True)
    gsr_added.sort(key=lambda x: x['seconds'], reverse=True)
    
    return current_script, gsr_added


def print_summary(current_script: list[dict], gsr_added: list[dict], schnorr: dict | None, num_machines: int):
    schnorr_time = schnorr['seconds'] if schnorr else 0
    
    print("\nBENCHMARK ANALYSIS")
    print(f"Averaged across {num_machines} machine(s)\n")
    
    if schnorr:
        print(f"Schnorr baseline: {schnorr_time:.3f}s")
    
    print("\nCURRENT BITCOIN SCRIPT")
    if current_script:
        print(f"{'Rank':<6} {'Operation':<45} {'Time (s)':<12} {'vs Schnorr':<15} {'Varops %'}")
        for i, r in enumerate(current_script[:10], 1):
            ratio = r['seconds'] / schnorr_time if schnorr_time > 0 else 0
            print(f"{i:<6} {r['name']:<45} {r['seconds']:<12.3f} {ratio:<15.2f}x {r['varops_percentage']:.1f}%")
        
        worst = current_script[0]
        print(f"\nWorst case: {worst['name']}")
        std = worst.get('seconds_std', 0)
        if std > 0:
            print(f"Time: {worst['seconds']:.3f} ± {std:.3f}s")
        else:
            print(f"Time: {worst['seconds']:.3f}s")
        if schnorr_time > 0:
            print(f"Ratio: {worst['seconds'] / schnorr_time:.2f}x Schnorr baseline")
    else:
        print("No current script operations found")
    
    print("\nNEW GSR OPERATIONS")
    if gsr_added:
        print(f"{'Rank':<6} {'Operation':<45} {'Time (s)':<12} {'vs Schnorr':<15} {'Varops %'}")
        for i, r in enumerate(gsr_added[:10], 1):
            ratio = r['seconds'] / schnorr_time if schnorr_time > 0 else 0
            print(f"{i:<6} {r['name']:<45} {r['seconds']:<12.3f} {ratio:<15.2f}x {r['varops_percentage']:.1f}%")
        
        worst = gsr_added[0]
        print(f"\nWorst case: {worst['name']}")
        std = worst.get('seconds_std', 0)
        if std > 0:
            print(f"Time: {worst['seconds']:.3f} ± {std:.3f}s")
        else:
            print(f"Time: {worst['seconds']:.3f}s")
        if schnorr_time > 0:
            print(f"Ratio: {worst['seconds'] / schnorr_time:.2f}x Schnorr baseline")
    else:
        print("No new GSR operations found")
    
    print("\nCOMPARISON")
    if current_script and gsr_added:
        curr_worst = current_script[0]['seconds']
        gsr_worst = gsr_added[0]['seconds']
        print(f"Worst current script: {curr_worst:.3f}s ({current_script[0]['name']})")
        print(f"Worst new GSR: {gsr_worst:.3f}s ({gsr_added[0]['name']})")
        print(f"Difference: {gsr_worst/curr_worst:.2f}x")
    print()


def create_averaged_visualization(current_script: list[dict], gsr_added: list[dict], schnorr: dict | None,
                                   num_machines: int, output_path: str):
    schnorr_time = schnorr['seconds'] if schnorr else 1

    curr_top = [r for r in current_script if r['schnorr_equivalents'] >= AVERAGE_CUTOFF]
    gsr_top = [r for r in gsr_added if r['schnorr_equivalents'] >= AVERAGE_CUTOFF]

    max_operations = max(len(curr_top), len(gsr_top))
    fig_height = max(10, min(30, max_operations * 0.6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
    subtitle = f'All individual data points across {num_machines} machine(s) - Operations with average >= {AVERAGE_CUTOFF:,.0f} Schnorr equivalents'
    fig.suptitle(f'Benchmark Results: Current Bitcoin Script vs New GSR Operations\n({subtitle})',
                 fontsize=14, fontweight='bold')

    curr_color = '#3498db'
    gsr_color = '#27ae60'
    schnorr_color = '#e74c3c'

    if curr_top:
        names = [r['name'][:40] + '...' if len(r['name']) > 40 else r['name'] for r in curr_top]

        for i, r in enumerate(curr_top):
            times_all = r.get('seconds_all', [r['seconds']])
            y_pos = [i] * len(times_all)

            ax1.scatter(times_all, y_pos, color=curr_color, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            ax1.scatter(r['seconds'], i, color=curr_color, s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        ax1.axvline(x=schnorr_time, color=schnorr_color, linestyle='--', linewidth=2, label=f'Schnorr baseline ({schnorr_time:.2f}s)')
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_title('Current Bitcoin Script', fontsize=12, fontweight='bold', color=curr_color)
        ax1.invert_yaxis()
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No current script operations', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Current Bitcoin Script', fontsize=12, fontweight='bold', color=curr_color)

    if gsr_top:
        names = [r['name'][:40] + '...' if len(r['name']) > 40 else r['name'] for r in gsr_top]

        for i, r in enumerate(gsr_top):
            times_all = r.get('seconds_all', [r['seconds']])
            y_pos = [i] * len(times_all)

            ax2.scatter(times_all, y_pos, color=gsr_color, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            ax2.scatter(r['seconds'], i, color=gsr_color, s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        ax2.axvline(x=schnorr_time, color=schnorr_color, linestyle='--', linewidth=2, label=f'Schnorr baseline ({schnorr_time:.2f}s)')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('Execution Time (seconds)')
        ax2.set_title('New Operations Added by GSR', fontsize=12, fontweight='bold', color=gsr_color)
        ax2.invert_yaxis()
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No new GSR operations', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('New Operations Added by GSR', fontsize=12, fontweight='bold', color=gsr_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_schnorr_equivalents_visualization(current_script: list[dict], gsr_added: list[dict], schnorr: dict | None,
                                             num_machines: int, output_path: str):
    curr_top = [r for r in current_script if r['schnorr_equivalents'] >= AVERAGE_CUTOFF]
    gsr_top = [r for r in gsr_added if r['schnorr_equivalents'] >= AVERAGE_CUTOFF]

    max_operations = max(len(curr_top), len(gsr_top))
    fig_height = max(10, min(30, max_operations * 0.6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
    subtitle = f'All individual data points across {num_machines} machine(s) - Operations with average >= {AVERAGE_CUTOFF:,.0f} Schnorr equivalents'
    fig.suptitle(f'Benchmark Results: Current Bitcoin Script vs New GSR Operations\n({subtitle})',
                 fontsize=14, fontweight='bold')

    curr_color = '#3498db'
    gsr_color = '#27ae60'
    schnorr_color = '#e74c3c'

    if curr_top:
        names = [r['name'][:40] + '...' if len(r['name']) > 40 else r['name'] for r in curr_top]

        for i, r in enumerate(curr_top):
            schnorr_eqs_all = r.get('schnorr_equivalents_all', [r['schnorr_equivalents']])
            y_pos = [i] * len(schnorr_eqs_all)

            ax1.scatter(schnorr_eqs_all, y_pos, color=curr_color, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            ax1.scatter(r['schnorr_equivalents'], i, color=curr_color, s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        ax1.axvline(x=SCHNORR_BASELINE, color=schnorr_color, linestyle='--', linewidth=2, label=f'Block limit ({SCHNORR_BASELINE:,} sigs)')
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Schnorr Signature Equivalents (per block)')
        ax1.set_title('Current Bitcoin Script', fontsize=12, fontweight='bold', color=curr_color)
        ax1.invert_yaxis()
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    else:
        ax1.text(0.5, 0.5, 'No current script operations', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Current Bitcoin Script', fontsize=12, fontweight='bold', color=curr_color)

    if gsr_top:
        names = [r['name'][:40] + '...' if len(r['name']) > 40 else r['name'] for r in gsr_top]

        for i, r in enumerate(gsr_top):
            schnorr_eqs_all = r.get('schnorr_equivalents_all', [r['schnorr_equivalents']])
            y_pos = [i] * len(schnorr_eqs_all)

            ax2.scatter(schnorr_eqs_all, y_pos, color=gsr_color, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            ax2.scatter(r['schnorr_equivalents'], i, color=gsr_color, s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        ax2.axvline(x=SCHNORR_BASELINE, color=schnorr_color, linestyle='--', linewidth=2, label=f'Block limit ({SCHNORR_BASELINE:,} sigs)')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('Schnorr Signature Equivalents (per block)')
        ax2.set_title('New Operations Added by GSR', fontsize=12, fontweight='bold', color=gsr_color)
        ax2.invert_yaxis()
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    else:
        ax2.text(0.5, 0.5, 'No new GSR operations', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('New Operations Added by GSR', fontsize=12, fontweight='bold', color=gsr_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_per_machine_visualization(all_machine_data: list[tuple], output_path: str):
    num_machines = len(all_machine_data)
    if num_machines == 0:
        return
    
    machine_names = []
    curr_worst_times = []
    curr_worst_names = []
    gsr_worst_times = []
    gsr_worst_names = []
    schnorr_times = []
    
    for results, machine_info in all_machine_data:
        cpu_short = machine_info['cpu'][:30] + '...' if len(machine_info['cpu']) > 30 else machine_info['cpu']
        machine_name = f"{cpu_short}\n({machine_info['arch']})"
        machine_names.append(machine_name)
        
        schnorr = get_schnorr_baseline(results)
        schnorr_times.append(schnorr['seconds'] if schnorr else 0)
        
        current_script, gsr_added = analyze_results(results)
        
        if current_script:
            curr_worst_times.append(current_script[0]['seconds'])
            curr_worst_names.append(current_script[0]['name'][:25] + '...' if len(current_script[0]['name']) > 25 else current_script[0]['name'])
        else:
            curr_worst_times.append(0)
            curr_worst_names.append('N/A')
        
        if gsr_added:
            gsr_worst_times.append(gsr_added[0]['seconds'])
            gsr_worst_names.append(gsr_added[0]['name'][:25] + '...' if len(gsr_added[0]['name']) > 25 else gsr_added[0]['name'])
        else:
            gsr_worst_times.append(0)
            gsr_worst_names.append('N/A')
    
    fig, ax = plt.subplots(figsize=(14, max(6, num_machines * 1.5)))
    fig.suptitle('Worst Case Performance Across All Machines\n(Current Script vs New GSR Operations)', 
                 fontsize=14, fontweight='bold')
    
    y_pos = range(num_machines)
    bar_height = 0.35
    
    curr_color = '#3498db'
    gsr_color = '#27ae60'
    schnorr_color = '#e74c3c'
    
    bars1 = ax.barh([y - bar_height/2 for y in y_pos], curr_worst_times, bar_height, 
                    label='Current Script Worst', color=curr_color, alpha=0.8)
    bars2 = ax.barh([y + bar_height/2 for y in y_pos], gsr_worst_times, bar_height,
                    label='New GSR Worst', color=gsr_color, alpha=0.8)
    
    for i, schnorr_time in enumerate(schnorr_times):
        if schnorr_time > 0:
            ax.plot(schnorr_time, i, 'o', color=schnorr_color, markersize=10, zorder=5)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if curr_worst_times[i] > 0:
            ax.text(bar1.get_width() + 0.02, bar1.get_y() + bar1.get_height()/2,
                   f'{curr_worst_times[i]:.2f}s', va='center', fontsize=8)
        if gsr_worst_times[i] > 0:
            ax.text(bar2.get_width() + 0.02, bar2.get_y() + bar2.get_height()/2,
                   f'{gsr_worst_times[i]:.2f}s', va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(machine_names, fontsize=9)
    ax.set_xlabel('Execution Time (seconds)')
    ax.set_ylabel('Machine')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=curr_color, alpha=0.8, label='Current Script Worst Case'),
        mpatches.Patch(facecolor=gsr_color, alpha=0.8, label='New GSR Worst Case'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=schnorr_color, markersize=10, label='Schnorr Baseline')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    print("\nPER-MACHINE WORST CASES")
    print(f"{'Machine':<40} {'Current Script':<15} {'New GSR':<15} {'Schnorr':<12}")
    for i, (results, machine_info) in enumerate(all_machine_data):
        cpu_short = machine_info['cpu'][:35] + '...' if len(machine_info['cpu']) > 35 else machine_info['cpu']
        print(f"{cpu_short:<40} {curr_worst_times[i]:<15.3f} {gsr_worst_times[i]:<15.3f} {schnorr_times[i]:<12.3f}")
    print()


def main():
    if len(sys.argv) < 2:
        csv_paths = list(Path('.').glob('*.csv'))
        if not csv_paths:
            print("No CSV files found in current directory.")
            print("Usage: python3 visualize_bench.py <csv_file1> [csv_file2] [csv_file3] ...")
            print("       Or provide a directory: python3 visualize_bench.py <directory>")
            print("       Or run with no arguments to automatically use all CSV files in current directory.")
            sys.exit(1)
        csv_paths = [str(path) for path in csv_paths]
    else:
        first_arg = Path(sys.argv[1])
        if len(sys.argv) == 2 and first_arg.is_dir():
            csv_paths = list(first_arg.glob('*.csv'))
            if not csv_paths:
                print(f"No CSV files found in directory: {first_arg}")
                print("Usage: python3 visualize_bench.py <csv_file1> [csv_file2] [csv_file3] ...")
                print("       Or provide a directory: python3 visualize_bench.py <directory>")
                print("       Or run with no arguments to automatically use all CSV files in current directory.")
                sys.exit(1)
            csv_paths = [str(path) for path in csv_paths]
        else:
            csv_paths = sys.argv[1:]
    
    for path in csv_paths:
        if not Path(path).exists():
            print(f"Error: CSV file not found: {path}")
            sys.exit(1)
    
    print(f"Reading benchmark results from {len(csv_paths)} file(s):")
    for path in csv_paths:
        print(f"  - {path}")
    
    averaged_results, all_machine_data = parse_multiple_csvs(csv_paths)
    
    schnorr = get_schnorr_baseline(averaged_results)
    current_script, gsr_added = analyze_results(averaged_results)
    
    print_summary(current_script, gsr_added, schnorr, len(csv_paths))
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    output_base = 'benchmark_analysis' if len(csv_paths) > 1 else Path(csv_paths[0]).stem + '_analysis'
    create_averaged_visualization(current_script, gsr_added, schnorr, len(csv_paths), f'plots/{output_base}_averaged.png')
    create_schnorr_equivalents_visualization(current_script, gsr_added, schnorr, len(csv_paths), f'plots/{output_base}_schnorr_equivalents.png')

    if len(csv_paths) > 1:
        create_per_machine_visualization(all_machine_data, f'plots/{output_base}_per_machine.png')
    else:
        create_per_machine_visualization(all_machine_data, f'plots/{output_base}_machine.png')


if __name__ == '__main__':
    main()
