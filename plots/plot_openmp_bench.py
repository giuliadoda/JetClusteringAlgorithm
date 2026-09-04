import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = '/mnt/POD/MCP_GD/JetClusteringAlgorithm/'
FIGURES_DIR = ROOT_DIR + 'plots/figures/benchmarks/openmp/'
BENCH_DIR = ROOT_DIR + 'benchmarks/'

BENCH_FILE = BENCH_DIR + 'benchmark_results_openmp_schedules_threads.csv'


colors = [  "#58508d", # violet
            "#bc5090", # purple
            "#ff6361", # orange
            "#ffa600"  # yellow
        ]

markers = ['o', 's', 'v', '^']


def get_baseline(df_grouped, schedule, threads_col='threads', ref_thread=None, value_col='avg_exec_time'):
    """
    Return the avg_exec_time (in minutes) for a given schedule at the reference
    thread count, looked up explicitly by name (never by positional index).
    """
    row = df_grouped[
        (df_grouped['schedule'] == schedule) &
        (df_grouped[threads_col] == ref_thread)
    ]
    if row.empty:
        raise ValueError(f"No baseline found for schedule={schedule}, threads={ref_thread}")
    return row[value_col].values[0] / 60  # minutes


if __name__ == "__main__":

    # read data
    bench_df = pd.read_csv(BENCH_FILE)

    # discard first run (warmup)
    bench_df = bench_df[bench_df['run'] >= 2]

    schedules = bench_df['schedule'].unique()
    chunks = bench_df['chunk'].unique()
    threads = np.sort(bench_df['threads'].unique())
    ref_thread = threads[0]  # baseline thread count for speedup/efficiency

    # averaging execution time per schedule with default chunk
    default_chunk = bench_df[bench_df['chunk'] == 'default']
    default_chunk = default_chunk.groupby(['schedule', 'threads']).agg(
        avg_exec_time=('time_sec', 'mean'),
        std_exec_time=('time_sec', 'std')
    ).reset_index()

    # PLOT I: time vs #threads per schedule with default chunks, speedup and efficiency
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    ax = ax.flatten()
    ax_time = ax[0]
    ax_su = ax[1]
    ax_ef = ax[2]

    # Ia: time vs #threads per schedule
    ax_time.grid(alpha=0.4)

    for i in range(len(schedules)):

        sched = schedules[i]
        df = default_chunk[default_chunk['schedule'] == sched].sort_values('threads')

        y = df['avg_exec_time'] / 60      # min
        y_err = df['std_exec_time'] / 60  # min

        ax_time.plot(threads, y, color=colors[i], linestyle='--', label=sched + ', chunks = default', marker=markers[i])
        ax_time.errorbar(threads, y, yerr=y_err, ecolor=colors[i], color=colors[i], capsize=2.5, linestyle='--')

    ax_time.set_ylabel('Average execution time (min)')
    ax_time.set_xlabel('# Threads')
    ax_time.set_xticks(threads)

    # Ib: speedup
    ax_su.grid(alpha=0.4)

    for i in range(len(schedules)):

        sched = schedules[i]
        df = default_chunk[default_chunk['schedule'] == sched].sort_values('threads')

        t1_i = get_baseline(default_chunk, sched, ref_thread=ref_thread)  # minutes, correct schedule

        y = df['avg_exec_time'] / 60      
        y_speedup = t1_i / y

        ax_su.plot((threads[0], threads[-1]), (y_speedup.iloc[0], y_speedup.iloc[0] * threads[-1]),
                   color='grey', linestyle='-.', alpha=0.6)

        ax_su.plot(threads, y_speedup, color=colors[i], linestyle='--', label=sched + ', chunks = default', marker=markers[i])

    ax_su.set_ylabel('Speedup')
    ax_su.set_xlabel('# Threads')
    ax_su.set_xticks(threads)

    # Ic: efficiency
    ax_ef.grid(alpha=0.4)

    ax_ef.axhline(1, color='grey', linestyle='-.', alpha=0.6, label='ideal')

    for i in range(len(schedules)):

        sched = schedules[i]
        df = default_chunk[default_chunk['schedule'] == sched].sort_values('threads')

        t1_i = get_baseline(default_chunk, sched, ref_thread=ref_thread)

        y = df['avg_exec_time'] / 60
        y_speedup = t1_i / y

        ax_ef.plot(threads, y_speedup / threads, color=colors[i], linestyle='--', label=sched + ', chunks = default', marker=markers[i])

    ax_ef.set_ylabel('Efficiency')
    ax_ef.set_xlabel('# Threads')
    ax_ef.set_xticks(threads)

    handles, labels = ax_ef.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=4)

    plt.savefig(FIGURES_DIR + 'time_threads_speedup_eff_default.png', dpi=300, bbox_inches='tight')

    # averaging execution time per schedule with chunks = 1000
    max_chunk = bench_df[bench_df['chunk'] == '1000']
    max_chunk = max_chunk.groupby(['schedule', 'threads']).agg(
        avg_exec_time=('time_sec', 'mean'),
        std_exec_time=('time_sec', 'std')
    ).reset_index()

    # PLOT II: time vs # threads per schedule with max #chunks (1000), speedup and efficiency
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    ax = ax.flatten()
    ax_time = ax[0]
    ax_su = ax[1]
    ax_ef = ax[2]

    # IIa
    ax_time.grid(alpha=0.4)

    for i in range(len(schedules)):

        sched = schedules[i]
        df = max_chunk[max_chunk['schedule'] == sched].sort_values('threads')

        y = df['avg_exec_time'] / 60      # min
        y_err = df['std_exec_time'] / 60  # min

        ax_time.plot(threads, y, color=colors[i], linestyle='--', label=sched + ', chunks = 1000', marker=markers[i])
        ax_time.errorbar(threads, y, yerr=y_err, ecolor=colors[i], color=colors[i], capsize=2.5, linestyle='--')

    ax_time.set_ylabel('Average execution time (min)')
    ax_time.set_xlabel('# Threads')
    ax_time.set_xticks(threads)

    # IIb: speedup
    ax_su.grid(alpha=0.4)

    for i in range(len(schedules)):

        sched = schedules[i]
        df = max_chunk[max_chunk['schedule'] == sched].sort_values('threads')

        t1_i = get_baseline(max_chunk, sched, ref_thread=ref_thread)

        y = df['avg_exec_time'] / 60
        y_speedup = t1_i / y

        ax_su.plot((threads[0], threads[-1]), (y_speedup.iloc[0], y_speedup.iloc[0] * threads[-1]),
                   color='grey', linestyle='-.', alpha=0.6)

        ax_su.plot(threads, y_speedup, color=colors[i], linestyle='--', label=sched + ', chunks = 1000', marker=markers[i])

    ax_su.set_ylabel('Speedup')
    ax_su.set_xlabel('# Threads')
    ax_su.set_xticks(threads)

    # IIc: efficiency
    ax_ef.grid(alpha=0.4)

    ax_ef.axhline(1, color='grey', linestyle='-.', alpha=0.6, label='ideal')

    for i in range(len(schedules)):

        sched = schedules[i]
        df = max_chunk[max_chunk['schedule'] == sched].sort_values('threads')

        t1_i = get_baseline(max_chunk, sched, ref_thread=ref_thread)

        y = df['avg_exec_time'] / 60
        y_speedup = t1_i / y

        ax_ef.plot(threads, y_speedup / threads, color=colors[i], linestyle='--', label=sched + ', chunks = 1000', marker=markers[i])

    ax_ef.set_ylabel('Efficiency')
    ax_ef.set_xlabel('# Threads')
    ax_ef.set_xticks(threads)

    handles, labels = ax_ef.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=4)

    plt.savefig(FIGURES_DIR + 'time_threads_speedup_eff_1000.png', dpi=300, bbox_inches='tight')

    # averaging execution time per schedule with best #threads vs #chunks
    bt = 8
    best_thread = bench_df[bench_df['threads'] == bt]
    best_thread = best_thread.groupby(['schedule', 'chunk']).agg(
        avg_exec_time=('time_sec', 'mean'),
        std_exec_time=('time_sec', 'std')
    ).reset_index()

    # PLOT III: time vs #chunks per schedule with best #threads
    fig, ax = plt.subplots()
    ax.grid(alpha=0.4)

    for i in range(len(schedules)):

        sched = schedules[i]
        df = best_thread[best_thread['schedule'] == sched].sort_values('chunk')

        x = df['chunk']

        y = df['avg_exec_time'] / 60
        y_err = df['std_exec_time'] / 60

        ax.scatter(x, y, color=colors[i], label=sched + f', #thread = {bt}', marker=markers[i])
        ax.errorbar(x, y, yerr=y_err, ecolor=colors[i], color=colors[i], capsize=2.5, lw=0, elinewidth=1)

    ax.set_ylabel('Average execution time (min)')
    ax.set_xlabel('# Chunks')
    ax.set_xticks(chunks)

    ax.set_ylim(7.5, 9)
    ax.set_yticks([7.5, 8, 8.5, 9])

    ax.legend()

    plt.savefig(FIGURES_DIR + f'time_vs_chunks_{bt}.png', dpi=300, bbox_inches='tight')