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


if __name__ == "__main__":

    # read data
    bench_df = pd.read_csv(BENCH_FILE)

    # discard first run (warmup)
    bench_df = bench_df[bench_df['run']>=2]

    schedules = bench_df['schedule'].unique()
    chunks = bench_df['chunk'].unique()
    threads = np.sort(bench_df['threads'].unique())

    # averaging execution time per schedule with default chunk 
    default_chunk = bench_df[bench_df['chunk']=='default']
    default_chunk = default_chunk.groupby(['schedule', 'threads']).agg(
        avg_exec_time = ('time_sec', 'mean'),
        std_exec_time = ('time_sec', 'std')
    ).reset_index()

    # PLOT I: time vs #threads per schedule with default chunks, speedup and efficiency
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (16,6))
    ax = ax.flatten()
    ax_time = ax[0]
    ax_su = ax[1]
    ax_ef = ax[2]

    ax_time.grid(alpha = 0.4)

    for i in range(len(schedules)):

        df = default_chunk[default_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60      # min
        y_err = df['std_exec_time']/60  # min

        ax_time.plot(threads, y, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = default', marker = markers[i])
        ax_time.errorbar(threads, y, yerr = y_err, ecolor = colors[i], color = colors[i], capsize = 2.5, linestyle = '--')

    ax_time.set_ylabel('Average execution time (min)')
    ax_time.set_xlabel('# Threads')
    ax_time.set_xticks(threads)

    # speedup
    ax_su.grid(alpha = 0.4)

    t1 = np.array(default_chunk[default_chunk['threads']==threads[0]]['avg_exec_time'])

    for i in range(len(schedules)):

        df = default_chunk[default_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60
        y = np.array(t1[i]/y)

        ax_su.plot((threads[0], threads[-1]), (y[0], y[0]*threads[-1]), color = 'grey', linestyle = '-.', alpha = 0.6)

        ax_su.plot(threads, y, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = default', marker = markers[i])

    ax_su.set_ylabel('Speedup')
    ax_su.set_xlabel('# Threads')
    ax_su.set_xticks(threads)

    # efficiency
    ax_ef.grid(alpha = 0.4)

    ax_ef.axhline(1, color = 'grey', linestyle = '-.', alpha = 0.6, label='ideal')

    for i in range(len(schedules)):
    
        df = default_chunk[default_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60
        y = t1[i]/y

        ax_ef.plot(threads, y/threads, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = default', marker = markers[i])
    
    ax_ef.set_ylabel('Efficiency')
    ax_ef.set_xlabel('# Threads')
    ax_ef.set_xticks(threads)   
    ax_ef.set_yticks(np.array([i+1 if i==0 else i*10 for i in range(9)]))

    handles, labels = ax_ef.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=4)

    plt.savefig(FIGURES_DIR+'time_threads_speedup_eff_default.png', dpi = 300, bbox_inches='tight')

    # averaging execution time per schedule with chunks = 1000
    max_chunk = bench_df[bench_df['chunk']=='1000']
    max_chunk = max_chunk.groupby(['schedule', 'threads']).agg(
            avg_exec_time = ('time_sec', 'mean'),
            std_exec_time = ('time_sec', 'std')
        ).reset_index()

    # PLOT II: time vs # threads per schedule with max #chunks, speedup and efficiency
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16,6))
    ax = ax.flatten()
    ax_time = ax[0]
    ax_su = ax[1]
    ax_ef = ax[2]

    ax_time.grid(alpha = 0.4)
    
    for i in range(len(schedules)):

        df = max_chunk[max_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60      # min
        y_err = df['std_exec_time']/60  # min

        ax_time.plot(threads, y, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = 1000', marker = markers[i])
        ax_time.errorbar(threads, y, yerr = y_err, ecolor = colors[i], color = colors[i], capsize = 2.5, linestyle = '--')

    ax_time.set_ylabel('Average execution time (min)')
    ax_time.set_xlabel('# Threads')
    ax_time.set_xticks(threads)

    # speedup
    ax_su.grid(alpha = 0.4)

    t1 = np.array(max_chunk[max_chunk['threads']==threads[0]]['avg_exec_time'])

    for i in range(len(schedules)):

        df = max_chunk[max_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60
        y = np.array(t1[i]/y)

        ax_su.plot((threads[0], threads[-1]), (y[0], y[0]*threads[-1]), color = 'grey', linestyle = '-.', alpha = 0.6)

        ax_su.plot(threads, y, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = 1000', marker = markers[i])

    ax_su.set_ylabel('Speedup')
    ax_su.set_xlabel('# Threads')
    ax_su.set_xticks(threads)

    # efficiency
    ax_ef.grid(alpha = 0.4)
    
    ax_ef.axhline(1, color = 'grey', linestyle = '-.', alpha = 0.6, label='ideal')

    for i in range(len(schedules)):
    
        df = max_chunk[max_chunk['schedule']==schedules[i]].sort_index(level='threads')

        y = df['avg_exec_time']/60
        y = t1[i]/y

        ax_ef.plot(threads, y/threads, color = colors[i], linestyle = '--', label = schedules[i]+', chunks = 1000', marker = markers[i])
    
    ax_ef.set_ylabel('Efficiency')
    ax_ef.set_xlabel('# Threads')
    ax_ef.set_xticks(threads)   
    ax_ef.set_yticks(np.array([i+1 if i==0 else i*10 for i in range(9)]))

    handles, labels = ax_ef.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=4)

    plt.savefig(FIGURES_DIR+'time_threads_speedup_eff_1000.png', dpi = 300, bbox_inches='tight')

    # averaging execution time per schedule with best #threads vs #chunks
    bt = 8
    best_thread = bench_df[bench_df['threads']==bt]
    best_thread = best_thread.groupby(['schedule', 'chunk']).agg(
        avg_exec_time = ('time_sec', 'mean'),
        std_exec_time = ('time_sec', 'std')
    ).reset_index()

    # PLOT III: time vs #chunks per schedule with best #threads
    fig, ax = plt.subplots()
    ax.grid(alpha = 0.4)

    for i in range(len(schedules)):

        df = best_thread[best_thread['schedule']==schedules[i]].sort_index(level='threads')

        x = df['chunk']

        y = df['avg_exec_time']/60
        y_err = df['std_exec_time']/60

        ax.plot(x, y, color=colors[i], linestyle = '--', label = schedules[i]+f', #thread = {bt}', marker = markers[i])
        ax.errorbar(x, y, yerr=y_err, ecolor = colors[i], color = colors[i], capsize = 2.5, linestyle = '--')

    ax.set_ylabel('Average execution time (min)')
    ax.set_xlabel('# Chunks')
    ax.set_xticks(chunks)

    ax.set_ylim(5.5,10)
    ax.set_yticks([6,7,8,9,10])

    ax.legend()

    plt.savefig(FIGURES_DIR+'time_vs_chunks.png', dpi = 300, bbox_inches='tight')