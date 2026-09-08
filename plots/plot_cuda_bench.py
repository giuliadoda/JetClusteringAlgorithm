import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = '/mnt/POD/MCP_GD/JetClusteringAlgorithm/'
FIGURES_DIR = ROOT_DIR + 'plots/figures/benchmarks/cuda/second_run/'
BENCH_DIR = ROOT_DIR + 'benchmarks/'

TIMING_FILE = BENCH_DIR + 'results_cuda_raw_2.csv'


colors = [  "#58508d", # violet
            "#bc5090", # purple
            "#ff6361", # orange
            "#ffa600", # yellow
            "#94d252", # green
            "#5ec1ff", # light blue
            "#df99ff"  # lilac
        ]

markers = ['o', 's', 'v', '^', '<', '>']


if __name__ == "__main__":

    # read data
    cudabench = pd.read_csv(TIMING_FILE)

    # discard warmup run
    cudabench = cudabench[cudabench['run'] > 1]

    thr_block = cudabench['thr_block'].unique()
    n_events = cudabench['n_events'].unique()

    # PLOT I: heatmaps #thr/block vs #events
    timings = ['wall_time_sec','h2d_ms','kernel_ms','d2h_ms','avg_event_ms']
    titles = ['Wall time', 'H2D transfer time', 'Kernel time', 'D2H transfer time', 'Average event time']

    for i in range(len(timings)):

        if i == 0:
            udm = 's'
        else:
            udm = 'ms'

        df = cudabench[['thr_block', 'n_events', 'run', timings[i]]]

        df = df.groupby(['thr_block', 'n_events']).agg(
            avg_exec_time = (timings[i], 'mean'),
            std_exec_time = (timings[i], 'std')
        ).reset_index()

        # plots 1D

        # thr_block vs n_events
        fig, ax = plt.subplots()
        ax.grid(alpha = 0.4)

        x = n_events

        for j in range(len(thr_block)):

            df_j = df[df['thr_block']==thr_block[j]]

            y = df_j['avg_exec_time']
            y_err = df_j['std_exec_time']

            ax.plot(x, y, label = str(thr_block[j]), color = colors[j], marker = markers[j])
            ax.errorbar(x, y, yerr=y_err, capsize = 2.5, color = colors[j], ecolor = colors[j])

        ax.set_xlabel('# Events')
        ax.set_ylabel(f'Average time ({udm})')
        ax.set_title(titles[i])

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(loc='upper center', ncols = 6)

        plt.savefig(FIGURES_DIR+timings[i]+'_thr_block.png', dpi = 300, bbox_inches='tight')

        # n_events vs thr_block
        fig, ax = plt.subplots()
        ax.grid(alpha = 0.4)

        x = thr_block

        for j in range(len(n_events)):

            df_j = df[df['n_events']==n_events[j]]

            y = df_j['avg_exec_time']
            y_err = df_j['std_exec_time']

            ax.plot(x, y, label = str(n_events[j]), color = colors[j], marker = markers[j])
            ax.errorbar(x, y, yerr=y_err, capsize = 2.5, color = colors[j], ecolor = colors[j])

        ax.set_xlabel('# Threads/block')
        ax.set_ylabel(f'Average time ({udm})')
        ax.set_title(titles[i])

        ax.set_yscale('log')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(loc='upper center', ncols = 5)

        plt.savefig(FIGURES_DIR+timings[i]+'_n_events.png', dpi = 300, bbox_inches='tight')

        # heatmap
        fig, ax = plt.subplots()

        heatmap_df = df.pivot(index='n_events', columns = 'thr_block', values = 'avg_exec_time')

        heatmap_df = heatmap_df.sort_index(axis=1)
        heatmap_df = heatmap_df.sort_index(axis=0)

        z = np.array(heatmap_df)

        im = ax.imshow(z, aspect='auto', cmap = 'seismic', norm='log', alpha = 0.85)

        ax.set_xticks(range(len(heatmap_df.columns)), labels=heatmap_df.columns)
        ax.set_xlabel('# Threads/block')
        ax.set_yticks(range(len(heatmap_df.index)), labels=heatmap_df.index)
        ax.set_ylabel('# Events')

        ax.invert_yaxis()

        ax.set_title(titles[i])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'Average time ({udm})')

        plt.savefig(FIGURES_DIR+timings[i]+'_heatmap.png', dpi = 300, bbox_inches='tight')

    # PLOT II: breakdown H2D / kernel / D2H vs thr_block
    fig, ax = plt.subplots(figsize=(8,6))
    ax.grid(alpha = 0.2)

    df_bd = cudabench[cudabench['n_events'] == 100000]

    df_bd = (
        df_bd.groupby('thr_block')[[ 'thr_block', 'h2d_ms', 'kernel_ms', 'd2h_ms']]
        .mean()
        .sort_index()
    )

    thr_block_x = df_bd['thr_block'].astype(int).astype(str).tolist()

    h2d = df_bd['h2d_ms']/1000
    kernel_stacked = df_bd['kernel_ms']/1000
    d2h = df_bd['d2h_ms']/1000

    ax.bar(thr_block_x, h2d, label = 'H2D', color = colors[-1])
    ax.bar(thr_block_x, kernel_stacked, bottom=h2d, label = 'kernel', color = colors[-2])
    ax.bar(thr_block_x, d2h, bottom=h2d + kernel_stacked, label = 'D2H', color = colors[-3])

    ax.set_xlabel("# Threads / block")
    ax.set_ylabel("Average time (s)")
    ax.set_title(f"Breakdown GPU times - # Events = 100000")
    ax.legend()

    plt.savefig(FIGURES_DIR+'breakdown.png', dpi = 300, bbox_inches='tight')

    # PLOT III: speedup and efficiency vs # threads/block
    
    # Baseline:
    #   smallest available number of threads/block,
    #   independently for each n_events.
    
    # Speedup:
    #   S = T_baseline / T(thr_block)
    
    # Efficiency:
    #   E = S / (thr_block / baseline_thr_block)

    # Average execution time for each (n_events, thr_block)
    scaling_df = (
        cudabench
        .groupby(['n_events', 'thr_block'])
        .agg(
            avg_exec_time=('wall_time_sec', 'mean'),
            std_exec_time=('wall_time_sec', 'std')
        )
        .reset_index()
        .sort_values(['n_events', 'thr_block'])
    )

    # Smallest number of threads/block = baseline
    ref_thr_block = scaling_df['thr_block'].min()

    # Sorted values for x-axis
    thr_values = np.sort(scaling_df['thr_block'].unique())

    # Sorted values for number of events
    events_values = np.sort(scaling_df['n_events'].unique())


    # ------------------------------------------------------------------
    # SPEEDUP
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(alpha=0.4)

    for i, n in enumerate(events_values):

        df = (
            scaling_df[scaling_df['n_events'] == n]
            .sort_values('thr_block')
        )

        # Baseline execution time for this n_events
        baseline = df[df['thr_block'] == ref_thr_block]

        if baseline.empty:
            raise ValueError(
                f"No baseline found for n_events={n}, "
                f"thr_block={ref_thr_block}"
            )

        t_baseline = baseline['avg_exec_time'].iloc[0]

        # Speedup
        speedup = t_baseline / df['avg_exec_time']

        ax.plot(
            df['thr_block'],
            speedup,
            color=colors[i % len(colors)],
            linestyle='--',
            marker=markers[i % len(markers)],
            label=str(n)
        )

    # Ideal linear speedup
    ideal_speedup = thr_values / ref_thr_block

    ax.plot(
        thr_values,
        ideal_speedup,
        color='grey',
        linestyle='-.',
        alpha=0.6,
        label='Ideal'
    )

    ax.set_xlabel("# Threads / block")
    ax.set_ylabel("Speedup")
    ax.set_title("CUDA speedup vs # Threads / block")

    ax.set_xticks(thr_values)

    ax.set_ylim(0,8)

    ax.legend(
        title="# Events"
    )

    plt.savefig(
        FIGURES_DIR + 'speedup_thr_block.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


    # ------------------------------------------------------------------
    # EFFICIENCY
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(alpha=0.4)

    for i, n in enumerate(events_values):

        df = (
            scaling_df[scaling_df['n_events'] == n]
            .sort_values('thr_block')
        )

        # Baseline execution time for this n_events
        baseline = df[df['thr_block'] == ref_thr_block]

        if baseline.empty:
            raise ValueError(
                f"No baseline found for n_events={n}, "
                f"thr_block={ref_thr_block}"
            )

        t_baseline = baseline['avg_exec_time'].iloc[0]

        # Speedup
        speedup = t_baseline / df['avg_exec_time']

        # Efficiency
        efficiency = speedup / (
            df['thr_block'] / ref_thr_block
        )

        ax.plot(
            df['thr_block'],
            efficiency,
            color=colors[i % len(colors)],
            linestyle='--',
            marker=markers[i % len(markers)],
            label=str(n)
        )

    # Ideal efficiency
    ax.axhline(
        1,
        color='grey',
        linestyle='-.',
        alpha=0.6,
        label='Ideal'
    )

    ax.set_xlabel("# Threads / block")
    ax.set_ylabel("Efficiency")
    ax.set_title("CUDA efficiency vs # Threads / block")

    ax.set_xticks(thr_values)

    ax.legend(
        title="# Events"
    )

    plt.savefig(
        FIGURES_DIR + 'efficiency_thr_block.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


 # ------------------------------------------------------------------
# PLOT IV: FLOPS vs # events and # threads/block
#
# kernel_ms = TOTAL kernel time for all events
#
# FLOPS = total_operations / total_kernel_time
#
# Error propagation:
#   sigma_FLOPS / FLOPS = sigma_T / T
# ------------------------------------------------------------------

# From Nsight Compute, order of magnitude
OPERATIONS_PER_EVENT = (
    362452999458        # dadd
    + 271916939493      # dmul
    + 2 * 725259954138  # dfma -> 2 FLOP
) / 100000

flops_df = (
    cudabench
    .groupby(['n_events', 'thr_block'])
    .agg(
        avg_kernel_ms=('kernel_ms', 'mean'),
        std_kernel_ms=('kernel_ms', 'std')
    )
    .reset_index()
)

# Convert total kernel time: ms -> s
flops_df['kernel_sec'] = (
    flops_df['avg_kernel_ms'] / 1000.0
)

flops_df['std_kernel_sec'] = (
    flops_df['std_kernel_ms'] / 1000.0
)

# Total number of operations
flops_df['total_operations'] = (
    OPERATIONS_PER_EVENT * flops_df['n_events']
)

# Achieved FLOPS
flops_df['flops'] = (
    flops_df['total_operations']
    / flops_df['kernel_sec']
)

# Error propagation:
# sigma_FLOPS = FLOPS * sigma_T / T
flops_df['std_flops'] = (
    flops_df['flops']
    * flops_df['std_kernel_sec']
    / flops_df['kernel_sec']
)

# Convert to GFLOPS
flops_df['gflops'] = (
    flops_df['flops'] / 1e9
)

flops_df['std_gflops'] = (
    flops_df['std_flops'] / 1e9
)


# ------------------------------------------------------------------
# FLOPS plot: one line for each n_events
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.grid(alpha=0.4)

events_values = np.sort(
    flops_df['n_events'].unique()
)

for i, n in enumerate(events_values):

    df = (
        flops_df[
            flops_df['n_events'] == n
        ]
        .sort_values('thr_block')
    )

    ax.errorbar(
        df['thr_block'],
        df['gflops'],
        yerr=df['std_gflops'],
        color=colors[i % len(colors)],
        linestyle='--',
        marker=markers[i % len(markers)],
        capsize=2.5,
        label=str(n)
    )

ax.set_xlabel("# Threads / block")
ax.set_ylabel("Achieved FP64 performance (GFLOPS)")
ax.set_title("Achieved FP64 performance vs # Threads / block")

ax.set_xticks(
    np.sort(
        flops_df['thr_block'].unique()
    )
)

ax.legend(title="# Events")

plt.savefig(
    FIGURES_DIR + 'flops_thr_block.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ------------------------------------------------------------------
# PLOT V: Memory bandwidth vs # threads/block
#
# kernel_ms = TOTAL kernel time for all events
#
# Bandwidth = total_bytes / total_kernel_time
#
# Error propagation:
#   sigma_BW / BW = sigma_T / T
# ------------------------------------------------------------------

# From Nsight Compute, order of magnitude
BYTES_PER_EVENT = (
    1.34e9       # DRAM read
    + 220.24e6   # DRAM write
) / 100000

bandwidth_df = (
    cudabench
    .groupby(['n_events', 'thr_block'])
    .agg(
        avg_kernel_ms=('kernel_ms', 'mean'),
        std_kernel_ms=('kernel_ms', 'std')
    )
    .reset_index()
)

# Convert total kernel time: ms -> s
bandwidth_df['kernel_sec'] = (
    bandwidth_df['avg_kernel_ms'] / 1000.0
)

bandwidth_df['std_kernel_sec'] = (
    bandwidth_df['std_kernel_ms'] / 1000.0
)

# Total bytes transferred
bandwidth_df['total_bytes'] = (
    BYTES_PER_EVENT * bandwidth_df['n_events']
)

# Achieved bandwidth
bandwidth_df['bandwidth_Bps'] = (
    bandwidth_df['total_bytes']
    / bandwidth_df['kernel_sec']
)

# Error propagation:
# sigma_BW = BW * sigma_T / T
bandwidth_df['std_bandwidth_Bps'] = (
    bandwidth_df['bandwidth_Bps']
    * bandwidth_df['std_kernel_sec']
    / bandwidth_df['kernel_sec']
)

# Convert B/s -> GB/s
bandwidth_df['bandwidth_GBs'] = (
    bandwidth_df['bandwidth_Bps'] / 1e9
)

bandwidth_df['std_bandwidth_GBs'] = (
    bandwidth_df['std_bandwidth_Bps'] / 1e9
)


# ------------------------------------------------------------------
# Bandwidth plot: one line for each n_events
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.grid(alpha=0.4)

events_values = np.sort(
    bandwidth_df['n_events'].unique()
)

for i, n in enumerate(events_values):

    df = (
        bandwidth_df[
            bandwidth_df['n_events'] == n
        ]
        .sort_values('thr_block')
    )

    ax.errorbar(
        df['thr_block'],
        df['bandwidth_GBs'],
        yerr=df['std_bandwidth_GBs'],
        color=colors[i % len(colors)],
        linestyle='--',
        marker=markers[i % len(markers)],
        capsize=2.5,
        label=str(n)
    )

ax.set_xlabel("# Threads / block")
ax.set_ylabel("Estimated DRAM bandwidth (GB/s)")
ax.set_title(
    "Estimated kernel memory bandwidth vs # Threads / block"
)

ax.set_xticks(
    np.sort(
        bandwidth_df['thr_block'].unique()
    )
)

ax.legend(title="# Events")

plt.savefig(
    FIGURES_DIR + 'bandwidth_thr_block.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()