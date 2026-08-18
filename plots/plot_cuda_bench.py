import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = '/mnt/POD/MCP_GD/JetClusteringAlgorithm/'
FIGURES_DIR = ROOT_DIR + 'plots/figures/benchmarks/cuda/'
BENCH_DIR = ROOT_DIR + 'benchmarks/'

TIMING_FILE = BENCH_DIR + 'results_cuda_raw.csv'


colors = [  "#58508d", # violet
            "#bc5090", # purple
            "#ff6361", # orange
            "#ffa600",  # yellow
            "#94d252",
            "#5ec1ff",
            "#df99ff"
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