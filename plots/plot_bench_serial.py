import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = '/mnt/POD/MCP_GD/JetClusteringAlgorithm/'
FIGURES_DIR = ROOT_DIR + 'plots/figures/benchmarks/serial/'
BENCH_DIR = ROOT_DIR + 'benchmarks/'

PER_EVENT_FILE = BENCH_DIR + 'serial_per_event_timings.csv'
TIMING_FILE = BENCH_DIR + 'serial_timings.csv'


colors = [  "#58508d", # violet
            "#bc5090", # purple
            "#ff6361", # orange
            "#ffa600"  # yellow
        ]

markers = ['o', 's', 'v', '^']


if __name__ == "__main__":

    # read data
    per_event_df = pd.read_csv(PER_EVENT_FILE)
    timing_df = pd.read_csv(TIMING_FILE)

    opt_levels = timing_df['opt_level'].unique()
    n_events = np.sort(timing_df['n_events'].unique())
    n_particles = np.sort(per_event_df['n_particles'].unique())

    # averaging total execution time over runs at fixed number of events for each opt level
    time_vs_events = timing_df.groupby(['opt_level', 'n_events']).agg(
        avg_exec_time = ('time_sec', 'mean'),
        std_exec_time = ('time_sec', 'std')
    )

    # averaging execution time per event (only for the maximum number of events)
    time_per_event_vs_npart = per_event_df[per_event_df['n_events']==100000]
    time_per_event_vs_npart = time_per_event_vs_npart.groupby(['opt_level', 'n_particles']).agg(avg_exec_time = ('time_sec', 'mean'), std_exec_time = ('time_sec', 'std'))

    # PLOT I: total execution time vs number of events for each opt level
    fig, ax = plt.subplots()
    ax.grid(alpha = 0.4)

    for i in range(len(opt_levels)):

        df = time_vs_events[time_vs_events.index.get_level_values('opt_level') == opt_levels[i]].sort_index(level='n_events')

        y = df['avg_exec_time']
        y_err = df['std_exec_time']

        ax.plot(n_events, y, color = colors[i], label = opt_levels[i], marker = markers[i])
        ax.errorbar(n_events, y, yerr = y_err, ecolor = colors[i], capsize = 0.5)

    ax.legend()
    ax.set_ylabel('Average execution time (s)')
    ax.set_xlabel('# Events')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(FIGURES_DIR+'time_vs_nevents.png', dpi = 300)


    # PLOT II: execution time per event vs number of particles

    fig, ax = plt.subplots()
    ax.grid(alpha = 0.4)

    for i in range(len(opt_levels)):

        df = time_per_event_vs_npart[time_per_event_vs_npart.index.get_level_values('opt_level') == opt_levels[i]].sort_index(level='n_particles')

        y = df['avg_exec_time']
        y_err = df['std_exec_time']

        ax.plot(n_particles, y, color = colors[i], marker = markers[i], label = opt_levels[i])
        ax.errorbar(n_particles, y, yerr = y_err, ecolor = colors[i], capsize = 0.5)

    ax.legend()
    ax.set_ylabel('Average execution time per event (s)')
    ax.set_xlabel('# Particles per event')

    plt.savefig(FIGURES_DIR+'time_per_event_vs_npart.png', dpi = 300)
