import numpy as onp
import pandas as pd

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def plot_egress_histogram(df, devices, dt=1, hist_window_seconds=2):
    """
    Returns an animation of the per device histograms of egress times.
    df - DataFrame of packet traces 
    devices - list of device ids
    dt - how many seconds to advance each frame in the animation 
    hist_window_seconds - sliding window size for lookback to generate histograms
    """
    fig, axes = plt.subplots(len(devices))
    fig.set_size_inches(8, 3 * len(devices))

    # Get all the packets associated with a given device
    df_ = df.loc[df['cur_hub'].isin(devices)]
    all_bins = {}
    for d in devices:
        df_device = df_.loc[df_['cur_hub'] == d]
        delta_values = (df_device['etime'] - df_device['timestamp']).values
        bins = onp.linspace(0.0, delta_values.max(), 50)
        all_bins[d] = bins
        print(delta_values.min(), delta_values.max())
    priorities = sorted(pd.unique(df_['priority']))

    # Compute the number of frames to show in our animation
    min_time, max_time = df_['timestamp'].min(), df_['etime'].max()
    n_frames = 1 + int((max_time - min_time - hist_window_seconds) // dt)
    def animate(frame):
        start = frame * dt
        end = frame * dt + hist_window_seconds
        legend = False
        for ax, device in zip(axes, devices):
            ax.clear()
            # Select all entries where egress happened between start and end
            sub_df = df_.loc[(df_['cur_hub'] == device) & (df_['etime'] <= end) & (df_['etime'] >= start)]
            for w in priorities:
                priority_df = sub_df.loc[sub_df['priority'] == w]
                delta_times = (priority_df['etime'] - priority_df['timestamp']).values
                ax.hist(delta_times, bins=all_bins[device], label=f'{w}', alpha=(1/len(priorities)) ** 0.5)
            ax.set_ylabel(f"Switch #{device} traffic")
            if not legend:
                ax.legend(title="Priority")
                legend = True
        fig.suptitle(f'Packet Egress times between {start} and {end}')

    anim = FuncAnimation(fig, animate, frames=n_frames)
    return anim