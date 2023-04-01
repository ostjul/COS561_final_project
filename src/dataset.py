import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys



class TracesDataset(Dataset):
    def __init__(self,
                 csv_paths: list,
                 n_timesteps: int,
                 y_label: str):

        self.n_timesteps = n_timesteps
        self.indices = [] # Tuples of (csv_idx, device_idx, row_idx)

        # Store list of list of np.arrays for xs and ys
        self.xs = []
        self.ys = []

        # Iterate through all CSVs
        for csv_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)

            xs_csv = []
            ys_csv = []

            # Group by unique devices
            unique_devices = df['cur_hub'].unique()
            # Iterate through rows for each device
            for device_idx, unique_device in enumerate(unique_devices):
                # Obtain rows with this device and sort by etime
                device_data = df.loc[df['cur_hub'] == unique_device]
                device_data = device_data.sort_values(['etime'])

                # Separate x and y values for this device
                xs_device = device_data.drop(y_label, axis=1).to_numpy()
                ys_device = device_data[y_label].to_numpy()

                # Append to list of data for each csv
                xs_csv.append(xs_device)
                ys_csv.append(ys_device)

                # Calculate indices for timeseries
                n_rows = device_data.shape[0]
                n_timeseries_data = n_rows - self.n_timesteps + 1
                # Create tuples using CSV index, device index, and rows
                timeseries_idxs = [(csv_idx, device_idx, row_idx) for row_idx in range(n_timeseries_data)]
                self.indices += timeseries_idxs

            # Append data for this csv to master lists
            self.xs.append(xs_csv)
            self.ys.append(ys_csv)

    def __getitem__(self, index):
        # Obtain the index for CSV, device, and row start
        csv_idx, device_idx, row_start_idx = self.indices[index]
        # Index into Xs and Ys for timeseries data
        xs = self.xs[csv_idx][device_idx][row_start_idx:row_start_idx + self.n_timesteps]
        ys = self.ys[csv_idx][device_idx][row_start_idx:row_start_idx + self.n_timesteps]

        # Return data and labels
        return xs, ys

    def __len__(self):
        return len(self.indices)
'''

class TracesDatasetOld(Dataset):
    def __init__(self,
                 csv_path: str,
                 n_timesteps: int):

        # Assume the csv is an output of ns.py
        df = pd.read_csv(csv_path)
        df = df.sort_values(['cur_hub', 'cur_port', 'etime'])
        df['time_diff'] = df['etime'] - df['timestamp (sec)']
        x_cols = [
            'index', #PID?
            'pkt len (byte)', # packet length
            'priority',
            'src_pc',
            'cur_port' # in port
        ]
        y_col = ['time_diff']

        # x_df = df[x_cols]
        # y_df = df[y_cols]
        x_timeseries = []
        y_timeseries = []
        unique_ports = df['cur_port'].unique()
        unique_hubs = df['cur_hub'].unique()
        for hub in unique_hubs:
            for port in unique_ports:

                cur_hub_port_df = df.loc[(df['cur_port'] == port) & (df['cur_hub'] == hub)]
                len_data = len(cur_hub_port_df)

                print("hub {} port {} has {} data points".format(hub, port, len_data))
                if len_data < n_timesteps:
                    raise ValueError("Timestep ({}) must be less than number of data points for current hub".format(len_data))


                n_rows = len_data - n_timesteps + 1
                x_data = []
                y_data = cur_hub_port_df[y_col]
                for i in range(n_rows):
                    x_data = cur_hub_port_df.iloc[i:i+n_timesteps][x_cols].to_numpy()
                    y_data = cur_hub_port_df.iloc[i:i+n_timesteps][y_col].to_numpy()
                    x_timeseries.append(x_data)
                    y_timeseries.append(y_data)
                    # print(x_data.shape, y_data.shape)

        self.x_timeseries = np.stack(x_timeseries, axis=0)
        self.y_timeseries = np.stack(y_timeseries, axis=0)
        print(self.x_timeseries.shape, self.y_timeseries.shape)

    def __getitem__(self, index):
        x = self.x_timeseries[index]
        y = self.y_timeseries[index]

        return x, y
    def __len__(self):
        return self.x_timeseries.shape[0]
        '''