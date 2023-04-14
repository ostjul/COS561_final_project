import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys

# x_labels should be ['pkt_len', 'cur_port', 'priority', 'flow_id', 'FIFO', 'DRR', 'SP', 'WFQ', 'load', 'mean_load_port_0', 'mean_load_port_1', 'mean_load_port_2', 'mean_load_port_3']
# y_label should be 'delay'

class TracesDataset(Dataset):
    def __init__(self,
                 csv_paths: list,
                 n_timesteps: int,
                 y_label: str= 'delay',
                 x_labels: list= ['pkt_len', 'cur_port', 'priority', 'flow_id', 'FIFO', 'DRR', 'SP', 'WFQ', 'load', 'mean_load_port_0', 'mean_load_port_1', 'mean_load_port_2', 'mean_load_port_3']):

        self.n_timesteps = n_timesteps
        self.indices = [] # Tuples of (csv_idx, device_idx, row_idx)

        # Store list of list of np.arrays for xs and ys
        self.xs = []
        self.ys = []
        
        # Iterate through all CSVs
        for csv_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            
            for x_label in x_labels:
                assert x_label in df.columns, "Column {} not found in {}".format(x_label, csv_path)
            assert y_label in df.columns, "Column {} not found in {}".format(y_label, csv_path)
            
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
                xs_device = device_data[x_labels].to_numpy(dtype=np.float32)
                ys_device = device_data[y_label].to_numpy(dtype=np.float32)

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

        self.num_feat = self.xs[0][0].shape[-1]

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