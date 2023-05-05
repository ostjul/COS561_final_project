import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys

# x_labels should be ['pkt_len', 'cur_port', 'priority', 'flow_id', 'FIFO', 'DRR', 'SP', 'WFQ', 'load', 'mean_load_port_0', 'mean_load_port_1', 'mean_load_port_2', 'mean_load_port_3']
# y_label should be 'delay'

x_labels_fid = ['pkt_len', 'cur_port', 'priority', 'flow_id', 'FIFO', 'DRR', 'SP', 'WFQ', 'load']
x_labels_small = ['pkt_len', 'cur_port', 'priority', 'load']
x_labels_mid = ['pkt_len', 'cur_port', 'priority', 'FIFO', 'DRR', 'SP', 'WFQ', 'load']

# non-links
useful_hubs = [12,13,14,15,16,17,18,19]


class TracesDataset(Dataset):
    def __init__(self,
                 csv_paths: list,
                 n_timesteps: int,
                 y_label: str= 'delay',
                 x_labels: list= x_labels_mid.copy(),
                 use_norm_time=False
                 ):
        
        if use_norm_time:
            if 'timestamp' not in x_labels:
                x_labels.insert(0, 'timestamp')
        
        self.n_timesteps = n_timesteps
        self.indices = [] # Tuples of (csv_idx, device_idx, row_idx)

        # Store list of list of np.arrays for xs and ys
        self.xs = []
        self.ys = []

        # Add load of all ports to x_labels
        first_df =  pd.read_csv(csv_paths[0])
        max_port_number = np.int(first_df.cur_port.max())
        for port_idx in range(max_port_number+1):
            new_port_load = 'mean_load_port_{}'.format(port_idx)
            if new_port_load not in x_labels:
                x_labels.append(new_port_load)

        self.x_labels = x_labels

        # Iterate through all CSVs
        for csv_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)

            df = df[df['cur_hub'].isin([12,13,14,15,16,17,18,19])]

            if 'pkt len (byte)' in df.columns:
                df = df.rename({'pkt len (byte)': 'pkt_len'}, axis='columns')

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
                # device_data = device_data.sort_values(['etime'])
                device_data = device_data.sort_values(['timestamp'])                

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

        self.num_feat = len(self.x_labels)

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
    
class HistogramTracesDataset(Dataset):
    def __init__(self,
                 csv_paths: list,
                 n_timesteps: int,
                 y_label: str= 'delay',
                 x_labels: list= x_labels_mid.copy()
                #  use_norm_time=False
                 ):
        
        # if use_norm_time:
        #     if 'timestamp' not in x_labels:
        #         x_labels.insert(0, 'timestamp')
        x_labels.insert(0, 'timestamp') # we always want to include timestamps
        
        self.n_timesteps = n_timesteps
        self.indices = [] # Tuples of (csv_idx, device_idx, row_idx)

        # Store list of list of np.arrays for xs and ys
        self.xs = []
        self.ys = []

        # Add load of all ports to x_labels
        first_df = pd.read_csv(csv_paths[0])
        max_port_number = first_df.cur_port.max()
        for port_idx in range(max_port_number+1):
            new_port_load = 'mean_load_port_{}'.format(port_idx)
            if new_port_load not in x_labels:
                x_labels.append(new_port_load)

        self.x_labels = x_labels
        
        
        # Iterate through all CSVs
        for csv_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)

            
            df = df[df['cur_hub'].isin(useful_hubs)]

            if 'pkt len (byte)' in df.columns:
                df = df.rename({'pkt len (byte)': 'pkt_len'}, axis='columns')

            for x_label in x_labels:
                assert x_label in df.columns, "Column {} not found in {}".format(x_label, csv_path)
            assert y_label in df.columns, "Column {} not found in {}".format(y_label, csv_path)

            xs_csv = []
            ys_csv = []

            # Group by unique devices
            unique_devices = df['cur_hub'].unique()

            csv_idx = 0
            # Iterate through rows for each device
            for device_idx, unique_device in enumerate(unique_devices):
                # Obtain rows with this device and sort by etime
                device_data = df.loc[df['cur_hub'] == unique_device]
                # device_data = device_data.sort_values(['etime'])
                device_data = device_data.sort_values(['timestamp'])                

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
                start_idxs = list(range(0, n_timeseries_data, self.n_timesteps))
                # Cut off last bit
                if start_idxs[-1] + self.n_timesteps > n_timeseries_data:
                    start_idxs = start_idxs[0:-1]
                timeseries_idxs = [(csv_idx, device_idx, row_idx) for row_idx in start_idxs]
                self.indices += timeseries_idxs

        # Append data for this csv to master lists
        self.xs.append(xs_csv)
        self.ys.append(ys_csv)

        self.num_feat = len(self.x_labels)
        
    def __getitem__(self, index):
        # Obtain the index for CSV, device, and row start
        csv_idx, device_idx, row_start_idx = self.indices[index]
        # Index into Xs and Ys for timeseries data
        xs = self.xs[csv_idx][device_idx][row_start_idx:row_start_idx + self.n_timesteps]
        ys = self.ys[csv_idx][device_idx][row_start_idx:row_start_idx + self.n_timesteps]

        # device_idx are coded using indices of useful_hubs
        # map back to the original hub values using values as each useful_hubs index
        # index_map = {i:x for i,x in enumerate(useful_hubs)} 
        hub = useful_hubs[device_idx] 

        

        # Return data and labels
        return xs, ys, hub
    
    def __len__(self):
        return len(self.indices)