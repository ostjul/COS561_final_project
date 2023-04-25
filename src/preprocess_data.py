import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys
from tqdm import tqdm

import datetime as dt

AVAILABLE_SCHEDULERS = ['FIFO', 'DRR', 'SP', 'WFQ']
def preprocess_csvs(csv_paths: list,
                    verbose: bool,
                    csv_save_dir: str=None):

    non_existent_paths = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            non_existent_paths.append(csv_path)

    if len(non_existent_paths) > 0:
        raise ValueError("{} paths in csv_paths do not exist: {}".format(len(non_existent_paths), non_existent_paths))

    n_csvs = len(csv_paths)
    for csv_idx, csv_path in enumerate(csv_paths):
        print("Processing {}/{} csv: {}".format(csv_idx + 1, n_csvs, csv_path))

        df = df = pd.read_csv(csv_path)
        df = df.sort_values(['cur_hub', 'cur_port', 'timestamp'])

        unique_ports = df['cur_port'].unique()
        n_ports = len(unique_ports)
        unique_devices = df['cur_hub'].unique()
        n_devices = len(unique_devices)
        
        # Data structure for storing rows with new load column
        processed_dfs = []
        for device_idx, device in tqdm(enumerate(unique_devices), total=n_devices):
            if verbose:
                print("Processing device {} ({}/{})".format(device, device_idx + 1, n_devices))
            for port_idx, port in enumerate(unique_ports):
                start_time_2 = dt.datetime.now()
                if verbose:
                    print("\tProcessing port {} ({}/{})".format(port, port_idx + 1, n_ports))
                cur_device_port_df = df.loc[(df['cur_port'] == port) & (df['cur_hub'] == device)].copy()
                len_data = len(cur_device_port_df)
                if verbose:
                    print("\tdevice {} port {} has {} rows".format(device, port, len_data))
                
                # BEGIN NEW CODE
                ingress_times = cur_device_port_df['timestamp'].values
                egress_times = cur_device_port_df['etime'].values
                cur_device_port_df['load'] = ((ingress_times[:,np.newaxis] < egress_times) & (egress_times[:,np.newaxis] > ingress_times)).sum(axis=1)
                # END NEW CODE   

                # BEGIN OLD CODE 
                # Calculate the load at the current port for each row
                # loads = []
                # for row_idx, row in cur_device_port_df.iterrows():
                #     ingress_time = row['timestamp']
                #     egress_time = row ['etime']
                #     # load = count number of rows that have timestamp or etime between [ingress, egress]
                #     load_rows = cur_device_port_df[
                #         # other row start in the middle of current row
                #         ((cur_device_port_df['timestamp'] >= ingress_time) & (cur_device_port_df['timestamp'] < egress_time)) | 
                #         # other row ends in middle of current row
                #         ((cur_device_port_df['etime'] >= ingress_time) & (cur_device_port_df['etime'] < egress_time)) | 
                #         # other row starts before current row and ends after current row
                #         ((cur_device_port_df['timestamp'] < ingress_time) & (cur_device_port_df['etime'] > egress_time))]
                #     # Count number of rows that match the criteria
                #     load = len(load_rows)
                #     loads.append(load)
                # Assign load column and append to list of dataframe 
                # cur_device_port_df['load'] = loads
                # END OLD CODE

                processed_dfs.append(cur_device_port_df)
                print("Time taken: {}".format(dt.datetime.now() - start_time_2))
            if verbose:
                print("")
        # Concatenate data frames for each device/port combination
        processed_df = pd.concat(processed_dfs)
        
        # Calculate average load for each port
        for port in unique_ports:
            avg_load = np.mean(processed_df[processed_df['cur_port'] == port]['load'].to_numpy())
            processed_df['mean_load_port_{}'.format(port)] = avg_load
            if verbose:
                print("average load for port {}: {}".format(port, avg_load))
        
        # Calculate the delay time (predicted variable)
        processed_df['delay'] = processed_df['etime'] - processed_df['timestamp']
        
        # One hot encode the scheduler
        # scheduler = processed_df['scheduler']
        if 'scheduler' in processed_df:
            one_hot_scheduler = pd.get_dummies(processed_df['scheduler'])
            processed_df = pd.concat([processed_df, one_hot_scheduler], axis=1)
        
        # If each available scheduler is not present in the DF, set column to 0
        zeros = np.zeros(len(processed_df))
        for scheduler in AVAILABLE_SCHEDULERS:
            if scheduler not in processed_df.columns:
                processed_df[scheduler] = zeros
        
        # Save new CSV
        csv_save_name = os.path.splitext(os.path.basename(csv_path))[0] + '_processed.csv'
        csv_save_path = os.path.join(csv_save_dir, csv_save_name)
        
        processed_df.to_csv(csv_save_path)
        print("Saved processed csv to {}\n".format(csv_save_path))
             
