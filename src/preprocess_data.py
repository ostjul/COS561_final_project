import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys
from tqdm import tqdm

AVAILABLE_SCHEDULERS = ['FIFO', 'DRR', 'SP', 'WFQ']

def numpy_ewma_vectorized(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

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

        df = pd.read_csv(csv_path)
        
        # replace 'timestamp (sec)' with 'timestamp'
        if 'timestamp' not in df.columns:
            df = df.rename(columns={'timestamp (sec)': 'timestamp'})
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
                
                if verbose:
                    print("\tProcessing port {} ({}/{})".format(port, port_idx + 1, n_ports))
                cur_device_port_df = df.loc[(df['cur_port'] == port) & (df['cur_hub'] == device)].copy()
                len_data = len(cur_device_port_df)
                if verbose:
                    print("\tdevice {} port {} has {} rows".format(device, port, len_data))
                
                # BEGIN NEW CODE
                ingress_times = cur_device_port_df['timestamp'].values
                egress_times = cur_device_port_df['etime'].values
                # (N, N)
                loads = ((ingress_times[:,np.newaxis] < egress_times) & (ingress_times[:,np.newaxis] > ingress_times))
                # Get the number of bytes that were in the queue when each packet ingressed.
                load_bytes = loads @ cur_device_port_df['pkt_len'].values
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
                cur_device_port_df['load'] = load_bytes
                # calculate an EWMA of the loads to give the more context information to the packet.
                mean_load_device_port = cur_device_port_df['load'].ewm(alpha=0.1).mean() # Play around with this value.
                cur_device_port_df['mean_load_port_{}'.format(port)] = mean_load_device_port
                
                # Add sub-df to list of dfs
                processed_dfs.append(cur_device_port_df)
                
            if verbose:
                print("")
        # Concatenate data frames for each device/port combination
        processed_df = pd.concat(processed_dfs)
        
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
             
