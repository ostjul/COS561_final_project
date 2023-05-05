import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os, sys
from tqdm import tqdm

AVAILABLE_SCHEDULERS = ['FIFO', 'DRR', 'SP', 'WFQ']

def numpy_ewma_vectorized(data, alpha):
    # alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/(1e-9 + pows[:-1])
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def preprocess_csvs(csv_paths: list,
                    verbose: bool,
                    csv_save_dir: str=None,
                    link_ids=None):

    non_existent_paths = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            non_existent_paths.append(csv_path)
            
    # IDs for link
    if link_ids is None:
        link_ids = [i for i in range(20,36)]
        
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
            # Skip links
            if device in link_ids:
                continue
            mean_loads = {}
            device_dfs = []
            if verbose:
                print("Processing device {} ({}/{})".format(device, device_idx + 1, n_devices))
            for port_idx, port in enumerate(unique_ports):
                
                if verbose:
                    print("\tProcessing port {} ({}/{})".format(port, port_idx + 1, n_ports))
                cur_device_port_df = df.loc[(df['cur_port'] == port) & (df['cur_hub'] == device)].copy()
                len_data = len(cur_device_port_df)
                if verbose:
                    print("\tdevice {} port {} has {} rows".format(device, port, len_data))
                
                ingress_times = cur_device_port_df['timestamp'].values
                egress_times = cur_device_port_df['etime'].values
                loads = ((ingress_times[:,np.newaxis] < egress_times) & (ingress_times[:,np.newaxis] > ingress_times))
                # Get the number of bytes that were in the queue when each packet ingressed.
                if 'pkt len (byte)' in cur_device_port_df.columns:
                    load_bytes = loads @ cur_device_port_df['pkt len (byte)'].values
                else:
                    load_bytes = loads @ cur_device_port_df['pkt_len'].values
                
                # Assign load column and append to list of dataframe 
                cur_device_port_df['load'] = load_bytes
                # calculate an EWMA of the loads to give the more context information to the packet.
                if len(load_bytes) > 0:
                    mean_load_device_port = numpy_ewma_vectorized(load_bytes, 0.05) # Play around with this value.
                cur_device_port_df['mean_load_port_{}'.format(port)] = mean_load_device_port
                
                # Add sub-df to list of dfs
                device_dfs.append(cur_device_port_df)
            device_dfs = pd.concat(device_dfs)
            
            for port, mean_load in mean_loads.items():
                device_dfs['mean_load_port_{}'.format(port)] = mean_load
            
            # Fill NaNs with 0's
            device_dfs = device_dfs.fillna(0)
            processed_dfs.append(device_dfs)
                
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
             
