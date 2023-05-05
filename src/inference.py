import numpy as np
from opt_einsum import contract # Makes things run much faster
import pandas as pd
import torch
import cloudpickle
import json
import datetime as dt

import ray

import networkx as nx

from preprocess_data import numpy_ewma_vectorized, AVAILABLE_SCHEDULERS
from dataset import x_labels_mid
from ptm import deepPTM, load_model_from_ckpt



# TODO: This should really be adjustable to variable number of ports.
x_labels = ['timestamp'] + x_labels_mid.copy() + [f'mean_load_port_{i}' for i in range(4)]
# TODO: Get this from the config file
block_size = 42


def to_matrix(dict_to_convert, shape):
    matrix = np.zeros(shape, dtype=np.int32)
    for key, value in dict_to_convert.items():
        matrix[key, value] = 1
    return matrix

def to_tensor(dict_to_convert, shape):
    tensor = np.zeros(shape, dtype=np.int32)
    for key, sub_dict in dict_to_convert.items():
        tensor[int(key)] = to_matrix(sub_dict, shape[1:])
    return tensor

# @ray.remote
class Device:
 
    def __init__(self, id, df, port_to_next_hop, flow_to_port):

        self.id = id
        self.df: pd.DataFrame = df
        self.port_to_next_hop = port_to_next_hop
        self.flow_to_port = flow_to_port
        self.forwarded_ids = set()

    def initial_forward(self):
        
        # Get all the packets we haven't forwarded yet
        block = self.df.loc[~self.df.index.isin(self.forwarded_ids)].copy()
        if len(block) == 0:
            return {}
        
        self.forwarded_ids.update(block.index.values)
        
        # Update the devices & ports in a vectorized way (about 20x faster than pd.apply())
        new_hubs_one_hot = np.eye(self.port_to_next_hop.shape[0])[block['cur_port'].values.astype(np.int32)] @ self.port_to_next_hop
        block['cur_hub'] = np.argmax(new_hubs_one_hot, axis=-1)

        one_hot_flows =  np.eye(self.flow_to_port.shape[1])[block['flow_id'].values.astype(np.int32)]
        new_ports_one_hot = contract('nd,dfk,nf->nk', new_hubs_one_hot, self.flow_to_port, one_hot_flows)
        new_ports = np.argmax(new_ports_one_hot, axis=-1)
        block['cur_port'] = np.where(new_ports_one_hot.sum(axis=-1), new_ports, -np.ones_like(new_ports))

        block = block.loc[block['cur_port'] != -1]

        # Now convert to forwarding dict
        forwarding_dict = {d: block.loc[block['cur_hub'] == d] for d in pd.unique(block['cur_hub'])}
        return forwarding_dict

    def forward(self, model):
        # Get the next block of packets to forward from this device.
        block = self.df.copy()
        # Ensure that the block is divisible by the block size 
        block = block.iloc[:-(len(block) % block_size)]
        block_idxes = block.index
        assert len(block) % block_size == 0

        if len(block) == 0:
            return {}

        if self.id < 20:
            # Block Preprocessing 
            processed_dfs = []
            # TODO: Vectorize this for loop, will probably be messy but this is where the performance bottleneck currently lies.
            for port in range(4):
                port_df = block.loc[block['cur_port'] == port].copy()
                ingress_times = port_df['timestamp'].values
                egress_times = port_df['etime'].values
                loads = ((ingress_times[:,np.newaxis] < egress_times) & (ingress_times[:,np.newaxis] > ingress_times)).astype(np.int32)
                load_bytes = loads @ port_df['pkt_len'].values.astype(np.int32)
                port_df['load'] = load_bytes
                mean_load_device_port = numpy_ewma_vectorized(load_bytes, 0.05)
                port_df[f'mean_load_port_{port}'] = mean_load_device_port
                processed_dfs.append(port_df)
            processed_df = pd.concat(processed_dfs)
            if 'scheduler' in processed_df:
                one_hot_scheduler = pd.get_dummies(processed_df['scheduler'])
                processed_df = pd.concat([processed_df, one_hot_scheduler], axis=1)
            
            # If each available scheduler is not present in the DF, set column to 0
            zeros = np.zeros(len(processed_df))
            for scheduler in AVAILABLE_SCHEDULERS:
                if scheduler not in processed_df.columns:
                    processed_df[scheduler] = zeros
            
            processed_block = processed_df[x_labels].fillna(0).values.astype(np.float32).reshape((-1, block_size, len(x_labels)))
            # Make sure that the egress times are > 0. 1e-7 seems to be a good lower bound based on empirical observation.
            predicted_delays = np.maximum(model(torch.tensor(processed_block)).reshape(-1).detach().numpy(), 1e-7)
            etimes = block['timestamp'].values + predicted_delays
        else:
            # TODO: For links, we may eventually want to compute them using the formula instead of using sim results.
            etimes = block['etime'].values
        
        # tmp_df = self.df.copy()
        # tmp_df.loc[block_idxes, 'etime'] = etimes
        self.df.loc[block_idxes, 'etime'] = etimes

        block['timestamp'] = etimes
        block.drop(columns=['etime'], inplace=True)

        # Update the devices & ports in a vectorized way (about 20x faster than pd.apply())
        new_hubs_one_hot = np.eye(self.port_to_next_hop.shape[0])[block['cur_port'].values.astype(np.int32)] @ self.port_to_next_hop
        block['cur_hub'] = np.argmax(new_hubs_one_hot, axis=-1)

        one_hot_flows =  np.eye(self.flow_to_port.shape[1])[block['flow_id'].values.astype(np.int32)]
        new_ports_one_hot = contract('nd,dfk,nf->nk', new_hubs_one_hot, self.flow_to_port, one_hot_flows)
        new_ports = np.argmax(new_ports_one_hot, axis=-1)
        block['cur_port'] = np.where(new_ports_one_hot.sum(axis=-1), new_ports, -np.ones_like(new_ports))
        # filter all the packets that have reached their final destination
        block = block.loc[block['cur_port'] != -1]

        # Now convert to forwarding dict
        forwarding_dict = {d: block.loc[block['cur_hub'] == d] for d in pd.unique(block['cur_hub'])}
        return forwarding_dict
    

    def aggregate(self, forward_dicts, initial=False):
        updates = []
        for forward_dict in forward_dicts:
            if self.id in forward_dict and len(forward_dict[self.id]) > 0:
                updates.append(forward_dict[self.id])
        if len(updates) > 0:
            old_df = self.df.copy()
            if initial:
                cat_df = pd.concat([self.df, *updates])
                self.df = cat_df[~cat_df.index.duplicated(keep='last')].sort_values('timestamp')
            else:
                self.df.update(pd.concat(updates))
            # Return true only if the dataframe has changed
            return not old_df.equals(self.df)
        else:
            return False

    def get_df(self):
        return self.df
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_next_timestamp(self):
        return self.next_timestamp
    


def run_simulation(G, trace_path, config_path, model_path):
    """
    Actually runs the inference loop for the simulation. Only fattree topology is currently supported.
    """

    # Load the config and the model
    with open(config_path) as f:
        specs = json.load(f)
   
    model_specs = specs["model_specs"]
    model = deepPTM(in_feat=13, # TODO: Get this from config somehow
                    lstm_config=model_specs["lstm_config"],
                    attn_config=model_specs["attn_config"],
                    time_steps=specs["n_timesteps"],
                    use_norm_time=specs['use_norm_time'])
    load_model_from_ckpt(model, model_path)
    model.eval()

    # Load the trace and the forwarding table
    df = pd.read_csv(f'{trace_path}.csv')
    with open(f'{trace_path}.flow_to_port', 'rb') as f:
        flow_to_port = cloudpickle.load(f)
    with open(f'{trace_path}.port_to_nexthop', 'rb') as f:
        port_to_nexthop = cloudpickle.load(f)

    all_devices = pd.unique(df['cur_hub'])
    num_ports = len(pd.unique(df['cur_port']))
    num_flows = len(pd.unique(df['flow_id']))

    port_to_nexthop_tensor = to_tensor(port_to_nexthop, (len(all_devices), num_ports, len(all_devices)))
    flow_to_port_tensor = to_tensor(flow_to_port, (len(all_devices), num_flows, num_ports))

    initial_packets = df.loc[df['cur_hub'] >= 20].copy()
    initial_devices = pd.unique(initial_packets['cur_hub'])
    # Give all the packets a pid
    initial_packets['pid'] = np.arange(len(initial_packets))
    initial_packets.set_index('pid', inplace=True)

    devices = [None] * len(all_devices)
    for d in all_devices:
        if d in initial_devices:
            device_initial_pkts = initial_packets.loc[initial_packets['cur_hub'] == d]
            devices[d] = Device(d, device_initial_pkts, port_to_nexthop_tensor[d], flow_to_port_tensor)
        else:
            devices[d] = Device(d, pd.DataFrame(columns=df.columns), port_to_nexthop_tensor[d], flow_to_port_tensor)

    print("Simulation Setup, running IRSA")
    start_time = dt.datetime.now()
    # IRSA pt. 1, forwarding all the initial packets
    diam_G = nx.diameter(G)
    for _ in range(diam_G):
        # Forward in parallel, and aggregate in parallel.
        forward_dicts = [dev.initial_forward() for dev in devices]
        res = [dev.aggregate(forward_dicts, initial=True) for dev in devices]
        # print(any(res))
        if not any(res):
            break

    # IRSA pt. 2
    for _ in range(diam_G):
        forward_dicts = [dev.forward(model) for dev in devices]
        res = [dev.aggregate(forward_dicts) for dev in devices]
        # Break if no updates were made.
        if not any(res):
            break

    print(f"Simulation took {(dt.datetime.now() - start_time).total_seconds()} seconds")
    # Now we can get the final dataframe simply by extracting the dataframes, concatentating, and re-sorting on timestamp.
    final_df = pd.concat([dev.get_df() for dev in devices]).sort_values('timestamp')
    return final_df

    