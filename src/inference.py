import numpy as np
from opt_einsum import contract # Makes things run much faster
import pandas as pd
import torch

import ray

from preprocess_data import numpy_ewma_vectorized, AVAILABLE_SCHEDULERS
from dataset import x_labels_mid

# TODO: This should really be adjustable to variable number of ports.
x_labels = ['timestamp'] + x_labels_mid.copy() + [f'mean_load_port_{i}' for i in range(4)]
# TODO: Get this from the config file
block_size = 42

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

            processed_block = processed_df[x_labels].values.astype(np.float32).reshape((-1, block_size, len(x_labels)))

            predicted_delays = model(torch.tensor(processed_block)).reshape(-1).detach().numpy()
            block['timestamp'] = block['timestamp'] + predicted_delays
        else:
            # TODO: For links, we may eventually want to compute them using the formula instead of using sim results.
            block['timestamp'] = block['etime']
        
        tmp_df = self.df.copy()
        tmp_df['etime'] = block['timestamp']
        self.df = tmp_df

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
