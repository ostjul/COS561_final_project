import numpy as np
import pandas as pd

import ray


block_size = 42 # TODO: Get this from the spec file

def scrub(df):
    # Zeros out all the etimes to ensure we don't have knowledge of egress times during inference.
    df['etime'] = np.zeros(len(df))
    return df

@ray.remote
class Device:
 
    def __init__(self, id, df):

        self.id = id
        self.df = df
        self.timestamp = 0.0
        self.num_updates = 0

    def forward_block(self, model, port_to_nexthop, flow_to_port):
        # Get the next block of packets to forward from this device.
        block = self.df[self.df['timestamp'] > self.timestamp].iloc[:block_size].copy() 
        if len(block) == 0:
            # No update can be done
            return self.timestamp, {}

        # TODO: Need to do our block preprocessing here.
        # processed_block = ...
        
        # TODO: Predict the egress times using the trained network.
        predicted_delays = np.ones(len(block)) * 0.003 # model(processed_block)

        # Don't actually simulate links. (this will only work for fattree16, should add an is_link flag to the device class for more generic behavior)
        etimes = (block['timestamp'] + predicted_delays) if self.id < 20 else block['etime']
    
        block['cur_hub'] = block.apply(lambda x: port_to_nexthop[self.id][x['cur_port']], axis=1)
        block['cur_port'] = block.apply(lambda x: (flow_to_port[x['cur_hub']]).get(x['flow_id'], -1), axis=1)
        block['path'] = block.apply(lambda x: f"{x['path']}-{x['cur_hub']}_{x['cur_port']}", axis=1)

        # Add the sojourn times to get the updated timestamps
        block['timestamp'] = etimes

        next_t = block['timestamp'].iloc[-1]
        # Get rid of all packets at their final destination.
        block = block.loc[(block['cur_port'] != -1)]

        # Now convert to forwarding dict
        forwarding_dict = {d: block.loc[block['cur_hub'] == d] for d in pd.unique(block['cur_hub'])}

        # Update the etimes for the packets in this device.
        tmp_df = self.df.copy()
        tmp_df.loc[(tmp_df['timestamp'] <= next_t) & (tmp_df['timestamp'] > self.timestamp), 'etime'] = etimes
        self.df = tmp_df
    
        # Note: We don't update the status, since it may be the case that we have to let other devices 'catch up' their timestamps.
        return next_t, forwarding_dict

    def update_time(self, new_time):
        self.timestamp = new_time
        self.num_updates += 1

    def add_new_packets(self, new_packets):
        self.df = pd.concat([self.df, *new_packets]).sort_values(['timestamp'])

    def get_df(self):
        return self.df
