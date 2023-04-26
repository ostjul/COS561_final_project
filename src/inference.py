import numpy as np
import pandas as pd


block_size = 42 # TODO: Get this from the paper.

def scrub(df):
    # Zeros out all the etimes to ensure we don't have knowledge of egress times during inference.
    df['etime'] = np.zeros(len(df))
    return df

# TODO: Make this a Ray actor
class Device:
 
    def __init__(self, id, df):

        self.id = id
        self.timestamp = 0.0
        self.num_updates = 0
        self.df = df

    def forward_block(self, model, port_to_nexthop, flow_to_port):
        # Get the next block of packets to forward from this device.
        block = self.df[self.df['timestamp'] > self.timestamp].iloc[:block_size].copy() 
        if len(block) == 0:
            return (self.timestamp, self.df), {}

        # TODO: Need to do our block preprocessing here.
        # processed_block = ...
        
        # TODO: Predict the egress times using the trained network.
        predicted_delays = np.ones(len(block)) * 0.003 # model(processed_block)

        # Don't actually simulate links. (this will only work for fattree16, should add an is_link flag to the device class for more generic behavior)
        etimes = (block['timestamp'] + predicted_delays) if self.id < 20 else block['etime']

        # TODO: Can we vectorize these?
        block['cur_hub'] = block.apply(lambda x: port_to_nexthop[self.id][x['cur_port']], axis=1)
        block['cur_port'] = block.apply(lambda x: (flow_to_port[x['cur_hub']]).get(x['flow_id'], -1), axis=1)
        block['path'] = block.apply(lambda x: f"{x['path']}-{x['cur_hub']}_{x['cur_port']}", axis=1)

        # Add the sojourn times to get the updated timestamps
        block['timestamp'] = etimes

        # Filter all the entries where cur_port == -1, as this indicates that the packet has arrived at its final destination
        next_t = block['timestamp'].iloc[-1]
        block = block.loc[block['cur_port'] != -1]

        # Now convert to forwarding dict
        forwarding_dict = {d: block.loc[block['cur_hub'] == d] for d in pd.unique(block['cur_hub'])}

        # Update the egress times for the current dataframe
        
        self.df.loc[(self.df['timestamp'] <= next_t) & (self.df['timestamp'] > self.timestamp), 'etime'] = etimes
        # Note: We don't update the status, since it may be the case that we have to let other devices 'catch up' their timestamps.
        return next_t, forwarding_dict

    def update_time(self, new_time):
        self.timestamp = new_time
        self.num_updates += 1

    def add_new_packets(self, new_packets):
        self.df = pd.concat([self.df, new_packets]).sort_values(['timestamp'])
