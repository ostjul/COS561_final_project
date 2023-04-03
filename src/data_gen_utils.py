import pandas as pd
from functools import reduce

# This merges multiple flow dataframes
def merge_flow_dfs(flow_dfs):
    return pd.concat(flow_dfs).sort_values(by='timestamp')

# This generates a pandas dataframe from a given flow
def flow_to_df(flow_id, flow):
    perhop_times = flow.pkt_sink.perhop_times[flow_id]
    arrival_times = flow.pkt_sink.arrivals[flow_id] 
    pkt_sizes = flow.pkt_sink.packet_sizes[flow_id]

    data_dict = {
        'timestamp': [],
        'pkt_len': [],
        'cur_hub': [],
        'cur_port': [],
        'path': [],
        'priority': [],
        'flow_id': [],
        'etime': [],
    }
        
    # Add the information for the rows of our df.
    for packet_hops, arrival_time in zip(perhop_times, arrival_times):
        a = sorted([(ts, dp) for dp, ts in packet_hops.items()])
        current_path = ''
        for i, (ts, dp) in enumerate(a):
            current_path += dp
            cur_hub, cur_port = tuple(dp.split('_'))
            etime = arrival_time if i == len(a) - 1 else a[i + 1][0]
            data_dict['timestamp'].append(ts)
            data_dict['pkt_len'].append(pkt_sizes[i])
            data_dict['cur_hub'].append(cur_hub)
            data_dict['cur_port'].append(cur_port)
            data_dict['path'].append(current_path)
            data_dict['priority'].append(0)
            data_dict['flow_id'].append(flow_id)
            data_dict['etime'].append(etime)

            current_path += '-'

    
    
    # Make the data dict
    df = pd.DataFrame(data_dict)

    return df