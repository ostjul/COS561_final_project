from functools import partial
from random import expovariate, sample
import random
from scipy import stats


import os
import argparse
import cloudpickle
import logging

import numpy as np
import simpy
import pandas as pd

import networkx as nx

from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.trace_generator import TracePacketGenerator
from ns.packet.sink import PacketSink
from ns.switch.switch import SimplePacketSwitch
from ns.switch.switch import FairPacketSwitch
from ns.topos.fattree import build as build_fattree
from ns.topos.utils import generate_fib, generate_flows, read_topo
from ns.utils.generators.pareto_onoff_generator import pareto_onoff_generator
from ns.utils.generators.MAP_MSP_generator import BMAP_generator

from data_gen_utils import flow_to_df, merge_flow_dfs



parser = argparse.ArgumentParser(description='Data Generation Script for DeepQueueNet using ns.py')
parser.add_argument('--port-rate', type=int, default=1024 * 1024 * 50, 
                    help='Port Rate in bits per second (default: 50 Mbps)')
parser.add_argument('--buffer-size', type=int, default=1024 * 100,
                    help='Buffer Size in bytes (default: 100 KB)')
parser.add_argument('--mean-pkt-size', type=int, default=1000,
                    help='Mean packet size in bytes (default: 1000 B)')
parser.add_argument('--scheduler', type=str, default='FIFO', choices=['DRR', 'WFQ', 'SP', 'FIFO'],
                    help='which scheduler to use. (options: DRR, WFQ, SP, FIFO  default: FIFO)')
parser.add_argument('--traffic-gen', type=str, default='Poisson', choices=['Poisson', 'OnOff'],
                    help='Which traffic generator to use for hosts. (options: Poisson, OnOff  default: Poisson)')
parser.add_argument('--num-ports', type=int, default=4,
                    help='Max number of ports in the switch (default: 4)')
parser.add_argument('--num-flows', type=int, default=100,
                    help='Number of flows to use in the simulation (default: 100)')
parser.add_argument('--duration', type=float, default=30.,
                    help='Number of seconds to run the simulation (default: 10)')
parser.add_argument('--output-dir', type=str, default='data',
                    help='Directory to write the csv results to (default: sim_data)')
parser.add_argument('--output-name', type=str, default='rsim',
                    help='Name of the output csv file (default: rsim)')
args = parser.parse_args()

# For packet size distributions
a = 50
scale = 250
primary_skew = stats.skewnorm(a, 2 * args.mean_pkt_size, scale)
secondary_skew = stats.skewnorm(a, args.mean_pkt_size, scale)

def interarrival(y):
    try:
        return next(y)
    except StopIteration:
        return

def packet_size():
    p = random.random()
    
    if p < 0.67125: #0.47125: # 67.125%
        return int(primary_skew.rvs(1))
    elif p < 0.92504:  # 25.28%
        return int(secondary_skew.rvs(1))
    elif p < 0.95905:  # 3.5%
        return 66
    elif p < 0.96875:  # 0.97%
        return 111
    elif p < 0.97485:  # 0.61%
        return 670
    else:  # remaining 2.515%
        return int(random.uniform(0, args.mean_pkt_size))


def generate_synthetic_traffic_dataset(G, all_flows):

    # Create the environment
    env = simpy.Environment()

    # Here we setup various ways of generating traffic.
    if args.traffic_gen == 'Poisson':

        packets_per_second = args.port_rate / (args.mean_pkt_size * 8) # packets per second
        utilization = 0.6 / args.num_flows # TODO: Vary the link load between 0.1 and 0.9
        lambd = packets_per_second * utilization
        iat_dist = partial(expovariate, lambd)

    elif args.traffic_gen == 'OnOff': 
            
        y = pareto_onoff_generator(on_min=0.5, # This value is from the paper
                                on_alpha=1.5,
                                off_min=0.2, # This value is from the paper
                                off_alpha=1.5,
                                on_rate=args.port_rate,
                                pktsize=args.mean_pkt_size)

        iat_dist = partial(interarrival, y)

    elif args.traffic_gen == 'MAP':
        # TODO This is complicated, because I need to setup the interarrival matrices D0 and D1.
        # This will look something like:
        # D0 = np.array([[-114.46031, 11.3081, 8.42701],
        #            [158.689, -29152.1587, 20.5697],
        #            [1.08335, 0.188837, -1.94212]])
        # D1 = np.array([[94.7252, 0.0, 0.0], [0.0, 2.89729e4, 0.0],
        #             [0.0, 0.0, 0.669933]])
        # y = BMAP_generator([D0, D1])
        # iat_dist = partial(interarrival, y)
        raise NotImplementedError("MAP generation is not implemented yet")

    # Setup the simpy components the proper way
    for fid in all_flows:
        
        pg = DistPacketGenerator(env,
                                f"Flow_{fid}",
                                iat_dist,
                                packet_size,
                                flow_id=fid)
        ps = PacketSink(env)

        all_flows[fid].pkt_gen = pg
        all_flows[fid].pkt_sink = ps

    # This ns.py utility function generates the specific forwarding table that will dictate the specific path that the flow 
    # will take to get from host to host.
    G = generate_fib(G, all_flows) # TODO: Look at this closer.

    # Assign weights to the flows here.
    if args.scheduler == 'SP':
        # Assign a flow a random priority between 1 and 3
        weights = list(np.random.randint(1, 4, len(all_flows)))
    elif args.scheduler == 'DRR' or args.scheduler == 'WFQ':
        # Assign a flow a random weight between 1 and 9
        weights = list(np.random.randint(1, 10, len(all_flows)))
    else:
        # Currently, only FIFO would use this
        weights = list(np.ones(len(all_flows)))
    
    # This will contain the dictionary of the forwarding tables for each switch.
    flow_to_port_tables = {}
    port_to_next_hop_tables = {}

    # Setup the switches in the network
    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        flow_to_port_tables[node_id] = node['flow_to_port']
        port_to_next_hop_tables[node_id] = node['port_to_nexthop']

        # Here we make a packet switch to simulate the queueing.
        if args.scheduler == 'FIFO':
            node['device'] = SimplePacketSwitch(env,
                                                args.num_ports,
                                                args.port_rate,
                                                args.buffer_size,
                                                element_id=f"{node_id}")
        else:
            # The fair packet switch implements the DRR, WFQ, and SP queueing schedulers
            node['device'] = FairPacketSwitch(env,
                                              args.num_ports,
                                              args.port_rate,
                                              args.buffer_size,
                                              weights,
                                              args.scheduler,
                                              element_id=f"{node_id}")
        node['device'].demux.fib = node['flow_to_port']

    # Map the forwarding table to specific ports where things will be forwarded to/from
    for n in ft.nodes():
        node = ft.nodes[n]
        for port_number, next_hop in node['port_to_nexthop'].items():
            node['device'].ports[port_number].out = ft.nodes[next_hop]['device']

    # Setup the packet sinks, as these will track all the stats of the packets.
    for flow_id, flow in all_flows.items():
        flow.pkt_gen.out = ft.nodes[flow.src]['device']
        ft.nodes[flow.dst]['device'].demux.ends[flow_id] = flow.pkt_sink

    # Run the simulation
    logging.info(f"Network setup complete, running the simulation for {args.duration} seconds...")
    env.run(until=args.duration)
    logging.info(f"Saving Results")
    # Now, we use our utility functions to aggregate the results into a per flow dataframe, and merge based on time
    dfs = [flow_to_df(*flow, weights, args.scheduler) for flow in all_flows.items()]
    df = merge_flow_dfs(dfs)
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Write to csv file.
    df.to_csv(os.path.join(args.output_dir, f'{args.output_name}.csv'), index=False)
    # This prints the number of non-link datapoints in our pipeline.
    print(len(df[df['cur_hub'] <= '19']))

    # Also write the forwarding table to the output 
    with open(os.path.join(args.output_dir, f'{args.output_name}.port_to_nexthop'), 'wb') as f:
        cloudpickle.dump(port_to_next_hop_tables, f)
    with open(os.path.join(args.output_dir, f'{args.output_name}.flow_to_port'), 'wb') as f:
        cloudpickle.dump(flow_to_port_tables, f)


if __name__ == "__main__":

    # Build the topology
    ft = build_fattree(args.num_ports)

    # aggregate the set of hosts in the network. This will allow us to generate random flows.
    hosts = {n for n in ft.nodes() if ft.nodes[n]['type'] == 'host'}
    all_flows = generate_flows(ft, hosts, args.num_flows)

    # Generate the synthetic traffic dataset
    generate_synthetic_traffic_dataset(ft, all_flows)