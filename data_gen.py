from functools import partial
from random import expovariate, sample

import os
import argparse
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

from utils import flow_to_df, merge_flow_dfs

parser = argparse.ArgumentParser(description='Data Generation Script for DeepQueueNet using ns.py')

parser.add_argument('--port-rate', type=int, default=10000, 
                    help='Port Rate in bytes/second (default: 10000)')
parser.add_argument('--buffer-size', type=int, default=1000,
                    help='Buffer Size in bytes (default: 1000)')
parser.add_argument('--num-ports', type=int, default=4,
                    help='Max number of ports in the switch (default: 4)')
parser.add_argument('--num-flows', type=int, default=5,
                    help='Number of flows to use in the simulation (default: 5)')
parser.add_argument('--duration', type=float, default=1000.,
                    help='Number of seconds to run the simulation (default: 1000)')
parser.add_argument('--output-dir', type=str, default='sim_data',
                    help='Directory to write the csv results to (default: sim_data)')
parser.add_argument('--output-name', type=str, default='rsim.csv',
                    help='Name of the output csv file (default: rsim.csv)')

args = parser.parse_args()

if __name__ == "__main__":

    # Create the environment
    env = simpy.Environment()

    # Build the topology TODO: Add options on this for different topologies
    ft = build_fattree(args.num_ports)

    # aggregate the set of hosts in the network. This will allow us to generate random flows.
    hosts = set()
    for n in ft.nodes():
        if ft.nodes[n]['type'] == 'host':
            hosts.add(n)

    # Generate the flows by randomly choosing two hosts on the network
    n_flows = args.num_flows
    all_flows = generate_flows(ft, hosts, n_flows)

    # TODO: parameterize this better.
    mean_pkt_size = 100
    # Generate a distribution of packet sizes.
    size_dist = partial(expovariate, 1.0 / mean_pkt_size)
    # Use the poisson process to specify packet generation. 
    arr_dist = partial(expovariate, 1 + np.random.rand())

    # Setup the simpy components the proper way
    for fid in all_flows:
        
        pg = DistPacketGenerator(env,
                                f"Flow_{fid}",
                                arr_dist,
                                size_dist,
                                flow_id=fid)
        ps = PacketSink(env)

        all_flows[fid].pkt_gen = pg
        all_flows[fid].pkt_sink = ps


    # This ns.py utility function generates the specific forwarding table that will dictate the specific path that the flow 
    # will take to get from host to host.
    ft = generate_fib(ft, all_flows) # TODO: Look at this closer.

    # Generate priorities/weights for the classes. TODO: Parameterize this somehow.
    n_classes_per_port = 4
    weights = {c: 1 for c in range(n_classes_per_port)}

    def flow_to_classes(f_id, n_id=0, fib=None):
        return (f_id + n_id + fib[f_id]) % n_classes_per_port
    
    # Setup the switches in the network
    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        flow_classes = partial(flow_to_classes,
                            n_id=node_id,
                            fib=node['flow_to_port'])
        
        # Here we make a packet switch to simulate the queueing.
        node['device'] = FairPacketSwitch(env,
                                        args.num_ports,
                                        args.port_rate,
                                        args.buffer_size,
                                        weights,
                                        'DRR', # This can be DRR, WFQ, or SP; the DeepQueueNet paper does experiments with all 3.
                                        flow_classes,
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
    env.run(until=args.duration)

    # Now, we use our utility functions to aggregate the results into a per flow dataframe, and merge based on time
    dfs = [flow_to_df(*flow) for flow in all_flows.items()]
    df = merge_flow_dfs(dfs)
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Write to csv file.
    df.to_csv(os.path.join(args.output_dir, args.output_name), index=False)




