# JAX STUFF
from jax import Array, grad, jit, vmap, tree_util
import jax.numpy as jnp

from chex import dataclass

from utils import tree_stack, tree_unstack, tree_index
@dataclass
class Packet:
    pid: int
    fid: int
    length: int
    trp: int
    in_port: int # Extra feature added in pre PFM augmentation

@dataclass
class Tau:
    timestamp: float # timestamp are floats in seconds (same as PCAP)
    packet: Packet

@dataclass
class Link:
    length: float
    propagation_speed: float
    bandwidth: float

    def _forward_tau(self, tau: Tau):
        return Tau(
            timestamp=tau.timestamp + tau.packet.length / self.bandwidth + self.length / self.propagation_speed,
            packet=tau.packet # Don't modify the packet, since this is only a link.
        )


    def forward(self, T_in):
        '''
        Given an incoming packet stream T_in, return the outgoing packet stream T_out

        Arg(s):
            T_in : list[CustomNode(Tau)]
                list of Taus

        Returns:
            T_out : list[CustomNode(Tau)]
                list of Taus with timestamps updated with link delay
        '''
        T_in = tree_stack(T_in)
        return tree_unstack(vmap(self._forward_tau, in_axes=(0, None)) (T_in))

'''
Module for packet forwarding in devices
'''
class PFM:
    def __init__(self, forwarding_table):
        self.tensor = None # TODO: set the forwarding tensor

    def forward(self, T_in):
        '''
        T_in is a list of Taus (packet stream)
        '''
        # TODO: calculate out port using forward tensor described in 3.2.2
        return T_in

'''
Module for traffic management in devices
'''
class PTM:
    def __init__(self, scheduler_type: str, model_restore_path):
        self.scheduler_type = scheduler_type # Still unsure where this is from
        self.priority_table = None # Input to config
        self.weight_table = None # Input to config
        self.model = None # TODO: Create model and load in model weights
        pass

    def forward(self, T_in):
        delays = [0 for i in range(len(T_in))]
        # delays = self.model(T_in)
        return delays

'''
Device class
'''
class Device:
    def __init__(self, forwarding_table):
        self.pfm = PFM(forwarding_table)
        self.ptm = PTM()

    def forward(self, T_in):
        '''
        TODO: modify the forward function
        '''
        return self.pfm.forward(T_in) + self.ptm.forward(T_in) # TODO: add delay from PTM