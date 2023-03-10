# JAX STUFF
from jax import Array, grad, jit, vmap, tree_util
import jax.numpy as jnp

from chex import dataclass

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
    def __init__(self,
                 length,
                 prop_speed,
                 bandwidth):

        self.length = length
        self.prop_speed = prop_speed,
        self.bandwidth = bandwidth


    def _forward_tau(self, tau: Tau):
        return Tau(
            timestamp=tau.timestamp + tau.packet.length / link.bandwidth + link.length / link.propagation_speed,
            packet=tau.packet # Don't modify the packet, since this is only a link.
        )


    def forward(self, T_in):
        '''
        Given an incoming packet stream T_in, return the outgoing packet stream T_out
        '''
        T_in = tree_stack(T_in)
        return tree_unstack(vmap(self._forward_tau, in_axes(0, None)) (T_in))

class PFM:
    def __init__(self, forwarding_table):
        self.tensor = None # TODO: set the forwarding tensor

    def forward(self, T_in):
        '''
        T_in is a list of Taus (packet stream)
        '''
        # TODO: calculate out port using forward tensor described in 3.2.2
        return T_in

@dataclass
class Device:
    def __init__(self, forwarding_table)
        self.pfm = PFM(forwarding_table)
        self.ptm = None

    def forward(self, T_in):
        '''
        TODO: modify the forward function
        '''
        return self.pfm.forward(T_in) + 0 # TODO: add delay from PTM