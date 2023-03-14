import torch
import torch.nn as nn


lstm_params={'layer':2,   'cell_neurons':[200,100],     'keep_prob':1}  
att=64. #attention output layer dim
mul_head=3
mul_head_output_nodes=32

