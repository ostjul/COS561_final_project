import torch
import torch.nn as nn


lstm_params={'layer':2,   'cell_neurons':[200,100],     'keep_prob':1}  
att=64. #attention output layer dim
mul_head=3
mul_head_output_nodes=32

class deepPTM(nn.Module):
    def __init__(self, in_feat, lstm_config= {"depth": 2, "width":[200,100], "keep_prob": 1}, attn={"dim": 64, "num_heads": 3, "out_dim": 32}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_lstm = nn.LSTM(
            input_size=in_feat,
            hidden_size=lstm_config["width"],
            num_layers=lstm_config["depth"],
            bidirectional=True,
            )
        
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(
            embed_dim=attn["dim"] * attn["num_heads"],
            num_heads=attn["num_heads"],),
            nn.Linear(attn["dim"], attn["out_dim"])])

        
        self.out_lstm = nn.LSTM(
            input_size=in_feat,
            hidden_size=lstm_config["width"],
            num_layers=lstm_config["depth"],
            bidirectional=True,
            )
        self.dec_layer = nn.Linear(attn["out_dim"], 1)




    def forward(self, pakcet_batch):
        enc = self.in_lstm(pakcet_batch)
        attn_out = self.attn(enc)
        lstm2_out = self.out_lstm(attn_out)
        t_pred = self.dec_layer(lstm2_out)

        return t_pred
