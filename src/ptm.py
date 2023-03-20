import torch
import torch.nn as nn


att=64. #attention output layer dim
mul_head=3
mul_head_output_nodes=32

fet_cols = ['pkt len (byte)', 'src', 'dst', 'TI0', 'TI1', 'TI2', 'TI3', 'load_dst0_0', 'load_dst1_0', 'load_dst2_0', 'load_dst3_0', 'inter_arr_sys']

batch_size = 4
time_steps = 42

in_feat = len(fet_cols)

class deepPTM(nn.Module):
    def __init__(self, in_feat=in_feat,
                 lstm_config= {"depth": 2, "width":[200,100], "keep_prob": 1, 'bidirectional': True, 'dropout': 0.},
                 attn_config={"dim": 64, "num_heads": 3, "out_dim": 32, 'dropout': 0.},
                 time_steps=42, *args, **kwargs) -> None:
        """
        """
        super().__init__(*args, **kwargs)

        # Feature vector Copied from the extracted features


        
        # TODO: Add initialization

        # Input LSTM
        self.in_lstm = nn.ModuleList(
            [nn.LSTM(input_size=in_feat, hidden_size=lstm_config["width"][0], num_layers=lstm_config["depth"], batch_first=True, 
                    dropout=lstm_config["dropout"], bidirectional=lstm_config["bidirectional"], )] +
            [nn.LSTM(input_size=lstm_config["width"][i], hidden_size=width, num_layers=lstm_config["depth"], batch_first=True,
                    dropout=lstm_config["dropout"], bidirectional=lstm_config["bidirectional"], ) 
                    for i, width in enumerate(lstm_config["width"][1:])]
                    )
        
        # KQV encoder
        self.val_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) for i in range(attn_config["num_heads"])]
        self.key_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) for i in range(attn_config["num_heads"])]
        self.query_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) for i in range(attn_config["num_heads"])]


        # Multihead attention
        self.attn = nn.MultiheadAttention(embed_dim=attn_config["dim"] * attn_config["num_heads"],
                                          num_heads=attn_config["num_heads"],
                                          dropout=attn_config["dropout"],
                                          batch_first=True)
        self.attn_out = torch.nn.Linear(attn_config["dim"] * attn_config["num_heads"],
                                        attn_config["out_dim"])


        
        # LSTM out
        self.out_lstm = nn.ModuleList(
            [nn.LSTM(input_size= attn_config["out_dim"], hidden_size=lstm_config["width"][0], num_layers=lstm_config["depth"], batch_first=True, 
                    dropout=lstm_config["dropout"], bidirectional=lstm_config["bidirectional"], )] +
            [nn.LSTM(input_size=lstm_config["width"][i], hidden_size=width, num_layers=lstm_config["depth"], batch_first=True,
                    dropout=lstm_config["dropout"], bidirectional=lstm_config["bidirectional"], ) 
                    for i, width in enumerate(lstm_config["width"][1:])]
                    )
        
        self.dec_layer = nn.Linear(lstm_config["width"][-1], 1)

        # Save configs
        self.lstm_config = lstm_config
        self.attn_config = attn_config
        attn_embbed_dim = attn_config["dim"]
        num_attn_heads = attn_config["num_heads"]
        self.time_steps = time_steps




    def forward(self, packet_batch):
        b = len(packet_batch)
        # TODO: Check dimensionalit and in and output codes
        x = self.in_lstm(packet_batch)

        v = torch.zeros(b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim)
        k = torch.zeros(b, self.time_steps,  self.num_attn_heads, self.attn_embbed_dim)
        q = torch.zeros(b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim)
        for i, (val_enc, key_enc, query_enc) in enumerate(zip(self.val_enc, self.key_enc, self.query_enc)):
            v[:,:, i*self.attn_embbed_dim: (i+1)*self.attn_embbed_dim] = val_enc(x)
            k[:,:, i*self.attn_embbed_dim: (i+1)*self.attn_embbed_dim] = key_enc(x)
            q[:,:, i*self.attn_embbed_dim: (i+1)*self.attn_embbed_dim] = query_enc(x)

        v = v.reshape(b, self.time_steps, -1)
        k =k.reshape(b, self.time_steps, -1)
        q = q.reshape(b, self.time_steps, -1)

        att = self.attn(q, k, v)
        x = self.attn_out(att)

        lstm_out = self.out_lstm(x)
        
        t_pred = self.dec_layer(lstm_out)

        return t_pred
    



placeholder = torch.randn(batch_size, time_steps, in_feat)

model = deepPTM()

t_out = model(placeholder)
