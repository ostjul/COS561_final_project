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
                 lstm_config= {"width":[200,100], "keep_prob": 1, 'bidirectional': True, 'dropout': 0.},
                 attn_config={"dim": 64, "num_heads": 3, "out_dim": 32, 'dropout': 0.},
                 time_steps=42, *args, **kwargs) -> None:
        """
        """
        super().__init__(*args, **kwargs)

        # TODO: Add initialization

        # Input LSTM
        self.in_lstm_fw = self.in_lstm_bw = self._get_multi_layer_lstm(
                in_feat=in_feat, hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
            
        # Use type 1 biderictonal lstm where both paths are independent across all alyers
        if lstm_config["bidirectional"]:
            self.in_lstm_bw = self._get_multi_layer_lstm(
                in_feat=in_feat, hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
        
        # KQV encoder
        self.val_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) 
                        for i in range(attn_config["num_heads"])]
        self.key_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) 
                        for i in range(attn_config["num_heads"])]
        self.query_enc = [torch.nn.Linear(lstm_config["width"][-1], attn_config["dim"]) 
                          for i in range(attn_config["num_heads"])]

        # Multihead attention
        self.attn = nn.MultiheadAttention(embed_dim=attn_config["dim"] * attn_config["num_heads"],
                                          num_heads=attn_config["num_heads"],
                                          dropout=attn_config["dropout"],
                                          batch_first=True)
        self.attn_out = torch.nn.Linear(attn_config["dim"] * attn_config["num_heads"],
                                        attn_config["out_dim"])
        
        # LSTM out
        self.out_lstm_fw = self._get_multi_layer_lstm(
                in_feat=attn_config["out_dim"], hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
            
        # Use type 1 biderictonal lstm where both paths are independent across all alyers
        if lstm_config["bidirectional"]: 
            self.out_lstm_bw = self._get_multi_layer_lstm(
                in_feat=attn_config["out_dim"], hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
        
        self.dec_layer = nn.Linear(lstm_config["width"][-1], 1)

        # Save configs
        self.lstm_config = lstm_config
        self.lstm_bidirectional = lstm_config["bidirectional"]
        self.attn_config = attn_config
        self.attn_embbed_dim = attn_config["dim"]
        self.num_attn_heads = attn_config["num_heads"]
        self.time_steps = time_steps

    def _get_multi_layer_lstm(self, in_feat, hidden_size=[200, 100], dropout=0.):
        lstm = nn.ModuleList(
            [nn.LSTM(input_size=in_feat, hidden_size=hidden_size[0],
                     num_layers=1, batch_first=True, dropout=dropout)] +
                     [nn.LSTM(input_size=hidden_size[i], hidden_size=width, 
                        num_layers=1, batch_first=True, dropout=dropout) 
                        for i, width in enumerate(hidden_size[1:])]
                        )
        return lstm


    # run type 1 biderictonal LSTM
    def _run_multi_layer_bi_directional(self, x, lstm_fw, lstm_bw):
        x_f = lstm_fw[0](x)[0]
        x_b = lstm_bw[0](x.flip(-1))[0]
        for l_f, l_b in zip(lstm_fw[1:], lstm_bw[1:]):
            x_f = l_f(x_f)[0]
            x_b = l_b(x_b)[0]

        x = x_f + x_b
        return x


    def forward(self, packet_batch):
        b = len(packet_batch)
        # TODO: Check dimensionalit and in and output codes
        x = packet_batch

        if self.lstm_bidirectional:
            x = self._run_multi_layer_bi_directional(x, self.in_lstm_fw, self.in_lstm_bw)
        else:
            for l in self.in_lstm_fw:
                x = l(x)[0]

        v = torch.zeros(b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim)
        k = torch.zeros(b, self.time_steps,  self.num_attn_heads, self.attn_embbed_dim)
        q = torch.zeros(b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim)

        for i, (val_enc, key_enc, query_enc) in enumerate(zip(self.val_enc, self.key_enc, self.query_enc)):
            v[:,:, i] = val_enc(x)
            k[:,:, i] = key_enc(x)
            q[:,:, i] = query_enc(x)

        v = v.reshape(b, self.time_steps, -1)
        k =k.reshape(b, self.time_steps, -1)
        q = q.reshape(b, self.time_steps, -1)

        att = self.attn(q, k, v)[0]
        x = self.attn_out(att)

        if self.lstm_bidirectional:
            x = self._run_multi_layer_bi_directional(x, self.out_lstm_fw, self.out_lstm_bw)
        else:
            for l in self.out_lstm_fw:
                x = l(x)[0]

        lstm_out = x
        
        t_pred = self.dec_layer(lstm_out)

        return t_pred
    



placeholder = torch.randn(batch_size, time_steps, in_feat)

model = deepPTM()

t_out = model(placeholder)
