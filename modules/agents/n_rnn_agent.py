import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from modules.layer.self_atten import SelfAttention

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.dims = 1024

        self.att = SelfAttention(input_shape-self.dims*(self.n_agents+1), args.att_heads, args.rnn_hidden_dim)  # args.att_embed_dim

        self.fc1 = nn.Linear(args.att_heads * args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.rnn_hidden_dim, args.n_actions)

        self.att_state = SelfAttention(self.dims, args.att_heads, args.att_embed_dim)
        self.fc3 = nn.Linear(args.att_heads * args.att_embed_dim * (self.n_agents+1), args.rnn_hidden_dim)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        # inputs = inputs.view(-1, e)
        inputs = inputs.view(-1, 1, e)

        state_inputs = inputs[:, :, : self.dims*(self.n_agents+1)]
        inputs = inputs[:, :, self.dims*(self.n_agents+1): ]

        inputs = self.att(inputs)
        inputs = inputs.squeeze() # 去除多余维度

        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # hh = self.rnn(x, h_in)

        state_inputs = self.att_state(state_inputs.reshape(-1, self.n_agents+1, self.dims))
        att_x = self.att_state(state_inputs)
        att_x = att_x.reshape(-1, (self.n_agents+1)*self.args.att_heads*self.args.att_embed_dim)
        att_x = att_x.squeeze() # 去除多余维度

        att_x = F.relu(self.fc3(att_x))

        final_x = th.cat([att_x, x], -1)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(final_x))
        else:
            q = self.fc2(final_x)

        return q.view(b, a, -1), hidden_state.view(b, a, -1)
            
