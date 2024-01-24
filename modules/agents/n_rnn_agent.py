# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
# import numpy as np
# import torch.nn.init as init
# from utils.th_utils import orthogonal_init_
# from torch.nn import LayerNorm
# from modules.layer.self_atten import SelfAttention


# class NRNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(NRNNAgent, self).__init__()
#         self.args = args
        
#         # attention state net
#         # State: all_q_d_fea + all_last_actions + time_step
#         # Obs: all_q_d_fea(att) + query_doc_fea + time_step + last_action_i
#         self.att = SelfAttention(768*10, args.att_heads, args.att_embed_dim)

#         self.fc1 = nn.Linear(input_shape - 768*10, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim + args.att_heads*args.att_embed_dim, args.att_heads*args.att_embed_dim)  # args.n_actions

#         self.att2 = SelfAttention(args.att_heads*args.att_embed_dim, args.att_heads, args.att_embed_dim)
#         self.fc3 = nn.Linear(args.att_heads*args.att_embed_dim, args.n_actions)

#         if getattr(args, "use_layer_norm", False):
#             self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
#         if getattr(args, "use_orthogonal", False):
#             orthogonal_init_(self.fc1)
#             orthogonal_init_(self.fc2, gain=args.gain)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         b, a, e = inputs.size()

#         inputs = inputs.view(-1, e)

#         # attention layer 1
#         attention_inputs = inputs[:, :768*10]
#         att = self.att(attention_inputs.unsqueeze(1)) # 输入维度batch_size, t, input_dim
#         att = att.squeeze() # 去除多余维度

#         inputs = inputs[:, 768*10: ]

#         x = F.relu(self.fc1(inputs), inplace=True)
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         hh = self.rnn(x, h_in)
        
#         att_hh = th.cat([att, hh], -1)

#         att_hh = F.relu(self.fc2(att_hh))

#         att_hh = self.att2(att_hh.unsqueeze(1))
#         att_hh = att_hh.squeeze()

#         if getattr(self.args, "use_layer_norm", False):
#             q = self.fc3(self.layer_norm(att_hh))
#         else:
#             q = self.fc3(att_hh)

#         return q.view(b, a, -1), hh.view(b, a, -1)
    

# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
# import numpy as np
# import torch.nn.init as init
# from utils.th_utils import orthogonal_init_
# from torch.nn import LayerNorm
# from modules.layer.self_atten import SelfAttention

# class NRNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(NRNNAgent, self).__init__()
#         self.args = args

#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

#         if getattr(args, "use_layer_norm", False):
#             self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
#         if getattr(args, "use_orthogonal", False):
#             orthogonal_init_(self.fc1)
#             orthogonal_init_(self.fc2, gain=args.gain)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         b, a, e = inputs.size()

#         inputs = inputs.view(-1, e)
#         x = F.relu(self.fc1(inputs))
#         # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         # hh = self.rnn(x, h_in)

#         if getattr(self.args, "use_layer_norm", False):
#             q = self.fc2(self.layer_norm(x))
#         else:
#             q = self.fc2(x)

#         return q.view(b, a, -1), hidden_state.view(b, a, -1)


# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
# import numpy as np
# import torch.nn.init as init
# from utils.th_utils import orthogonal_init_
# from torch.nn import LayerNorm
# from modules.layer.self_atten import SelfAttention

# class NRNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(NRNNAgent, self).__init__()
#         self.args = args

#         self.att = SelfAttention(input_shape, args.att_heads, args.rnn_hidden_dim)  # args.att_embed_dim

#         self.fc1 = nn.Linear(args.att_heads * args.rnn_hidden_dim, args.rnn_hidden_dim)
#         # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

#         if getattr(args, "use_layer_norm", False):
#             self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
#         if getattr(args, "use_orthogonal", False):
#             orthogonal_init_(self.fc1)
#             orthogonal_init_(self.fc2, gain=args.gain)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         b, a, e = inputs.size()

#         # inputs = inputs.view(-1, e)
#         inputs = inputs.view(-1, 1, e)
#         inputs = self.att(inputs)
#         inputs = inputs.squeeze() # 去除多余维度

#         x = F.relu(self.fc1(inputs))
#         # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         # hh = self.rnn(x, h_in)

#         if getattr(self.args, "use_layer_norm", False):
#             q = self.fc2(self.layer_norm(x))
#         else:
#             q = self.fc2(x)

#         return q.view(b, a, -1), hidden_state.view(b, a, -1)


# 12.23最后使用，带atten版本。
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


# # 12.23开始使用，带atten版本。使用的是WWW 2021的方法，将attention网络输出的ai作为输入。
# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
# import numpy as np
# import torch.nn.init as init
# from utils.th_utils import orthogonal_init_
# from torch.nn import LayerNorm
# from modules.layer.self_atten import SelfAttention

# class NRNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(NRNNAgent, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.dims = 1024

#         self.att = SelfAttention(input_shape-self.dims*(self.n_agents+1)-self.n_agents, args.att_heads, args.rnn_hidden_dim)  # args.att_embed_dim

#         self.fc1 = nn.Linear(args.att_heads * args.rnn_hidden_dim, args.rnn_hidden_dim)
#         # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim + args.rnn_hidden_dim, args.n_actions)

#         self.att_state = SelfAttention(self.dims, args.att_heads, args.att_embed_dim)
#         self.fc3 = nn.Linear(args.att_heads * args.att_embed_dim, args.rnn_hidden_dim)

#         if getattr(args, "use_layer_norm", False):
#             self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
#         if getattr(args, "use_orthogonal", False):
#             orthogonal_init_(self.fc1)
#             orthogonal_init_(self.fc2, gain=args.gain)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         b, a, e = inputs.size()

#         # inputs = inputs.view(-1, e)
#         inputs = inputs.view(-1, 1, e)

#         state_inputs = inputs[:, :, : self.dims*(self.n_agents+1)]
#         onehot_agent_id = inputs[: , :, -self.n_agents: ]
#         agent_id = th.argmax(onehot_agent_id, dim=-1).unsqueeze(-1)+1
#         inputs = inputs[:, :, self.dims*(self.n_agents+1): -self.n_agents]

#         inputs = self.att(inputs)
#         inputs = inputs.squeeze() # 去除多余维度

#         x = F.relu(self.fc1(inputs))
#         # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         # hh = self.rnn(x, h_in)

#         state_inputs = self.att_state(state_inputs.reshape(-1, self.n_agents+1, self.dims))
#         att_x = self.att_state(state_inputs)
#         att_x = att_x.reshape(-1, self.n_agents+1, self.args.att_heads*self.args.att_embed_dim)
#         # agent_id onehot
#         # att_x = att_x 的 agent_id onehot
#         att_x = att_x.squeeze() # 去除多余维度

#         att_x = F.relu(self.fc3(att_x))

#         final_x = th.cat([att_x, x], -1)

#         if getattr(self.args, "use_layer_norm", False):
#             q = self.fc2(self.layer_norm(final_x))
#         else:
#             q = self.fc2(final_x)

#         return q.view(b, a, -1), hidden_state.view(b, a, -1)