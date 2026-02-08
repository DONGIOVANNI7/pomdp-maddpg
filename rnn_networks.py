import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentACModel(nn.Module):
    """ Architecture 1: Input -> RNN -> Heads """
    def __init__(self, input_dim, action_dim, hidden_dim=64, rnn_type='LSTM', activation='relu'):
        super().__init__()
        self.activation = F.relu if activation == 'relu' else F.tanh
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        self.rnn.flatten_parameters() 
        rnn_out, new_hidden = self.rnn(x, hidden)
        rnn_out = self.activation(rnn_out)
        
        return self.actor(rnn_out), self.critic(rnn_out), new_hidden

class FeatureExtractorACModel(nn.Module):
    """ Architecture 2: Input -> Dense -> RNN -> Heads """
    def __init__(self, input_dim, action_dim, hidden_dim=64, rnn_type='LSTM', activation='relu'):
        super().__init__()
        self.activation_fn = F.relu if activation == 'relu' else F.tanh
        
        self.feature_fc = nn.Linear(input_dim, 64)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(64, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(64, hidden_dim, batch_first=True)
            
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        x_flat = x.view(-1, x.size(-1))
        feats = self.activation_fn(self.feature_fc(x_flat))
        feats = feats.view(batch_size, seq_len, -1)
        
        self.rnn.flatten_parameters()
        rnn_out, new_hidden = self.rnn(feats, hidden)
        
        return self.actor(rnn_out), self.critic(rnn_out), new_hidden