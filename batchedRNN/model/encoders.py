from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentEncoder(nn.Module):

    def __init__(self, rnn_type, input_dim, hidden_dim,
                 dropout_prob, bidirectional, num_layers):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_prob)
        self.embedding_layer = nn.Linear(in_features=self.input_dim,
                                           out_features=self.input_dim)
        if num_layers == 1:
            RNNDROPOUT = 0.0
        else:
            RNNDROPOUT = dropout_prob
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers,
                dropout=RNNDROPOUT)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers,
                dropout=RNNDROPOUT)
        else:
            raise ValueError('Unknown RNN type!')
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.embedding_layer.weight.data, mean=0, std=0.01)
        for i in range(self.num_layers):
            suffixes = ['']
            if self.bidirectional:
                suffixes.append('_reverse')
            for suffix in suffixes:
                weight_ih = getattr(self.rnn, f'weight_ih_l{i}{suffix}')
                weight_hh = getattr(self.rnn, f'weight_hh_l{i}{suffix}')
                bias_ih = getattr(self.rnn, f'bias_ih_l{i}{suffix}')
                bias_hh = getattr(self.rnn, f'bias_hh_l{i}{suffix}')
                init.orthogonal_(weight_hh.data)
                init.kaiming_normal_(weight_ih.data)
                init.constant_(bias_ih.data, val=0)
                init.constant_(bias_hh.data, val=0)
                if self.rnn_type == 'lstm':  # Set initial forget bias to 1
                    bias_ih.data.chunk(4)[1].fill_(1)

    def forward(self, encoder_inputs):
        encoder_inputs_emb = self.embedding_layer(encoder_inputs)
        encoder_inputs_emb = self.dropout(encoder_inputs_emb)
        encoder_hidden_states, rnn_state = self.rnn(encoder_inputs_emb)
        # For LSTM, encoder_states does not contain cell states.
        # Thus it is necessary to explicitly return the last state
        encoder_state = rnn_state
        return encoder_hidden_states, encoder_state
