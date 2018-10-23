class EncoderRNN(nn.Module):
    def __init__(self, sequence_length, num_features, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru = nn.GRU(num_features, hidden_size, n_layers, dropout=self.dropout).cuda()

        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        sequence_lengths = torch.IntTensor([self.sequence_length for i in range(input_seqs.size(1))], device=self._device).cuda()
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, sequence_lengths).cuda()
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden
