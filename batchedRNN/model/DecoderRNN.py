class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define layers
        self.gru = nn.GRU(self.output_size, hidden_size, n_layers, dropout=dropout).cuda()
        self.concat = nn.Linear(hidden_size * 2, hidden_size).cuda()
        self.out = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, input_seq, last_hidden):
        # Note: we run this one step at a time S = 1
        batch_size = input_seq.size(0)
        expandedInput = input_seq.unsqueeze(0)# S=1 x B X output size
        
        # Get current hidden state from decoder input and last hidden state
        rnn_output, hidden = self.gru(expandedInput.cuda(), last_hidden.cuda())
        
        output = self.out(rnn_output).cuda()
        
        # Return final output, hidden state
        return output, hidden