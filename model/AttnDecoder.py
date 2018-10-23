import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Attn as AF

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, method=None):
        super(AttnDecoderRNN, self).__init__()

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
        if method != None:
            self.attn = AF.Attn(method, hidden_size).cuda()

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time S = 1
        batch_size = input_seq.size(0)
        expandedInput = input_seq.unsqueeze(0)# S=1 x B X output size
        
        # Get current hidden state from decoder input and last hidden state
        rnn_output, hidden = self.gru(expandedInput.cuda(), last_hidden.cuda())
        
        #Calculate attention from current RNN state and all encoder outputs
        #apply  to encoder outputs to get weighted attention
        attn_weights = self.attn(rnn_output, encoder_outputs).cuda()
        context = attn_weights.bmm(encoder_outputs.transpose(0,1)).cuda() #B x S x NHidden
        
        rnn_output = rnn_output.squeeze(0) # S=1 x B x NHidden -> B x NHidden
        context = context.squeeze(1) # B x S=1 x NHidden -> B x NHidden
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        output = self.out(concat_output)
        
        # Return final output, hidden state
        return output, hidden, attn_weights
