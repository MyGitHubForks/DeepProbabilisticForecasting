import torch
from torch import nn

from . import encoders, decoders


class RecurrentSeq2Seq(nn.Module):

    def __init__(self, hidden_dim, dropout_prob,
                rnn_type, bidirectional, num_layers,
                attention_type, input_feeding,
                input_dim, output_dim, args):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.attention_type = dropout_prob
        self.input_feeding = input_feeding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = encoders.RecurrentEncoder(
            rnn_type=rnn_type, input_dim=input_dim,
            hidden_dim=hidden_dim, dropout_prob=dropout_prob,
            bidirectional=bidirectional, num_layers=num_layers)
        annotation_dim = hidden_dim
        if bidirectional:
            annotation_dim = hidden_dim * 2
        self.decoder = decoders.RecurrentDecoder(
            rnn_type=rnn_type, input_dim=output_dim, hidden_dim=hidden_dim,
            annotation_dim=annotation_dim, num_layers=num_layers,
            attention_type=attention_type, input_feeding=input_feeding,
            dropout_prob=dropout_prob, args=args)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def remove4D(self, encoder_inputs):
        inp0, inp1 = encoder_inputs[:,:,0,:], encoder_inputs[:,:,1,:]
        return torch.cat([inp0, inp1], dim=2)


    def forward(self, encoder_inputs, decoder_inputs, epoch):
        encoder_hidden_states, encoder_state = self.encoder(
            encoder_inputs=self.remove4D(encoder_inputs))
        decoder_state = decoders.DecoderState(
            rnn_state=encoder_state, input_feeding=self.input_feeding)
        logits, _, _ = self.decoder(
            annotations=encoder_hidden_states,
            targets=decoder_inputs, state=decoder_state, epoch=epoch)
        return logits
