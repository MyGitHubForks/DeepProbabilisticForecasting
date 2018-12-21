import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from . import attention


class RecurrentDecoder(nn.Module):

    def __init__(self, rnn_type, input_dim, hidden_dim,
                 annotation_dim, num_layers, attention_type, input_feeding,
                 dropout_prob, args):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.annotation_dim = annotation_dim
        self.num_layers = num_layers
        self.input_feeding = input_feeding
        self.dropout_prob = dropout_prob
        self.args = args
        self.attention_type = attention_type
        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Linear(in_features=self.input_dim,
                                        out_features=self.input_dim)
        rnn_input_size = input_dim
        if input_feeding and attention_type != "None":
            rnn_input_size += hidden_dim
        if num_layers == 1:
            RNNDROPOUT = 0.0
        else:
            RNNDROPOUT = self.dropout_prob
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=RNNDROPOUT)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=RNNDROPOUT)
        else:
            raise ValueError('Unknown RNN type!')
        if attention_type == 'dot':
            assert hidden_dim == annotation_dim, (
                'hidden_dim and annotation_dim must be same when using'
                ' dot attention.')
            self.attention = attention.DotAttention(
                hidden_dim=hidden_dim, dropout_prob=dropout_prob)
        elif attention_type == 'mlp':
            self.attention = attention.MLPAttention(
                hidden_dim=hidden_dim, annotation_dim=annotation_dim,
                dropout_prob=dropout_prob)
        elif attention_type == "None":
            self.attention = nn.Sequential(nn.Linear(in_features=hidden_dim,
            out_features=hidden_dim),
            nn.Tanh())
        else:
            raise ValueError('Unknown attention type!')
        self.output_linear = nn.Linear(in_features=hidden_dim,
                                       out_features=input_dim)
        self.reset_parameters()
        self.resizeEncoderState = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)

    def reset_parameters(self):
        init.normal_(self.word_embedding.weight.data, mean=0, std=0.01)
        for i in range(self.num_layers):
            weight_ih = getattr(self.rnn, f'weight_ih_l{i}')
            weight_hh = getattr(self.rnn, f'weight_hh_l{i}')
            bias_ih = getattr(self.rnn, f'bias_ih_l{i}')
            bias_hh = getattr(self.rnn, f'bias_hh_l{i}')
            init.orthogonal_(weight_hh.data)
            init.kaiming_normal_(weight_ih.data)
            init.constant_(bias_ih.data, val=0)
            init.constant_(bias_hh.data, val=0)
            if self.rnn_type == 'lstm':  # Set initial forget bias to 1
                bias_ih.data.chunk(4)[1].fill_(1)
        if self.attention_type != "None":
            self.attention.reset_parameters()
        init.normal_(self.output_linear.weight.data, mean=0, std=0.01)
        init.constant_(self.output_linear.bias.data, val=0)

    def scheduleSample(self, epoch):
        eps = max(self.args.scheduling_start - 
            (self.args.scheduling_start - self.args.scheduling_end)* epoch / self.args.n_epochs,
            self.args.scheduling_end)
        return np.random.binomial(1, eps)

    def forward(self, annotations, targets, state, epoch):
        """
        Args:
            annotations (Variable): A float variable of size
                (max_src_length, batch_size, context_dim).
            decoder_inputs (Variable): A long variable of size
                (seq_length, batch_size, n_features) that either the previous
                target or prediction.
            state (DecoderState): The current state of the decoder.

        Returns:
            logits (Variable): A float variable containing unnormalized
                log probabilities. It has the same size as words.
            state (DecoderState): The updated state of the decoder.
            attention_weights (Variable): A float variable of size
                (length, batch_size, max_src_length), which contains
                the attention weight for each time step of the context.
        """
        inp = torch.zeros(targets[0].size())
        decoder_inputs = torch.cat([inp.unsqueeze(0), targets[:-1]], dim=0)
        state = copy.copy(state)
        bidirectionalEncoder = state.prepForDecoder(num_layers=self.num_layers)
        if bidirectionalEncoder:
            if isinstance(state.rnn, tuple):
                hidden = self.resizeEncoderState(state.rnn[0])
                cell = self.resizeEncoderState(state.rnn[1])
                state.update_rnn_state((hidden, cell))
            else:
                state.update_rnn_state(self.resizeEncoderState(state.rnn))
        if not self.input_feeding:
            embedded_inputs = self.word_embedding(decoder_inputs)
            embedded_inputs = self.dropout(embedded_inputs)
            rnn_outputs, rnn_state = self.rnn(input=embedded_inputs, hx=state.rnn)
            if self.attention_type != "None":
                attentional_states, attention_weights = self.attention(
                    queries=rnn_outputs, annotations=annotations)
            else:
                attentional_states = rnn_outputs
                attention_weights = torch.zeros((1,1))
                if self.args.cuda:
                    attention_weights = attention_weights.cuda()
            state.update(rnn_state=rnn_state)
        else:
            target_seq_len, batch_size, _ = decoder_inputs.size()
            attentional_states = []
            attention_weights = []
            if state.attention is None:
                zero_attentional_state = torch.zeros((batch_size, self.hidden_dim))
                # zero_attentional_state = (
                #     embedded_inputs.data.new(batch_size, self.hidden_dim).zero_())
                zero_attentional_state = Variable(zero_attentional_state)
                state.update_attentional_state(zero_attentional_state)
            if self.args.no_schedule_sampling or not self.training:
                sample=0
            else:
                sample = self.scheduleSample(epoch)
            for t in range(target_seq_len):
                inp = self.word_embedding(inp)
                inp = self.dropout(inp)
                if self.attention_type != "None":
                    decoder_input_t = torch.cat([inp, state.attention], dim=1)
                    decoder_input_t = decoder_input_t.unsqueeze(0)
                    rnn_output_t, rnn_state_t = self.rnn(
                        input=decoder_input_t, hx=state.rnn)
                    attentional_state_t, attention_weights_t = self.attention(
                        queries=rnn_output_t, annotations=annotations)
                else:
                    attentional_state_t, rnn_state_t = self.rnn(
                        input=inp.unsqueeze(0), hx=state.rnn)
                    attention_weights_t = torch.zeros((1,1))
                    if self.args.cuda:
                        attention_weights_t = attention_weights_t.cuda()
                attentional_state_t = self.output_linear(attentional_state_t.squeeze(0))
                attentional_states.append(attentional_state_t)
                attention_weights.append(attention_weights_t)
                state.update_state(rnn_state=rnn_state_t,
                                   attentional_state=attentional_state_t)
                if sample:
                    inp = targets[t].squeeze()
                else:
                    inp = attentional_state_t.squeeze()
            attentional_states = torch.stack(attentional_states, dim=0)
            attention_weights = torch.cat(attention_weights, dim=0)
        logits = attentional_states
        return logits, state, attention_weights


class DecoderState(dict):

    def __init__(self, rnn_state, attentional_state=None,
                 input_feeding=False, decoder_layers=None):
        super().__init__()
        self['input_feeding'] = input_feeding
        self['rnn'] = rnn_state
        assert input_feeding or attentional_state is None
        self['attention'] = None
        if input_feeding:
            self['attention'] = attentional_state

    @staticmethod
    def apply_to_rnn_state(fn, rnn_state):
        if isinstance(rnn_state, tuple):  # LSTM
            return tuple(fn(s) for s in rnn_state)
        else:
            return fn(rnn_state)

    @property
    def input_feeding(self):
        return self['input_feeding']

    @property
    def rnn(self):
        return self['rnn']

    @property
    def attention(self):
        return self['attention']

    def prepForDecoder(self, num_layers):
        if isinstance(self["rnn"], tuple):
            # LSTM
            newState = []
            for i in range(2):
                numLayersTDirections, batch_size, hidden_dim = self["rnn"][i].size()
                separateDirections = self["rnn"][i].view(num_layers, -1, batch_size, hidden_dim)
                if separateDirections.size(1) == 2:
                    forwardHidden = separateDirections[:,0,:,:]
                    reverseHidden = separateDirections[:,1,:,:]
                    hidden = torch.cat([forwardHidden, reverseHidden], dim=2)
                else:
                    hidden = separateDirections[:,0,:,:]
                newState.append(hidden)
            self["rnn"] = (newState[0], newState[1])
        else:
            # GRU
            numLayersTDirections, batch_size, hidden_dim = self["rnn"].size()
            separateDirections = self["rnn"].view(num_layers, -1, batch_size, hidden_dim)
            if separateDirections.size(1) == 2:
                forwardHidden = separateDirections[:,0,:,:]
                reverseHidden = separateDirections[:,1,:,:]
                hidden = torch.cat([forwardHidden, reverseHidden], dim=2)
            else:
                hidden = separateDirections[:,0,:,:]
            self["rnn"] = hidden
        return separateDirections.size(1) == 2



    def update_state(self, rnn_state=None, attentional_state=None):
        self.update_rnn_state(rnn_state)
        self.update_attentional_state(attentional_state)

    def update_rnn_state(self, rnn_state):
        self['rnn'] = rnn_state

    def update_attentional_state(self, attentional_state):
        if self.input_feeding and attentional_state is not None:
            self['attention'] = attentional_state

    def repeat(self, beam_size):
        new_state = copy.copy(self)
        new_state['rnn'] = self.apply_to_rnn_state(
            fn=lambda s: s.repeat(1, beam_size, 1),
            rnn_state=new_state['rnn'])
        if self.input_feeding and new_state['attention'] is not None:
            new_state['attention'] = (
                new_state['attention'].repeat(beam_size, 1))
        return new_state

    def beam_update(self, batch_index, beam_indices, beam_size):
        def beam_update_fn(v):
            # The shape is (..., beam_size * batch_size, state_dim).
            orig_size = v.size()
            new_size = (
                orig_size[:-2]
                + (beam_size, orig_size[-2] // beam_size, orig_size[-1]))
            # beam_of_batch: (..., beam_size, state_dim)
            beam_of_batch = v.view(*new_size).select(-2, batch_index)
            beam_of_batch.data.copy_(
                beam_of_batch.data.index_select(-2, beam_indices))

        self.apply_to_rnn_state(fn=beam_update_fn, rnn_state=self['rnn'])
        if self.input_feeding and self['attention'] is not None:
            beam_update_fn(self['attention'])
