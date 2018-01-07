import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    """

    def __init__(self, vocab_size, max_len, dim_hidden,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        super().__init__()

        self.bidirectional_encoder = bidirectional
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True, dropout=dropout_p)

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.max_length = max_len
        self.use_attention = use_attention

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, self.dim_hidden)
        if use_attention:
            self.attention = Attention(self.dim_hidden)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        logits = self.out(output.view(-1, self.dim_hidden))
        predicted_softmax = F.log_softmax(logits, dim=1).view(batch_size, seq_len, -1)
        return predicted_softmax, hidden, attn

    def forward(self, encoder_outputs, encoder_hidden, targets=None, teacher_forcing_ratio=1):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used.

        Outputs: seq_probs,
        - **seq_probs** (batch_size, max_length, vocab_size): tensors  containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        batch_size, encoder_len, dim_hidden = encoder_outputs.size()
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        seq_probs = []
        seq_preds = []

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            # use targets as rnn inputs
            decoder_input = targets[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            seq_probs = decoder_output

        else:
            # <sos> as initial input
            decoder_input = targets[:, 0].unsqueeze(1)
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                seq_probs.append(decoder_output)
                step_output = decoder_output.squeeze(1)
                _, symbols = torch.max(step_output, 1)
                symbols = symbols.unsqueeze(1)
                seq_preds.append(symbols)
                decoder_input = symbols
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        return seq_probs, seq_preds

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
