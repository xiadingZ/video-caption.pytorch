import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .Attention import Attention


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

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, rnn_dropout_p=0.1):
        super().__init__()

        self.bidirectional_encoder = bidirectional
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
                                 bidirectional=bidirectional, batch_first=True, dropout=rnn_dropout_p)

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(dim_hidden)

        self.out = nn.Linear(dim_hidden, self.dim_output)

    def forward(self, encoder_outputs, encoder_hidden, targets=None, teacher_forcing_ratio=1):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used.

        Outputs: seq_probs,
        - **seq_probs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        seq_probs = []
        seq_preds = []

        decoder_hidden = None
        if use_teacher_forcing:
            # use targets as rnn inputs
            for i in range(self.max_length - 1):
                current_words = self.embedding(targets[:, i])
                if decoder_hidden is None:
                    context = torch.mean(encoder_outputs, dim=1)
                else:
                    context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat(
                    [current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                self.rnn.flatten_parameters()
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logits = self.out(decoder_output.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        else:
            # <sos> as initial input
            current_words = Variable(torch.LongTensor(
                [self.sos_id] * batch_size)).cuda().unsqueeze(1)
            current_words = self.embedding(current_words)
            for i in range(self.max_length - 1):
                if decoder_hidden is None:
                    context = torch.mean(encoder_outputs, dim=1)
                else:
                    context = self.attention(decoder_hidden, encoder_outputs)
                decoder_input = torch.cat(
                    [current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                self.rnn.flatten_parameters()
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logits = self.out(decoder_output.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        return seq_probs, seq_preds

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h)
                                    for h in encoder_hidden])
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
