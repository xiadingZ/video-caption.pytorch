import torch.nn as nn


class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None,
                teacher_forcing_ratio=1):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels
            teacher_forcing_ratio (int, optional):

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_hidden=encoder_hidden,
                                           encoder_outputs=encoder_outputs,
                                           targets=target_variable,
                                           teacher_forcing_ratio=teacher_forcing_ratio)
        return seq_prob, seq_preds
