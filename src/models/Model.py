# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 19:08
# @Author  : Leesure
# @File    : Model.py
# @Software: PyCharm
# import random

import torch.nn as nn
import torch
from torch.autograd import Variable
import random
from transformers.generation_beam_search import BeamSearchScorer


# noinspection PyTypeChecker
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
            input:  (1) src (max_len, batch_size)
                    (2) trg (max_len, batch_size)

        """
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size

        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
        # encoder_output = (max_len, batch_size, hid_size)
        # hidden = (num_layers*bidirectional, batch_size, hid_size)
        encoder_output, hidden = self.encoder(src)
        #
        hidden = hidden[:self.decoder.n_layers]
        output = trg.data[0, :].unsqueeze(0)  # [1, batch_size], value = [SOS]

        for t in range(1, max_len):
            # output = [batch_size, vocab_size]
            output, hidden = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

    """Use Beam search to generate the Questions"""

    def generate(self, src, max_len, num_beams, device,
                 eos_token_id: int, pad_token_id: int, sos_token_id: int,
                 gene_mode="beam_search", ):
        assert gene_mode == "beam_search" and num_beams > 1, \
            f"In beam_search model, {num_beams} can must more than one"

        batch_size = src.size(1)
        vocab_size = self.decoder.output_size
        beam_scorer = BeamSearchScorer(batch_size, max_length=max_len, num_beams=num_beams, device=device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        hidden = hidden.repeat(1, num_beams, 1)
        encoder_output = encoder_output.repeat(1, num_beams, 1)
        # output = [batch_size, ].repeat = [1, batch_size * num_beams]
        output = torch.tensor([sos_token_id]).repeat(1, batch_size * num_beams).to(device)
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        # beam_scores= [batch_size* num_beams,1]
        beam_scores = beam_scores.view(-1, 1)
        input_seq = output.transpose(0, 1)
        next_tokens = torch.zeros(batch_size * num_beams)
        next_indices = torch.zeros(batch_size * num_beams)
        for t in range(1, max_len):
            # output = [batch_size*num_beams, vocab_size]

            output, hidden = self.decoder(output, hidden, encoder_output)
            # process output into size of [batch_size*num_beams, vocab_size]
            next_token_scores = output + beam_scores
            next_token_scores = next_token_scores.view(batch_size, -1)
            next_token_scores, next_token_index = torch.topk(next_token_scores, 2 * num_beams,
                                                             dim=1, largest=True, sorted=True)
            next_indices = next_token_index // vocab_size
            next_tokens = next_token_index % vocab_size
            beam_output = beam_scorer.process(input_seq, next_token_scores, next_tokens, next_indices,
                                              pad_token_id=pad_token_id, eos_token_id=eos_token_id)

            beam_scores = beam_output["next_beam_scores"].unsqueeze(1)
            output = beam_output["next_beam_tokens"].unsqueeze(0)
            next_beam_indices = beam_output["next_beam_indices"]
            input_seq = torch.cat([input_seq[next_beam_indices, :], beam_output["next_beam_tokens"].unsqueeze(1)],
                                  dim=-1)

        sequence_output = beam_scorer.finalize(input_seq, beam_scores, next_tokens, next_indices,
                                               pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        return sequence_output
