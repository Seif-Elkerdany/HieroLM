#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class HieroLM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(HieroLM, self).__init__()
        src_pad_token_idx = vocab.vocab['<pad>']
        self.embed_size = embed_size
        self.model_embeddings = nn.Embedding(len(vocab.vocab),embed_size,padding_idx=src_pad_token_idx)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True)
        self.target_vocab_projection = nn.Linear(2 * hidden_size, len(vocab.vocab), bias=False)

    def forward(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)

        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        X = self.model_embeddings(source_padded)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(nn.utils.rnn.pack_padded_sequence(X,source_lengths))
        enc_hiddens = nn.utils.rnn.pad_packed_sequence(enc_hiddens)[0]

        return enc_hiddens

    def predict(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)
        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        predictions = torch.argmax(P, dim=-1) * target_masks

        return predictions, target_masks, target_padded, source_lengths
    
    def predict_realtime(self, source: List[List[str]], device) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        #print(torch.argmax(P, dim=-1).shape)
        prediction_idx = torch.argmax(P, dim=-1)[-1][0].cpu().item()
        prediction = self.vocab.vocab.id2word[prediction_idx]

        return prediction

    @property
    def device(self) -> torch.device:
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = HieroLM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
