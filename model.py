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
        self.num_layers = 4

        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = embed_size if i == 0 else hidden_size
            self.encoder_layers.append(
                nn.LSTM(input_dim, hidden_size, bias=True, bidirectional=False)
            )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.vocab), bias=False)
        
        self.layer_norm_input = nn.LayerNorm(embed_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)

        self.target_vocab_projection.weight = self.model_embeddings.weight
        self.init_weights()

    def init_weights(self):
        """
        Orthogonal initialization for Recurrent weights
        Xavier initialization for Input weights
        Constant 1 initialization for Forget Gate bias
        """
        # We now iterate over the ModuleList
        for layer in self.encoder_layers:
            for name, param in layer.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    n = self.hidden_size
                    param.data[n:2*n].fill_(1.0)

    def forward(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)

        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        target_gold_words_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Manual loop over LSTM layers to implement Residual (Skip) Connections
        """
        X = self.model_embeddings(source_padded)
        X = self.layer_norm_input(X)
        X = self.dropout(X)

        # We need to maintain the packed structure for efficiency,
        # but we must unpack to do the addition (Residual Connection), then repack.
        
        current_input = X
        
        for i, layer in enumerate(self.encoder_layers):
            # 1. Pack
            packed_input = nn.utils.rnn.pack_padded_sequence(current_input, source_lengths, enforce_sorted=False)
            
            # 2. Pass through LSTM layer
            packed_output, (hidden, cell) = layer(packed_input)
            
            # 3. Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
            
            # 4. Apply Dropout
            output = self.dropout(output)

            # 5. Residual Connection (Skip Connection)
            # We can only add if shapes match. 
            # Usually Layer 0 changes size (Embed -> Hidden), so we skip residual there.
            # Layers 1+ are Hidden -> Hidden, so we add residual.
            if i > 0 or (self.embed_size == self.hidden_size):
                output = output + current_input
            
            # Update current_input for the next iteration
            current_input = output

        # Final LayerNorm
        enc_hiddens = self.layer_norm_output(current_input)

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
        return self.model_embeddings.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(
            model_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,  
        )

        args = params['args']  # dict: embed_size, hidden_size, dropout_rate
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
