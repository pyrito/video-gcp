from typing_extensions import Required
import numpy as np

from blox import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from gcp.prediction.models.transformer.encoder import Encoder, EncoderLayer
from gcp.prediction.models.transformer.decoder import Decoder, DecoderLayer
from gcp.prediction.models.transformer.attention import MultiHeadedAttention
from gcp.prediction.models.transformer.embeddings import PositionalEncoding, Embeddings
from gcp.prediction.models.transformer.utils import PositionwiseFeedForward, make_std_mask


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_transformer(N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # TODO(rnair, vkarthik): dont need embeddings, dont need generator (since this is seq2seq)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        c(position),
        c(position))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def transformer_forward(model, e_0, e_g, sequence, inp_masks):
    outputs = AttrDict()
    sequence = sequence.squeeze()
    e_0 = e_0.squeeze().unsqueeze(1)
    e_g = e_g.squeeze().unsqueeze(1)
    src_input = torch.cat([e_g, e_0, sequence[:, :-1, :]], dim=1)
    tgt_output = torch.cat([e_0, sequence], dim=1)
    tgt_mask = make_std_mask(inp_masks, 0)
    src_masks = torch.bmm(inp_masks.unsqueeze(2), inp_masks.unsqueeze(1))
    out = model.forward(src_input, tgt_output, 
                            src_masks, tgt_mask)
    
    outputs.raw_x = out
    out = out[:, 1:, :].unsqueeze(-1).unsqueeze(-1)
    outputs.x = out
    return outputs

def transformer_loss(model, logits, labels, seq_len):
    loss_info = AttrDict()
    loss = nn.MSELoss()
    breakdown = torch.Tensor(seq_len)
    for i in range(seq_len):
        breakdown[i] = loss(logits[:, i, :], labels[:, i, :])
    value = torch.mean(breakdown)
    loss_info.value = value
    loss_info.weight = 1.0
    loss_info.breakdown = breakdown[1:]
    return loss_info