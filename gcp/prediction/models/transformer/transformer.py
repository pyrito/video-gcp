import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import copy

from gcp.prediction.models.transformer.encoder import Encoder, EncoderLayer
from gcp.prediction.models.transformer.decoder import Decoder, DecoderLayer
from gcp.prediction.models.transformer.attention import MultiHeadedAttention
from gcp.prediction.models.transformer.embeddings import PositionalEncoding, Embeddings
from gcp.prediction.models.transformer.utils import PositionwiseFeedForward


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


def make_model(src_vocab, tgt_vocab, N=6, 
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

def transformer_forward(model, context, sequence):
    # TODO(rnair, vkarthik):
    # - Need goal state to be prepended to the sequence
        # Decoder Output: S0 S1 S2 ... SG <STOP>
        #                  |  |  |      |   |
        # Encoder Input:  SG S0 S1 ... SG-1 SG
    # - Need to mask out upper triangle
    # - Need to figure out what mask/stop sequences are
    # - Need to interface with gcp input/output structure

    # Some stuff in this might be helpful:
    # class Batch:
    #     "Object for holding a batch of data with mask during training."
    #     def __init__(self, src, trg=None, pad=0):
    #         self.src = src
    #         self.src_mask = (src != pad).unsqueeze(-2)
    #         if trg is not None:
    #             self.trg = trg[:, :-1]
    #             self.trg_y = trg[:, 1:]
    #             self.trg_mask = \
    #                 self.make_std_mask(self.trg, pad)
    #             self.ntokens = (self.trg_y != pad).data.sum()
        
    #     @staticmethod
    #     def make_std_mask(tgt, pad):
    #         "Create a mask to hide padding and future words."
    #         tgt_mask = (tgt != pad).unsqueeze(-2)
    #         tgt_mask = tgt_mask & Variable(
    #             subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    #         return tgt_mask
    pass


def transformer_loss(model, logits, labels):
    # TODO(rnair, vkarthik): need to write the loss function that returns a dictionary as follows:
    # tf_loss: {
    #   'value': 1 x 1 tensor of loss value,
    #   'weight': 1.0,
    #   'breakdown': length seq_len(79) tensor of loss value per item in sequence
    # }
    # Use MSELoss to calculate loss between each predicted element and its corresponding decoded vector
    # sum of MSELosses is the total loss.
    pass