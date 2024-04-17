import numpy as np
import pandas as pd
from datasets import load_dataset
import sentencepiece as spm
import gensim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import torch
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import torch.nn.functional as F
import numpy as np
import random
import math


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, pad_mask=None, atn_mask=False):
    d_k = q.size()[-1]
    
    # Move q, k, and v tensors to the same device
    q, k, v = q.to(get_device()), k.to(get_device()), v.to(get_device())
    
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if atn_mask:
        dia_mask = torch.full(scaled.size(), float('-inf'), device=get_device())
        dia_mask = torch.triu(dia_mask, diagonal=1)
        scaled += dia_mask
    attention = F.softmax(scaled, dim=-1)
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1) * pad_mask.unsqueeze(1).unsqueeze(3)
        # Move pad_mask to the same device
        pad_mask = pad_mask.to(get_device())
        attention = attention.masked_fill(pad_mask==0, 0)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self, device=torch.device('cpu')):  # Pass device as an argument
        even_i = torch.arange(0, self.d_model, 2).float().to(device)  # Move tensor to device
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length, device=device).reshape(self.max_sequence_length, 1)  # Move tensor to device
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, pad_mask=None, atn_mask=False):
        batch_size, sequence_length, d_model = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, pad_mask, atn_mask = atn_mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.self_attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    # Override the forward method to handle None values for pad_mask
    def forward(self, y, pad_mask, atn_mask):
        _y = y
        
        # Check if pad_mask is None before attempting to move it to device
        if pad_mask is not None:
            pad_mask = pad_mask.to(get_device())
        
        y = self.self_attention1(y, pad_mask, atn_mask)
        y = self.dropout1(y) 
        y = self.norm1(y + _y) 
        _y = y
        
        y = self.ffn(y) 
        y = self.dropout3(y) 
        y = self.norm3(y + _y) 
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        y, pad_mask, atn_mask = inputs
        for module in self._modules.values():
            y = module(y, pad_mask, atn_mask) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, y, pad_mask = None, atn_mask = True):
        y = self.layers(y, pad_mask, atn_mask)
        return y

class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                vocab_size               
                ):
        super().__init__()
        self.d_model = d_model

        self.dec_embedding = nn.Embedding(vocab_size, d_model)
        self.dec_pos_encoding = PositionalEncoding(d_model, 1)
        
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, input_tokens, pad_mask=None, atn_mask=True):
        
        # Move input tensors to the appropriate device
        input_tokens = input_tokens.to(self.device)
        pad_mask = pad_mask.to(self.device) if pad_mask is not None else None
        max_sequence_length = input_tokens.shape[1]  # Fix this line to get the correct sequence length

        # Compute token embeddings
        token_embeddings = self.dec_embedding(input_tokens) 
        self.dec_pos_encoding = PositionalEncoding(self.d_model, max_sequence_length)
        token_pos_encodings = self.dec_pos_encoding(device=self.device)  # Pass device argument
        token_embeddings_with_pos = token_embeddings + token_pos_encodings.unsqueeze(0)

        # Perform the rest of the forward pass
        out = self.decoder(token_embeddings_with_pos, pad_mask, atn_mask)
        out = self.linear(out)
        return out
