import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Pitch, Velocity, Duration, Position, Bar]
        self.n_tokens = [89, 66, 4609, 1537, 518]      # [3,18,88,66]
        self.classes = ['Pitch', 'Velocity', 'Duration', 'Position', 'Bar']
        self.emb_sizes = [32, 64, 512, 256, 128]

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i in range(len(self.classes)):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)
        
        self.dense = nn.Linear(6, bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        emb_linear = self.dense(input_ids.float())
        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        return y
    
    def get_rand_tok(self):
        vel_rand = random.choice(range(self.n_tokens[1]))
        return vel_rand
            

class MidiBertLM(nn.Module):
    def __init__(self, midibert: MidiBert):
        super().__init__()
        
        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.hidden_size)

    def forward(self, x, attn, performer):
        x = self.midibert(x, attn)
        return self.mask_lm(x, performer)
        
class VelActivation(nn.Module):
    def __init__(self, inputs=None):
        super(VelActivation, self).__init__()
        if inputs is not None:
            self.inputs = inputs
    
    def forward(self, x):
        return torch.clamp(x, 0, 66)

class DurActivation(nn.Module):
    def __init__(self, inputs=None):
        super(DurActivation, self).__init__()
        if inputs is not None:
            self.inputs = inputs
    
    def forward(self, x):
        return torch.clamp(x, -1000, 1000)
    
class PosActivation(nn.Module):
    def __init__(self, inputs=None):
        super(PosActivation, self).__init__()
        if inputs is not None:
            self.inputs = inputs
    
    def forward(self, x):
        return torch.clamp(x, 0)


class MLM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i in range(3):
            self.proj.append(nn.Linear(hidden_size + 128, 1))
        self.proj = nn.ModuleList(self.proj)
        self.dense = nn.Linear(hidden_size + 128, 128)
        
        self.performer_emb = Embeddings(6, 128) #final: 128
        self.velact = VelActivation()
        self.duract = DurActivation()
        self.posact = PosActivation()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, y, performer):
        # feed to bert 
        y = y.hidden_states[-1]

        y_p = self.performer_emb(performer)
        y_p = y_p[:, None, :]
        y_p = torch.repeat_interleave(y_p, 1000, dim=1)
        y = torch.cat([y, y_p], dim=-1)
                
        ys = []
        for i in range(3):
            if i == 0:
                y0 = self.proj[i](y)
                y0 = self.velact(y0)
                ys.append(y0) # (batch_size, seq_len, dict_size)
            elif i == 1:
                y1 = self.proj[i](y)
                y1 = self.duract(y1)
                ys.append(y1) # (batch_size, seq_len, dict_size)
            elif i == 2:
                y2 = self.proj[i](y)
                y2 = self.posact(y2)
                ys.append(y2) # (batch_size, seq_len, dict_size)
        return ys
        