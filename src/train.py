"""Architecture copied unchanged from the original trained instrument."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class LanguageModel(nn.Module):
    def __init__(self,vocab_size,architecture,with_position,config):
        super().__init__();self.architecture=architecture;self.with_position=with_position
        d=config['embedding_dim'];self.embedding=nn.Embedding(vocab_size,d,padding_idx=0)
        self.events=nn.Embedding(8,d)
        nn.init.normal_(self.embedding.weight,std=.04)
        nn.init.normal_(self.events.weight,std=.02)
        self.dropout=nn.Dropout(config['dropout'])
        if architecture=='lstm':
            self.core=nn.LSTM(d,config['lstm_hidden'],num_layers=2,batch_first=True,dropout=config['dropout'])
            self.project=nn.Linear(config['lstm_hidden'],d)
        elif architecture=='transformer':
            self.position=nn.Embedding(config['context_length'],d)
            layer=nn.TransformerEncoderLayer(d,config['heads'],dim_feedforward=config['ff_dim'],
                dropout=config['dropout'],batch_first=True,norm_first=True,activation='gelu')
            self.core=nn.TransformerEncoder(layer,4,enable_nested_tensor=False);self.project=nn.LayerNorm(d)
        else:raise ValueError(architecture)
        self.output_bias=nn.Parameter(torch.zeros(vocab_size))
        self.count=sum(p.numel() for p in self.parameters())
        if self.count>5_000_000:raise ValueError(f'Model exceeds 5M: {self.count}')

    def forward(self,x,events,return_hidden=False):
        h=self.embedding(x)
        if self.with_position:h=h+self.events(events)
        if self.architecture=='transformer':
            h=h*math.sqrt(h.shape[-1])+self.position(torch.arange(x.shape[1],device=x.device))
            mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
            h=self.core(self.dropout(h),mask=mask,is_causal=True)
        else:h,_=self.core(self.dropout(h))
        h=self.project(h)
        logits=F.linear(h,self.embedding.weight,self.output_bias)
        return (logits,h) if return_hidden else logits
