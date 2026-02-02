from dataclasses import dataclass
import numpy as np


class LayerNorm:

    def __init__(self,n_emb):
        
        self.gain=np.ones(n_emb)
        self.bias=np.zeros(n_emb)

    def forward(self,x,):
        #Batch norm implementation
        self.bmeani=x.mean(-1,keepdims=True)
        self.bvari=x.var(-1,keepdims=True)
        eps=1e-5
        self.bvari=(self.bvari+eps)**-0.5
        self.bdiff=(x-self.bmeani)*self.bvari
        self.output=x*self.bdiff+self.bias

        return self.output
    
class Block:

    def __init__(self,config):
        
        self.ln_1=LayerNorm(config.n_emb)
        self.ln_2=LayerNorm(config.n_emb)
        self.attn=CausalSelfAttention(config)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size=1024
    vocab_size=50257 #Total number of token types
    n_emb=786
    n_layer=12
    n_head=12

class GPT():

    def __init__(self,config):
        
        self.config=config
        
        self.transformer={
            'wte':np.random.randn(config.n_emb,config.n_emb),#Token embeddings
            'wpe':np.random.randn(config.n_emb,config.n_emb),#Positional embeddings
            'h':[Block(config)for _ in range(config.n_layer)],#Number of heads/attention blocks
            'ln_f':LayerNorm(config.n_emb)
        }

        #Classifier
        self.lm_head=np.random.randn(config.n_emb,config.n_vocab_size)


    def forward(self,idx,targets=None):
 
        B,T=idx.shape
        assert T<=self.block.size,"Cannot forward sequence length beyond block size"

        #forward the token and positional embeddings
        pos=np.arange(0,T,dtype=np.long,device=idx.device) # T
        pos_emb=self.transformer['wpe'][pos] # T,n_emb
        tok_emb=self.transformer['wte'][idx] # T,n_emb
        x=pos_emb+tok_emb

        #forward the blocks of transformers
        for block in self.transformer.h:
            x=block(x)

        #Forward the final layer norm and classifier
        x=self.transformer['ln_f'](x)
        logits=x @ self.lm_head #(B,T,vocab_size)

        loss=None
        
        if targets is not None:

            #Reshaping logits and targets to compute the cross entropy loss as per PyTorch documents
            logits=np.reshape(-1,logits.shape[-1])
            targets=targets.flatten()

            #Cross_entropy
            logits_max=np.max(logits,1,keepdims=True)
            logits-=logits_max #Normalization
            counts=np.exp(logits)
            probs=counts/np.sum(counts,1,keepdims=True)
            loss=-probs[np.arange(0,T),targets].log().mean()

        return logits,loss



        
        
        
