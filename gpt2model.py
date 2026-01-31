from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
#-----------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #key,query,value projections for all heads but in a batch
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)

        #output_projection
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        
        #regularization
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self,x):
        B,T,C=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        
        #View k,q,v (B,T,C)->B,nh,T,C/nh
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        #Flash attention
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        
        #output projection
        y=self.c_proj(y)
        return y
        
class MLP(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self,config):
        """
        Layer Normalization is include pre attention and post attention 
        This is different from Attention is All you need Paper
        where normalized inputs was passed into residual connections
        """
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd)
        ))

        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        #weight sharing scheme
        self.transformer.wte.weight=self.lm_head.weight

        #initialization params 
        self.apply(self._init_weights)

        # Inside the GPT class __init__, after self.apply(self._init_weights):
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # GPT-2 initialization scaling for residual layers
                # std = 0.02 / sqrt(2 * n_layer)
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * (2 * self.config.n_layer)**-0.5)

    def _init_weights(self,module):

        if isinstance(module,nn.Linear):
            #Linear layer std->0.02
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        #idx is of shape(B,T)
        B,T=idx.size()

        assert T<=self.config.block_size,f"Cannot forward sequence of length {T},block size limit exceeded"
        
        #forward the token and positional embeddings
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)#shape T
        pos_emb=self.transformer.wpe(pos)#positional embeddings of shape (T,n_embd)
        tok_emb=self.transformer.wte(idx)#Token embeddings of shape(B,T,n_embd)
        x=tok_emb+pos_emb

        #forward the blocks of transformer
        for block in self.transformer.h:
            x=block(x)
            
        #forward the final layernorm and the classifier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) #(B,T,vocab_size)

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    
    @classmethod
    def from_pretrained(cls,model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer,n_head,and n_embd are determined from model_type
        config_args={
            'gpt2':          dict(n_layer=12,n_head=12,n_embd=768), #124M params
            'gpt2-medium':   dict(n_layer=24,n_head=16,n_embd=1024),#350M params
            'gpt2-large':    dict(n_layer=36,n_head=20,n_embd=1280),#774M params
            'gpt2-xl':       dict(n_layer=48,n_head=25,n_embd=1600)
        }[model_type]

        config_args['vocab_size']=50257
        config_args['block_size']=1024

        config=GPTConfig(**config_args)
        model=GPT(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')] #discard this with mask/buffer

        model_hf=GPT2LMHeadModel.from_pretrained(f"openai-community/{model_type}")
        sd_hf=model_hf.state_dict()

        sd_keys_hf=sd_hf.keys()
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked.bias')]
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed=['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        #basically the openai checkpoints use a Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf)==len(sd_keys),f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w)for w in transposed):
               #special treatment for the Conv1D weights and we need to transpose
               with torch.no_grad():
                   sd[k].copy_(sd_hf[k].t())

            else:
                #vanilla copy over other parameters
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    #--------------------------------------------------------

class DataLoaderLite:

    def __init__(self,B,T):
        self.B=B
        self.T=T

        with open('input.txt','r')as f:
            text=f.read()

        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T) } batches")
        
        #state
        self.current_position=0

    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=(buf[:-1]).view(B,T)#inputs
        y=(buf[1:]).view(B,T)#targets

        #advance position in the tensor
        self.current_position+=B*T

        #if loading the next batch would be out of tokens,then reset
        if self.current_position+(B*T+1)>len(self.tokens):
            self.current_position=0

        return x,y

num_return_sequences=5
max_length=50

#Finding device type
device='cpu'
if torch.cuda.is_available():
    device='cuda'
elif getattr(torch.backends,'mps') and torch.backends.mps.is_available():
    device='mps'

print(f"Running on device : {device}")

train_loader=DataLoaderLite(B=4,T=1024)

#Torch uses TF32
torch.set_float32_matmul_precision('high')

#model=GPT.from_pretrained('gpt2')
model=GPT(GPTConfig()) # Our own model
model.eval()
model.to(device)
model=torch.compile(model)



optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
#Loss calculation by picking next batches
for i in range(50):
    t0=time.time()
    x,y=train_loader.next_batch()
    x=x.to(device)
    y=y.to(device)

    #Autocasting forward pass for AMP
    #Some operations in float32,bfloat16
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,loss=model(x,y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)*1000 #milliseconds 
    tokens_per_sec=(train_loader.B * train_loader.T)/(t1-t0)
    print(f"step {i} | Loss {loss.item():.4f} | Time taken : {dt:.4f} ms | Tokens per sec: {tokens_per_sec:.4f} ")

import sys;sys.exit(0)()


#sampling

import tiktoken
enc=tiktoken.get_encoding('gpt2')

#generate! x is (B,T)which is (5,8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1)<max_length:
    #Forward the model to get logits:
    with torch.no_grad():
        logits=model(x) #(B,T,vocab_size)
        #take the logits at last position
        logits=logits[:,-1,:]
        #get the probabilies
        probs=F.softmax(logits,-1)

        topk_probs,topk_indicies=torch.topk(probs,50,dim=-1)
        ix=torch.multinomial(topk_probs,1)
        xcol=torch.gather(topk_indicies,-1,ix)
        x=torch.cat([x,xcol],dim=-1)

for i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)

