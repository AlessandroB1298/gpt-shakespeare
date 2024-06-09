import torch
import torch.nn as nn
from torch.nn import functional as F
# read lines from data file

block_size = 8
batch_size = 64
learning_rate = 3e-1
max_iters = 5000# steps
eval_iters = 500
n_embeds = 32
num_heads = 6
n_layer = 6
dropout= 0.2 #regularization technique 
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use gpu if you have one

torch.manual_seed(1337)

with open("/Users/alchemie/Desktop/chat/data/input.txt",  "r", encoding="utf-8") as data:
    text= data.read()
    #print(text[:1000])

chars = sorted(list(set(text)))

vocab_size = len(chars)
#print(" ".join(chars))
#print(vocab_size)


#Create our mapping for tokenization

stoi = {ch: i for i,ch in enumerate(chars)} # iterate through all the chars and create a lookup table for the chars to the integers 

itoi = {i: ch for i,ch in enumerate(chars)} # Same thing but from the integers to the chars.

# Lambda functions are anoymous functions that can have any number of arguments but can only return on expression

encode = lambda s: [stoi[c] for c in s] # encoding a string, by translating all of the chars individually 

decode = lambda l: ''.join([itoi[i] for i in l]) # decode by translating the integers to chars and joining them together

#print(encode("hii there"))
#print(decode(encode("hii there")))

# tokenize the entire dataset using torch

data = torch.tensor(encode(text) , dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

'''
split into train and validation
'''

n = int(0.9 * len(data)) # gathering 90 percent of our data and using that as train_data
train_data = data[:n]
valid_data = data[n:]


# evaulate loss function, that gets the average between splits train and val
@torch.no_grad() # context manager tells pytorch that we will not use .backward on anything in the following function

def estimate_loss():
    out = {}
    model.eval() #eval 
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()

        out[split] = losses.mean() #get average loss between the two splits
    model.train() #train mode
    return out

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self,head_size):
        super(). __init__()
        self.key = nn.Linear(n_embeds, head_size, bias=None) # vector saying what do I contain
        self.query = nn.Linear(n_embeds, head_size, bias=None) #vector that essentially is saying what am I looking for.
        self.dropout = nn.Dropout(dropout)
        # these make up our weights

        self.value = nn.Linear(n_embeds, head_size, bias=None)# Later on when we are calculating our self-attention we use value to aggregate, because we don't want to use our raw x value
        #tril or lower-triangle part of a 2d matrix we use for self-attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #[T,T]

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        weights = q @ k.transpose(-2,-1) * C**-0.5 #[B,T,C] @ [B,C,T] ----> [B,T,T]
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        # perform the weighted aggregation on the values
        v = self.value(x)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__ (self, num_heads, head_size):
        super(). __init__()
        self.heads= nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size , n_embeds)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer followed by a non-linearity"""
    def __init__ (self, n_embeds):
        super(). __init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(4 * n_embeds, n_embeds),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embeds, num_heads):
        super(). __init__()
        head_size = n_embeds//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd= FeedForward(n_embeds)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))# added the x + for risidual connections, so we can fork off, do computations, and come back
        x = x + self.ffwd(self.ln2(x))
        return x

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #generates random posotions of batch_size offsets of 4x8 tensor, gives 4 nums generated by len of data -block_size
    x = torch.stack([data[i:i+block_size] for i in ix])  # first block_size char starting at i
    y = torch.stack([data[i+1: i+block_size+1] for i in ix]) # similar but it is the next char hence i+1
    return x,y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds) # create emebedding table that is 65x65
        self.position_embedding_table = nn.Embedding(block_size, n_embeds) #also need idx positions going forward 
        #self.sa_head = Head(n_embeds) # create self_attention_head in our language model class
        #self.sa_heads = MultiHeadAttention(4, n_embeds//4) #4 heads for 8-dimensional self_attention
        #self.ffwd = FeedForward(n_embeds)
        self.blocks = nn.Sequential(
            Block(n_embeds, num_heads=num_heads),
            Block(n_embeds, num_heads=num_heads),
            Block(n_embeds, num_heads=num_heads),
            nn.LayerNorm(n_embeds),
        )
        self.lm_head = nn.Linear(n_embeds, vocab_size) #linear layer because our model is getting more complex, best practice
        

    def forward(self, idx, targets=None):
        # logits are the raw outputs from the final layer of the deep learning model
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) #batch by time by channel tensor or [64,8,65]
        pos_embed = self.position_embedding_table(torch.arange(T, device = device)) #[T,C] integers from 0 -> T-1
        x = tok_embed + pos_embed # [B,T,C]
        x = self.blocks(x)
        logits = self.lm_head(x) #[B,T,vocab_size]
        
        if targets == None: #if we have the targets we will get a loss value, else we will just get the logits
            loss = None
        else:   
            # reshape our logits tensor 
            B,T,C = logits.shape 
            logits = logits.view(B*T, C) #stretching out the array from 3d to 2d
            targets = targets.view(B*T) # targets is in the shape of B,T, now it is a vector 
            loss = F.cross_entropy(logits, targets) 
        return logits, loss # We are predicting what is coming next based off of the identity of a single tokens

    #generate function
    def generate(self,idx,max_new_tokens):
        #idx is an array [B,T] in our current context
        for _ in range(max_new_tokens):
            #get the predictions
            #crop idx to the last block_size
            idx_cond = idx[:, -block_size:] # because we use pos_embed, idx can never be larger than block_size, because we will run out of scope
            logits,loss = self(idx_cond)
            logits = logits[:, -1, :] #get the last step, convert to [B, C], we take out the T dimension, because they aren't predicitons for what comes next
            probs = F.softmax(logits, dim=1) # convert to softmax, [0,1]
            idx_next = torch.multinomial(probs, num_samples=1) # [B,1] for each batch dimension, we will get a single prediction for what comes next.
            idx = torch.cat((idx, idx_next), dim=1) # append sample index or integers to the running dimension, [B, T+1]
        return idx
    

#xb, yb = get_batch('train')

model = BigramLanguageModel()
m = model.to(device)
# for smaller nerual networks we can get away with much higher learning rates if we use AdamW compared to SGD
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) #AdamW is a much more complex optimizer, it customizes each parameter's learning rate based on its graident history

def training_loop(batch_size: int):
    for iter in range(max_iters):
        if iter % eval_iters == 0 or iter == max_iters -1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
        xb,yb = get_batch('train')
        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(decode(m.generate(idx = torch.zeros((1,1) , dtype=torch.long) , max_new_tokens=500)[0].tolist())) 


# calling our training loop with a batch size of 32
training_loop(batch_size)




#this will print inaccurate readings because right now it is a random model
#print(decode(m.generate(idx = torch.zeros((1,1) , dtype=torch.long) , max_new_tokens=100)[0].tolist())) 

