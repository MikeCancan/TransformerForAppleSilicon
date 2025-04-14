import torch
import torch.nn as nn
import math

'''
Embedding Layer : Translate the word into vector
'''
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(self.vocab_size,self.d_model)
    
    def forward(self,x):
        return self.embed(x) * math.sqrt(self.d_model)
    
'''
Positional Encoding Layer:
Mark the position of every word, and encode it into the vector.
(otherwise 'A loves B' equals 'B loves A')
'''
class PositionCoding(nn.Module):
    def __init__(self,seq_len,d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        pe = torch.zeros(size=(seq_len,d_model))
        position = torch.arange(0,seq_len).reshape(-1,1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        #Static statistics
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        # x.shape:(batch,seq_len,d_model)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return x
'''
The core part: Attention Mechanism

Query,Key,Value and the attention_score matrix.
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head:int,dropout:float):
        super().__init__()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.head = head
        self.d_k = self.d_model // head
        assert self.d_model%head == 0
    
    def attention(self,query,key,value,mask):
        #q,k,v are the same size as input sentence,which is (batch,seq_len,d_model)
        d_k = self.d_model // self.head
        attention_score = (query@key.transpose(-1,-2)) / math.sqrt(d_k)
        seq_len = attention_score.shape[2]

        future_mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(device='mps')
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)
        future_mask = future_mask.expand(attention_score.shape[0],attention_score.shape[1],seq_len,seq_len)
        future_mask = future_mask.masked_fill_(future_mask==1,-1e9)

        #Match the dimension of attention_score
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score = attention_score + future_mask

        #attention_score: (batch,seq_len,seq_len)
        attention_score = attention_score.softmax(dim=-1)
        #attention@value : (batch,seq_len,d_mdoel)
        return (attention_score@value),attention_score
    
    def forward(self,q,k,v,mask):
        #(batch,seq_len,d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # now we need to divide them into every head
        #SIZE:(batch,head,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.head,-1).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.head,-1).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.head,-1).transpose(1,2)
        #Original X:(batch,seq_len,d_model)--->After passing the param to attention:(batch,head,seq_len,d_k)---->at last (batch,seq_len,d_model)
        x,attention_score = self.attention(query,key,value,mask)
        #Translate the result from every head into one final head:(batch,seq_len,d_model),in which we can obtain different features from different dimension
        x = x.contiguous().view(x.shape[0],x.shape[2],self.d_k * self.head)
        return self.w_o(x)
'''
Normalize the feature,which is the last dimension.

Norm(x) = gamma*[(x-mean)/sqrt(var+eps)]+beta
And remember, d_model is our target(the last dimension)
'''
class LayerNorm(nn.Module):
    def __init__(self,d_model:int,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        x = (x-mean) / torch.sqrt(var+self.eps)
        return self.gamma * x + self.beta
'''

Projection Layer: Full Connect: d_model -> vocab_size

'''
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        #Point the specific dimension
        return torch.log_softmax(self.linear(x),dim=-1)
    
'''
Residual Connection: verry easy to accomplish
The function of this layer is to plus the input and the output, which can
decrease the possibility of grad disappearing.
for further knowledge and using, please check ResNet
'''
class ResidualConnection(nn.Module):
    def __init__(self,d_model:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(self.d_model)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
'''
FeedForward Layer:
d_model -> d_ff -> d_model
'''
class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.linear_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
'''
EncoderBlock: Attention: embedding layer and positional coding layer are not belong to EncoderBlock
1.src_mask is essential to encoder
2.the function of the encoderblock is just to unify those layers I have defined above
'''
class EncoderBlock(nn.Module):
    def __init__(self,d_model:int,feedforward:FeedForward,attention:MultiHeadAttention,dropout:float):
        super().__init__()
        self.feedforward = feedforward
        self.attention = attention
        self.residual = nn.ModuleList([ResidualConnection(d_model,dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,src_mask):
        x = self.residual[0](x,lambda x:self.attention(x,x,x,src_mask))
        #We just need to pass the sublayer into residual layer
        x = self.residual[1](x,self.feedforward)
        return x
'''
Now I defined every layer in Encoder, and unifyed them in encoderblock.
Now I encapsulate encoderblock into Encoder
'''
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model:int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
    
    def forward(self,x,src_mask):
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)
'''
As the same,now I define the DecoderBlock.
Watch the cross-attention Mechanism.
'''
class DecoderBlock(nn.Module):
    def __init__(self,self_attention:MultiHeadAttention,cross_attention:MultiHeadAttention,feedforward:FeedForward,d_model:int,dropout:float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.residual = nn.ModuleList([ResidualConnection(d_model,dropout) for _ in range(3)])
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual[0](x,lambda x:self.self_attention(x,x,x,tgt_mask))
        x = self.residual[1](x,lambda x:self.cross_attention(encoder_output,x,encoder_output,src_mask))
        x = self.residual[2](x,self.feedforward)
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model:int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

'''
Now we encapsulate every class to define transformer
'''
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:Embedding,tgt_embed:Embedding,src_pos:PositionCoding,tgt_pos:PositionCoding,proj:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj
    
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,tgt,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.proj(x)
'''
Encapsulate all of these into a function, which include all of the hyperparameter

attention:
'''

def build_transformer(src_vocab,tgt_vocab,src_seq,tgt_seq,d_model=256,d_ff=2048,head=8,N=6,dropout=0.1):
    #embedding 
    src_embed = Embedding(src_vocab,d_model)
    tgt_embed = Embedding(tgt_vocab,d_model)
    #positionCoding
    src_pos = PositionCoding(src_seq,d_model)
    tgt_pos = PositionCoding(tgt_seq,d_model)

    #Build encoder blocks

    encoder_blocks = []
    
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model,head,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(d_model,feed_forward_block,encoder_self_attention_block,dropout) 
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,head,dropout)
        cross_attention_block = MultiHeadAttention(d_model,head,dropout)
        dec_feedforward = FeedForward(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,cross_attention_block,dec_feedforward,d_model,dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model=d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks),d_model=d_model)

    projection_layer = ProjectionLayer(d_model,tgt_vocab)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer