import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initializes the multi-head attention mechanism.
        Args:
            d_model (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
        Attributes:
            d_model (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimensionality of each attention head.
            qkv_proj (nn.Linear): Linear layer to project the input into query, key, and value vectors.
            out_proj (nn.Linear): Linear layer to project the concatenated output of all attention heads.
        """
        # Initialize the parent class
        super().__init__()
        # Store the model dimensions and number of heads
        self.d_model = d_model
        self.num_heads = num_heads
        # Calculate the dimension of each attention head
        self.head_dim = d_model // num_heads
        
        # Define the linear layers for projecting the input
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """
        Perform the forward pass of the multi-head attention mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_len, seq_len) 
                           to prevent attention to certain positions. Default is None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying 
                  multi-head attention and projection.
        """
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scaling = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """
    A feed-forward neural network module used in the Transformer model.
    Args:
        d_model (int): The dimensionality of the input and output features.
        d_ff (int): The dimensionality of the hidden layer.
        dropout (float, optional): The dropout probability. Default is 0.1.
    Attributes:
        net (nn.Sequential): The sequential container of the feed-forward network layers.
    Methods:
        forward(x):
            Passes the input tensor through the feed-forward network.
    Example:
        >>> feed_forward = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 512)  # Batch size of 64, feature size of 512
        >>> output = feed_forward(x)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    This class implements a single layer of the Transformer encoder as described in the paper
    "Attention Is All You Need" by Vaswani et al.
    Args:
        d_model (int): The dimension of the input and output vectors.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward network.
        dropout (float, optional): The dropout rate. Default is 0.1.
    Attributes:
        self_attn (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward (FeedForward): Feed-forward network.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        norm2 (nn.LayerNorm): Layer normalization after feed-forward network.
        dropout (nn.Dropout): Dropout layer.
    Methods:
        forward(x, mask=None):
            Performs a forward pass through the encoder layer.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
                mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length, seq_length). Default is None.
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple layers of TransformerEncoderLayer.
    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): Dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        dropout (float, optional): Dropout rate. Default is 0.1.
    Methods:
        forward(x, mask=None):
            Passes the input through the encoder layers.
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).
                mask (Tensor, optional): Mask tensor of shape (batch_size, seq_length). Default is None.
            Returns:
                Tensor: Output tensor of shape (batch_size, seq_length, d_model).
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    Args:
        d_model (int): The dimension of the model.
        max_len (int, optional): The maximum length of the input sequences. Default is 5000.
    Attributes:
        pe (torch.Tensor): The positional encoding matrix of shape (1, max_len, d_model).
    Methods:
        forward(x):
            Adds positional encoding to the input tensor.
            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            Returns:
                torch.Tensor: The input tensor with positional encoding added, of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    """
    Transformer model consisting of an embedding layer, positional encoding, and multiple encoder layers.
    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): Dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        vocab_size (int): Size of the vocabulary.
        dropout (float, optional): Dropout rate. Default is 0.1.
    Attributes:
        embedding (nn.Embedding): Embedding layer to convert input tokens to embeddings.
        pos_encoding (PositionalEncoding): Positional encoding to add positional information to embeddings.
        encoder (TransformerEncoder): Transformer encoder consisting of multiple encoder layers.
        dropout (nn.Dropout): Dropout layer.
    Methods:
        forward(x, mask=None):
            Passes the input through the embedding layer, positional encoding, dropout, and encoder layers.
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_length).
                mask (Tensor, optional): Mask tensor of shape (batch_size, seq_length). Default is None.
            Returns:
                Tensor: Output tensor of shape (batch_size, seq_length, d_model).
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Perform the forward pass of the Transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length). Default is None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        return self.encoder(x, mask)