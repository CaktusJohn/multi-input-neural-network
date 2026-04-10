import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(context)
        return output

class ImprovedRNNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.2, output_dim=32):
        super().__init__()
        
        # Bidirectional RNN with more layers
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Bidirectional doubles the hidden dimension
        rnn_output_dim = hidden_dim * 2
        
        # Multi-head attention
        self.attention = MultiHeadAttention(rnn_output_dim * 2, num_heads=4)
        
        # Layer normalization - use correct normalized_shape
        self.layer_norm1 = nn.LayerNorm([rnn_output_dim * 2])
        self.layer_norm2 = nn.LayerNorm([rnn_output_dim * 2])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(rnn_output_dim * 2, rnn_output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim * 2, rnn_output_dim * 2),
            nn.Dropout(dropout)
        )
        
        # Residual connection projection
        self.residual_proj = nn.Linear(rnn_output_dim * 2, output_dim)  # Fix: use rnn_output_dim * 2 for bidirectional
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(rnn_output_dim * 2, output_dim),  # Fix: use rnn_output_dim * 2 for bidirectional
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)  # Less dropout at output
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)  # [batch, seq_len, hidden_dim*2]
        
        # Apply layer normalization - use correct dimensions
        rnn_out_norm = self.layer_norm1(rnn_out)
        
        # Multi-head attention with residual connection
        attn_out = self.attention(rnn_out_norm)
        attn_out = self.dropout(attn_out)
        rnn_out = rnn_out + attn_out  # Residual connection
        rnn_out = self.layer_norm2(rnn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(rnn_out)
        
        # Simple mean pooling instead of attention
        context_vector = torch.mean(rnn_out, dim=1)  # [batch, hidden_dim*2]
        
        # Final projection
        output = self.output_proj(context_vector)
        return output

class ImprovedMultiInputRNNModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, num_layers=3, dropout=0.2, output_dim=1):
        super().__init__()
        
        # Create improved RNN branches for each test
        self.branches = nn.ModuleDict({
            test_name: ImprovedRNNBranch(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=32
            )
            for test_name, input_dim in input_dims.items()
        })
        
        # Calculate concatenated size
        concatenated_size = 32 * len(input_dims)
        
        # Deep head with better architecture
        self.head = nn.Sequential(
            # First block
            nn.Linear(concatenated_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            # Second block
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            # Third block
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            
            # Fourth block
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            
            # Final output
            nn.Linear(32, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, xs):
        branch_outputs = []
        for i, test_name in enumerate(self.branches.keys()):
            branch_outputs.append(self.branches[test_name](xs[i]))
        concatenated = torch.cat(branch_outputs, dim=1)
        prediction = self.head(concatenated)
        return prediction

def create_model(model_type="improved_rnn", input_dims=None, hidden_dim=64, num_layers=3, dropout=0.2):
    if input_dims is None:
        raise ValueError("input_dims must be provided")
    
    return ImprovedMultiInputRNNModel(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
