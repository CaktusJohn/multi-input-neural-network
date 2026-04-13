import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class BiLSTMBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3, output_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output dimension after bidirectional concatenation
        lstm_output_dim = hidden_dim * 2
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        
        # BiLSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Output projection
        output = self.output_proj(context_vector)  # [batch_size, output_dim]
        
        return output

class MultiInputBiLSTMModel(nn.Module):
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3, output_dim: int = 1):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create BiLSTM branches for each test
        self.branches = nn.ModuleDict({
            test_name: BiLSTMBranch(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=64  # Each branch outputs 64 features
            )
            for test_name, input_dim in input_dims.items()
        })
        
        # Calculate concatenated size
        concatenated_size = 64 * len(input_dims)
        
        # Deep head network
        self.head = nn.Sequential(
            # First layer
            nn.Linear(concatenated_size, concatenated_size // 2),
            nn.LayerNorm(concatenated_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Second layer
            nn.Linear(concatenated_size // 2, concatenated_size // 4),
            nn.LayerNorm(concatenated_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Third layer
            nn.Linear(concatenated_size // 4, concatenated_size // 8),
            nn.LayerNorm(concatenated_size // 8),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            
            # Final output
            nn.Linear(concatenated_size // 8, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        branch_outputs = []
        
        for i, (test_name, branch) in enumerate(self.branches.items()):
            branch_output = branch(xs[i])
            branch_outputs.append(branch_output)
        
        # Concatenate all branch outputs
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Pass through head network
        prediction = self.head(concatenated)
        
        return prediction

def create_bilstm_model(input_dims: Dict[str, int], hidden_dim: int = 128, 
                       num_layers: int = 2, dropout: float = 0.3, output_dim: int = 1) -> MultiInputBiLSTMModel:
    """
    Create a MultiInputBiLSTMModel with the specified parameters.
    
    Args:
        input_dims: Dictionary mapping test names to input dimensions
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        output_dim: Output dimension (default 1 for regression)
    
    Returns:
        MultiInputBiLSTMModel instance
    """
    return MultiInputBiLSTMModel(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim
    )

if __name__ == "__main__":
    # Test the model
    input_dims = {
        'test1': 10,
        'test2': 8,
        'test3': 12,
        'test4': 15,
        'test5': 20
    }
    
    model = create_bilstm_model(input_dims)
    
    # Create dummy data
    batch_size = 4
    seq_len = 50
    xs = [torch.randn(batch_size, seq_len, dim) for dim in input_dims.values()]
    
    # Forward pass
    output = model(xs)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model structure
    print("\nModel structure:")
    print(model)
