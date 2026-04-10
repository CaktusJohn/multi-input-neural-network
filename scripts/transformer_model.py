import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x

class TransformerBranch(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, output_dim=32):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class HybridBranch(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, output_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.transformer_encoder = TransformerEncoder(128, nhead, num_layers, dim_feedforward, dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MultiInputTransformerModel(nn.Module):
    def __init__(self, input_dims, model_type="transformer", d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, output_dim=1):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "transformer":
            self.branches = nn.ModuleDict({
                test_name: TransformerBranch(
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                for test_name, input_dim in input_dims.items()
            })
        elif model_type == "hybrid":
            self.branches = nn.ModuleDict({
                test_name: HybridBranch(
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                for test_name, input_dim in input_dims.items()
            })
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        branch_output_dim = 32
        concatenated_size = branch_output_dim * len(input_dims)
        
        self.head = nn.Sequential(
            nn.Linear(concatenated_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, xs):
        branch_outputs = []
        for i, test_name in enumerate(self.branches.keys()):
            branch_outputs.append(self.branches[test_name](xs[i]))
        concatenated = torch.cat(branch_outputs, dim=1)
        prediction = self.head(concatenated)
        return prediction

def create_model(model_type="transformer", input_dims=None, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
    if input_dims is None:
        raise ValueError("input_dims must be provided")
    
    return MultiInputTransformerModel(
        input_dims=input_dims,
        model_type=model_type,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
