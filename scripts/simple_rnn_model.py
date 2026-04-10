import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.1, output_dim=16):
        super().__init__()
        # Простая RNN (не LSTM!)
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Два полносвязных слоя как у вас было
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)
        
        # Берем последний hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Два полносвязных слоя как у вас
        x = self.fc1(last_hidden)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SimpleMultiInputRNNModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=32, num_layers=2, dropout=0.1, output_dim=1):
        super().__init__()
        
        # Создаем ветки для каждого теста
        self.branches = nn.ModuleDict({
            test_name: SimpleRNNBranch(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=16
            )
            for test_name, input_dim in input_dims.items()
        })
        
        # Размерность после конкатенации
        concatenated_size = 16 * len(input_dims)
        
        # Голова как у вас была
        self.head = nn.Sequential(
            nn.Linear(concatenated_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
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

def create_model(model_type="simple_rnn", input_dims=None, hidden_dim=32, num_layers=2, dropout=0.1):
    if input_dims is None:
        raise ValueError("input_dims must be provided")
    
    return SimpleMultiInputRNNModel(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
