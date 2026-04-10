
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Улучшенный механизм многоголового внимания.
    Позволяет модели фокусироваться на различных аспектах последовательности.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value проекции
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Выходная проекция
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Проекции
        Q = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Разделение на головы
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Применение весов
        context = torch.matmul(attention_weights, V)
        
        # Объединение голов
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Глобальный пулинг (среднее по последовательности)
        context_vector = context.mean(dim=1)  # (batch_size, hidden_dim)
        
        return self.out_proj(context_vector), attention_weights.mean(dim=1)  # возвращаем средние веса по головам


class SubtestBranch(nn.Module):
    """
    Класс, определяющий архитектуру одной "ветки" нейросети.
    Каждая ветка обрабатывает данные одного конкретного подтеста.
    Ее задача - взять последовательность данных и преобразовать ее в вектор признаков фиксированного размера.
    """
    def __init__(self, input_dim, lstm_hidden_dim=64, lstm_layers=2, output_dim=32, num_heads=4):
        super().__init__()
        # LSTM-слой для обработки последовательности с dropout
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, 
                           batch_first=True, dropout=0.1 if lstm_layers > 1 else 0)
        
        # Улучшенный механизм многоголового внимания
        self.attention = MultiHeadAttention(lstm_hidden_dim, num_heads, dropout=0.1)
        
        # Двухслойная обработка для лучшей feature extraction
        self.fc1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2)
        self.fc2 = nn.Linear(lstm_hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.1)
        # Layer normalization вместо BatchNorm
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x имеет размерность: (размер батча, длина последовательности, количество признаков)

        # LSTM возвращает выходы на каждом шаге
        lstm_outputs, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        # Применяем многоголовое внимание
        context_vector, attention_weights = self.attention(lstm_outputs)
        
        # Двухслойная обработка с dropout
        x = self.relu(self.fc1(context_vector))
        x = self.dropout(x)
        embedding = self.relu(self.fc2(x))
        embedding = self.layer_norm(embedding)

        return embedding

class MultiInputModel(nn.Module):
    """
    Основная модель нейросети. Она состоит из пяти независимых "веток" (SubtestBranch),
    по одной для каждого подтеста. Выходы всех веток затем объединяются (конкатенируются)
    и подаются в общую "голову" для финального предсказания.
    """
    def __init__(self, input_dims: dict, common_hidden_dim=256, output_dim=1):
        """
        Args:
            input_dims (dict): Словарь, который сопоставляет имя теста с количеством признаков в нем.
                               Пример: {"T1back": 13, "TStroop": 5, ...}
        """
        super().__init__()
        
        # `nn.ModuleDict` - это удобный способ хранения нескольких слоев/модулей в виде словаря.
        # Создаем по одной "ветке" (SubtestBranch) для каждого теста, передавая ей нужное количество входных признаков.
        self.branches = nn.ModuleDict({
            test_name: SubtestBranch(input_dim=dim)
            for test_name, dim in input_dims.items()
        })
        
        # Вычисляем общий размер вектора после объединения выходов всех веток.
        # Каждая ветка выдает вектор размером 32 (улучшенная емкость)
        concatenated_size = 32 * len(input_dims)
        
        # "Голова" модели - это несколько полносвязных слоев, которые принимают объединенный вектор
        # и делают на его основе финальное предсказание (в нашем случае - возраст).
        '''
        self.head = nn.Sequential(
            nn.Linear(concatenated_size, common_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Слой Dropout для регуляризации (борьбы с переобучением)
            nn.Linear(common_hidden_dim, common_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_hidden_dim // 2, output_dim) # Выходной слой с одним нейроном
        )'''
        # Углубленная голова с residual connections и layer normalization
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


    def forward(self, xs: list[torch.Tensor]):
        """
        Args:
            xs (list[torch.Tensor]): Список из 5 тензоров, по одному для каждого подтеста,
                                     в заранее определенном порядке.
        """
        branch_outputs = []
        # Прогоняем каждый входной тензор через соответствующую ему "ветку".
        for i, test_name in enumerate(self.branches.keys()):
            branch_outputs.append(self.branches[test_name](xs[i]))
            
        # Объединяем (конкатенируем) выходы всех веток в один длинный вектор.
        # `dim=1` означает, что мы склеиваем их по второй размерности (размер батча, признаки).
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Подаем объединенный вектор в "голову" для получения финального предсказания.
        prediction = self.head(concatenated)
        
        return prediction
