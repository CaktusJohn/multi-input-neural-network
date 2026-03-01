import torch
import torch.nn as nn

class SubtestBranch(nn.Module):
    """
    Класс, определяющий архитектуру одной "ветки" нейросети.
    Каждая ветка обрабатывает данные одного конкретного подтеста.
    Ее задача - взять последовательность данных и преобразовать ее в вектор признаков фиксированного размера.
    """
    def __init__(self, input_dim, lstm_hidden_dim=32, lstm_layers=1, output_dim=16):
        super().__init__()
        # LSTM-слой для обработки последовательности. `batch_first=True` означает, что
        # входной тензор будет иметь размерность (batch_size, sequence_length, num_features).
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        # Полносвязный слой, который преобразует выход LSTM в вектор-представление (embedding) меньшего размера.
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        # Функция активации
        self.relu = nn.ReLU()

    def forward(self, x):
        # x имеет размерность: (размер батча, длина последовательности, количество признаков)
        
        # LSTM возвращает `output` (выходы на каждом шаге) и кортеж `(h_n, c_n)` (последнее скрытое и клеточное состояние).
        # Нам нужно только последнее скрытое состояние `h_n`, так как оно содержит обобщенную информацию о всей последовательности.
        _, (h_n, _) = self.lstm(x)
        
        # `h_n` имеет размерность: (количество слоев LSTM, размер батча, размер скрытого состояния).
        # Берем скрытое состояние самого последнего слоя LSTM.
        last_hidden_state = h_n[-1] # Размерность: (размер батча, размер скрытого состояния)
        
        # Пропускаем это состояние через полносвязный слой, чтобы получить итоговый вектор-представление.
        embedding = self.fc(last_hidden_state)
        embedding = self.relu(embedding)
        
        return embedding

class MultiInputModel(nn.Module):
    """
    Основная модель нейросети. Она состоит из пяти независимых "веток" (SubtestBranch),
    по одной для каждого подтеста. Выходы всех веток затем объединяются (конкатенируются)
    и подаются в общую "голову" для финального предсказания.
    """
    def __init__(self, input_dims: dict, common_hidden_dim=128, output_dim=1):
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
        # Каждая ветка выдает вектор размером 16 (как определено в `output_dim` класса SubtestBranch).
        concatenated_size = 16 * len(input_dims)
        
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
        self.head = nn.Sequential(
            nn.Linear(concatenated_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
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
