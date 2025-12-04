import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Импортируем нашу архитектуру модели из соседнего файла.
from multi_input_model import MultiInputModel

# --- Конфигурация ---
# Список имен тестов в том порядке, в котором мы будем их загружать и подавать в модель.
TEST_NAMES = ["T1back", "TStroop", "T258", "T274", "T278"]
# Размер батча - количество примеров, обрабатываемых за один шаг обучения.
BATCH_SIZE = 32
# Скорость обучения (learning rate) - шаг, с которым модель обновляет свои веса.
LEARNING_RATE = 1e-4 
# Максимальное количество эпох обучения.
NUM_EPOCHS = 170
# "Терпение" для механизма ранней остановки. Если ошибка на валидации не улучшается
# в течение `PATIENCE` эпох, обучение прекращается.
PATIENCE = 10 

# этот параметр должен совпадать с STROOP_PROCESSING_WAY в prepare_multi_test_data.py
STROOP_PROCESSING_WAY = 2

# --- 1. Загрузка данных ---
print("Загрузка подготовленных данных...")
# Загружаем 5 массивов с данными тестов в список `Xs`.
Xs = []
for name in TEST_NAMES:
    data = np.load(f"X_{name}.npy")
    # Специальная обработка для TStroop, если он был сохранен как 4D тензор
    if name == "TStroop" and STROOP_PROCESSING_WAY == 1:
        # Преобразуем (batch_size, num_subtests, max_len_subtest, num_features) в
        # (batch_size, num_subtests * max_len_subtest, num_features)
        batch_size, num_subtests, max_len_subtest, num_features = data.shape
        data = data.reshape(batch_size, num_subtests * max_len_subtest, num_features)
    Xs.append(data)
y = np.load("y_aligned.npy")

# --- 2. Cоздание собственного класса Dataset ---
class MultiInputDataset(Dataset):
    """
    Кастомный класс датасета для PyTorch. 
    """
    def __init__(self, xs_list, y_arr):
        # Преобразуем все numpy-массивы в тензоры PyTorch при инициализации.
        self.xs = [torch.tensor(x, dtype=torch.float32) for x in xs_list]
        self.y = torch.tensor(y_arr, dtype=torch.float32)

    def __len__(self):
        # Возвращает общее количество примеров в датасете.
        return len(self.y)

    def __getitem__(self, idx):
        # Возвращает один пример из датасета по его индексу `idx`.
        # Результат - это список из 5 тензоров (по одному срезу от каждого теста) и одно число (возраст).
        return [x[idx] for x in self.xs], self.y[idx]

# --- 3. Разделение данных на обучающую, валидационную и тестовую выборки ---
# делим не сами данные, а ИНДЕКСЫ. Это гарантирует, что для одного и того же
# испытуемого данные из всех 5 тестов попадут в одну и ту же выборку (train/val/test).
indices = list(range(len(y)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42) 

def create_subset(indices_list):
    """Вспомогательная функция для создания датасета по списку индексов."""
    subset_xs = [x[indices_list] for x in Xs]
    subset_y = y[indices_list]
    return MultiInputDataset(subset_xs, subset_y)

print("Создание загрузчиков данных (DataLoader)...")
train_dataset = create_subset(train_indices)
val_dataset = create_subset(val_indices)
test_dataset = create_subset(test_indices)

# DataLoader'ы будут подавать данные в модель батчами. `shuffle=True` для обучающей выборки
# перемешивает данные в начале каждой эпохи, что улучшает обучение.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Инициализация модели ---
# Собираем словарь с количеством признаков для каждого теста. Это нужно для инициализации модели.
input_dims = {name: Xs[i].shape[2] for i, name in enumerate(TEST_NAMES)}

model = MultiInputModel(input_dims=input_dims)
# Функция потерь (Loss Function). L1Loss - это MAE.
criterion = nn.L1Loss() 
# Оптимизатор. Adam - один из самых популярных и эффективных алгоритмов оптимизации.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Модель инициализирована:")
print(model)

# --- 5. Цикл обучения ---
best_val_loss = float('inf') # Начальное значение лучшей ошибки на валидации (бесконечность).
patience_counter = 0 # Счетчик для ранней остановки.

print(f"Начало обучения на {NUM_EPOCHS} эпох...")
for epoch in range(NUM_EPOCHS):
    # --- Фаза обучения (Training) ---
    model.train() # Переводим модель в режим обучения.
    train_loss = 0
    # tqdm - обертка для `train_loader` для отображения красивого progress bar'а.
    for x_batch, y_batch in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS} [Обучение]"):
        optimizer.zero_grad() # Обнуляем градиенты с предыдущего шага.
        y_pred = model(x_batch).squeeze() # Делаем предсказание и убираем лишние размерности.
        loss = criterion(y_pred, y_batch) # Считаем ошибку.
        loss.backward() # Вычисляем градиенты (обратное распространение ошибки).
        optimizer.step() # Обновляем веса модели.
        train_loss += loss.item() # Суммируем ошибку.
    
    avg_train_loss = train_loss / len(train_loader)

    # --- Фаза валидации (Validation) ---
    model.eval() # Переводим модель в режим оценки (отключаются Dropout и т.д.).
    val_preds = []
    val_targets = []
    with torch.no_grad(): # В этом блоке градиенты не вычисляются для экономии ресурсов.
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch).squeeze()
            val_preds.append(y_pred.cpu().numpy()) # Собираем предсказания
            val_targets.append(y_batch.cpu().numpy()) # и реальные значения.
            
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    # Считаем среднюю абсолютную ошибку (MAE) на валидационной выборке.
    avg_val_loss = mean_absolute_error(val_targets, val_preds)

    print(f"Эпоха [{epoch+1}/{NUM_EPOCHS}] | Ошибка на обучении: {avg_train_loss:.4f} | Ошибка на валидации (MAE): {avg_val_loss:.4f}")

    # --- Ранняя остановка и сохранение лучшей модели ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Сохраняем состояние модели (ее веса), если она показала лучший результат.
        torch.save(model.state_dict(), 'best_multi_input_model.pth')
        print(f"Ошибка на валидации улучшилась. Модель сохранена в 'best_multi_input_model.pth'")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Ранняя остановка: ошибка не улучшалась {patience_counter} эпох.")
            break

# --- 6. Финальная оценка на тестовой выборке ---
print("\n--- Тестирование ---")
# Загружаем веса лучшей модели, сохраненной ранее.
model.load_state_dict(torch.load('best_multi_input_model.pth'))
model.eval() # Переводим в режим оценки.

test_preds = []
test_targets = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader, desc="[Тест]"):
        y_pred = model(x_batch).squeeze()
        test_preds.append(y_pred.cpu().numpy())
        test_targets.append(y_batch.cpu().numpy())

test_preds = np.concatenate(test_preds)
test_targets = np.concatenate(test_targets)

# Считаем и выводим финальные метрики на данных, которые модель еще не видела.
test_mse = mean_squared_error(test_targets, test_preds)
test_mae = mean_absolute_error(test_targets, test_preds)

print(f"\nИтоговые результаты на тестовой выборке:")
print(f"  Средняя квадратичная ошибка (MSE): {test_mse:.4f}")
print(f"  Средняя абсолютная ошибка (MAE): {test_mae:.4f}")
