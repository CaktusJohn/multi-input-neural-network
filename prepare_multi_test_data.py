import json
import numpy as np
import os



JSON_PATH = "calculator_9_2025-03-19_filter.json"
# Директория, куда будут сохранены обработанные .npy файлы.
OUTPUT_DIR = "."
# Значение, которое будет использоваться для заполнения пропусков (NaN/None) и для выравнивания (padding).
FILLING_VALUE = -1.0
STROOP_PROCESSING_WAY = 2

TEST_CONFIG = {
    "T1back": ["Stimul", "Goal", "Color", "Duration", "Show", "Hide", "Down", "Up", "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"],
    "TStroop": ["True", "Error", "SMR", "MR", "Time"],
    "T258": ["Stimul", "H", "H+", "dH+", "H-", "dH-", "t+", "t-", "ERR", "ERR_LIM"],
    "T274": ["Stimul", "Goal", "Duration", "Interval", "Show", "Hide", "Down", "Up", "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"],
    "T278": ["Stimul", "Goal", "Duration", "Interval", "Show", "Hide", "Down", "Up", "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"]
}

TEST_NAMES = list(TEST_CONFIG.keys())


def prepare_TStroop(rec, features, TStroop_way=2):
    
    subtest_keys = ["mono", "trueColor", "color", "trueText"]
    num_features = len(features)

    if TStroop_way == 1:
        # --- Логика для создания структурированного 3D-массива ---
        all_subtests = []
        max_subtest_len = 0
        
        # Шаг 1: Проходим по каждому из 4-х подтестов, извлекаем их данные
        # и находим максимальную длину подтеста у ДАННОГО испытуемого.
        for key in subtest_keys:
            subtest_seq = []
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    subtest_seq.append([row.get(f, np.nan) for f in features])
            
            # Если подтест пустой, создаем заглушку, чтобы сохранить структуру из 4-х элементов
            if not subtest_seq:
                all_subtests.append(np.full((1, num_features), np.nan))
            else:
                all_subtests.append(np.array(subtest_seq, dtype=float))

            # Обновляем максимальную длину подтеста
            if len(all_subtests[-1]) > max_subtest_len:
                max_subtest_len = len(all_subtests[-1])

        # Шаг 2: Теперь, зная максимальную длину, выравниваем каждый из 4-х подтестов до нее.
        padded_subtests = []
        for sub_seq in all_subtests:
            pad_width = max_subtest_len - len(sub_seq)
            if pad_width > 0:
                # Создаем "подушку" из значений-заполнителей
                padding = np.full((pad_width, num_features), FILLING_VALUE, dtype=float)
                # Добавляем ее к данным подтеста
                padded_subtests.append(np.vstack([sub_seq, padding]))
            else:
                padded_subtests.append(sub_seq)

        # Шаг 3: "Складываем" 4 выровненных 2D-массива в один 3D-массив.
        return np.stack(padded_subtests)

    else: # TStroop_way == 2
        # --- Логика для простого объединения в 2D-массив ---
        seq = []
        for key in subtest_keys:
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    values = []
                    for f in features:
                        values.append(row.get(f, np.nan))
                    seq.append(values)
        if not seq:
            return np.full((1, len(features)), np.nan)
        return np.array(seq, dtype=float)


def prepare_test(rec, features):
    """
    Извлекает данные для стандартных тестов (не TStroop) из одной записи.
    """
    seq = []
    for row in rec["test_results"]: #список словарей
        values = []
        for f in features:
            values.append(row.get(f, np.nan))    #добавляем в values данные по нужным признакам (строка)
        seq.append(values)       #добавляем в seq строку
        
    # Если запись теста пустая, вернем массив с одной строкой NaN'ов
    if not seq:
        return np.full((1, len(features)), np.nan)
        
    return np.array(seq, dtype=float)


def run_preparation():
    """
    Основная функция, которая запускает весь процесс подготовки данных.
    """
    print(f"Загрузка данных из {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Шаг 1: Группировка всех записей по ID испытуемого ---
    # Создаем словарь, где будем хранить все попытки тестов для каждого человека.
    print("Группировка данных по ID испытуемого...")
    grouped_by_person = {} #словар где значение это список
    for record in data:
        person_id = record.get("id_person")
        test_name = record.get("test_name")

        # Пропускаем записи, если в них нет id или они не относятся к нужным тестам
        if not person_id or test_name not in TEST_NAMES:
            continue
        
        # Если такого person_id еще нет в словаре, создаем для него запись
        if person_id not in grouped_by_person:
            grouped_by_person[person_id] = {
                "age": record.get("age"),
                "records": []
            }
        # Добавляем запись о прохождении теста этому человеку
        grouped_by_person[person_id]["records"].append(record)

    # --- Шаг 2: Обработка данных для каждого испытуемого ---
    # Теперь для каждого человека мы извлечем данные по каждому из 5 тестов.
    # Будем использовать 5 списков, по одному на каждый тест.
    print("Обработка тестов для каждого испытуемого...")
    all_tests_data = {name: [] for name in TEST_NAMES} # словарь с пустыми значениями и ключами
    ages = []
    
    # Сортируем ID, чтобы порядок испытуемых был всегда одинаковым
    sorted_person_ids = sorted(grouped_by_person.keys())

    for person_id in sorted_person_ids:
        person_data = grouped_by_person[person_id]
        ages.append(person_data["age"])
        
        person_records = person_data["records"]
        
        # Для каждого из 5 тестов ищем, проходил ли его данный человек
        for test_name in TEST_NAMES:
            
            # Ищем запись, соответствующую текущему тесту
            found_record = None
            for rec in person_records:
                if rec["test_name"] == test_name:
                    found_record = rec
                    break # Нашли, берем первую попавшуюся попытку
            
            # Если человек проходил тест, обрабатываем его
            if found_record:
                features = TEST_CONFIG[test_name]
                if test_name == "TStroop":
                    processed_seq = prepare_TStroop(found_record, features, TStroop_way=STROOP_PROCESSING_WAY)
                else:
                    processed_seq = prepare_test(found_record, features)
                all_tests_data[test_name].append(processed_seq)
            else:
                # Если человек НЕ проходил тест, добавляем `None` в качестве метки
                all_tests_data[test_name].append(None)

    # --- Шаг 3: Паддинг (выравнивание) и сохранение данных по каждому тесту ---
    print("Выравнивание и сохранение данных...")
    for test_name in TEST_NAMES:
        X_list = all_tests_data[test_name]
        num_features = len(TEST_CONFIG[test_name])
        batch_size = len(X_list)
        
        if test_name == "TStroop" and STROOP_PROCESSING_WAY == 1:
            max_subtest_len = 0
            for seq_person_4d in X_list:
                if seq_person_4d is not None and seq_person_4d.ndim == 3:
                    if seq_person_4d.shape[1] > max_subtest_len:
                        max_subtest_len = seq_person_4d.shape[1]
            if max_subtest_len == 0: max_subtest_len = 1
            
            num_subtests_stroop = 4
            X_padded = np.full((batch_size, num_subtests_stroop, max_subtest_len, num_features), FILLING_VALUE, dtype=float)
            
            for i, seq_person_4d in enumerate(X_list):
                if seq_person_4d is not None and seq_person_4d.ndim == 3:
                    seq_person_4d[np.isnan(seq_person_4d)] = FILLING_VALUE
                    X_padded[i, :, :seq_person_4d.shape[1], :] = seq_person_4d
            
            output_path = os.path.join(OUTPUT_DIR, f"X_{test_name}.npy")
            print(f"Сохранение данных {test_name} в {output_path} с формой {X_padded.shape}")
            np.save(output_path, X_padded)
        
        else:
            # Находим максимальную длину последовательности в данном тесте
            max_len = 0
            for seq in X_list:
                if seq is not None:
                    if len(seq) > max_len:
                        max_len = len(seq)
            if max_len == 0: max_len = 1 # Защита от случая, если тест никем не пройден
            
            # Создаем пустой массив-контейнер, заполненный значением-заполнителем
            X = np.full((batch_size, max_len, num_features), FILLING_VALUE, dtype=float)
            
            # Копируем данные каждого испытуемого в этот массив
            for i, seq in enumerate(X_list):
                if seq is not None:
                    # Заменяем все nan (пропущенные значения) на наше значение-заполнитель
                    seq[np.isnan(seq)] = FILLING_VALUE
                    # Копируем последовательность в итоговый массив
                    X[i, :seq.shape[0], :] = seq
            
            # Сохраняем итоговый массив
            output_path = os.path.join(OUTPUT_DIR, f"X_{test_name}.npy")
            print(f"Сохранение данных {test_name} в {output_path} с формой {X.shape}")
            np.save(output_path, X)

    # --- Шаг 4: Сохранение возрастов ---
    y = np.array(ages, dtype=float)
    # Заменяем возможные пропуски в возрастах
    y[np.isnan(y)] = FILLING_VALUE
    
    output_path = os.path.join(OUTPUT_DIR, "y_aligned.npy")
    print(f"Сохранение выровненных возрастов в {output_path} с формой {y.shape}")
    np.save(output_path, y)

if __name__ == "__main__":
    run_preparation()