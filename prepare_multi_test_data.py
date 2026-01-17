import json
import numpy as np
import os



JSON_PATH = "calculator_9_2025-03-19_filter.json" 
OUTPUT_DIR = "." # Директория, куда будут сохранены обработанные .npy файлы.
FILLING_VALUE = -1.0 # Значение, которое будет использоваться для заполнения пропусков (NaN/None) и для выравнивания (padding).
STROOP_PROCESSING_WAY = 1

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
        # Логика для создания структурированного 3D-массива 
        all_subtests = []
        max_subtest_len = 0
        
        # Шаг 1: Проходим по каждому из 4-х подтестов, извлекаем их данные
        # и находим максимальную длину подтеста у ДАННОГО испытуемого.
        for key in subtest_keys:
            subtest_seq = []
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    subtest_seq.append([row.get(f) for f in features])                #*  2d
            all_subtests.append(np.array(subtest_seq, dtype=float))  #3d

            # Обновляем максимальную длину подтеста     #*
            if len(all_subtests[-1]) > max_subtest_len:
                max_subtest_len = len(all_subtests[-1])

        # Шаг 2: Теперь, зная максимальную длину, выравниваем каждый из 4-х подтестов до нее.
        padded_subtests = []
        for sub_seq in all_subtests:
            pad_width = max_subtest_len - len(sub_seq)
            if pad_width > 0:
                # Создаем "подушку" из значений-заполнителей, двумерный массив
                padding = np.full((pad_width, num_features), FILLING_VALUE, dtype=float)
                # Добавляем ее к данным подтеста
                padded_subtests.append(np.vstack([sub_seq, padding])) #vstack объединяет массивы по строкам, снизу добавляет
            else:
                padded_subtests.append(sub_seq)

        # Шаг 3: Складываем 4 выровненных 2D-массива в один 3D-массив.
        return np.stack(padded_subtests)

    else: # TStroop_way == 2
        # Логика для простого объединения в 2D-массив 
        seq = []
        for key in subtest_keys:
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    values = []
                    for f in features:
                        values.append(row.get(f)) #*
                    seq.append(values)

        return np.array(seq, dtype=float)


def prepare_test(rec, features):
    """
    Извлекает данные для стандартных тестов (не TStroop) из одной записи.
    """
    seq = []
    for row in rec["test_results"]: #список словарей
        values = []
        for f in features:
            values.append(row.get(f))    #добавляем в values данные по нужным признакам (строка)     *
        seq.append(values)       #добавляем в seq строку
        
    return np.array(seq, dtype=float)


def run_preparation():
    """
    Основная функция, которая запускает весь процесс подготовки данных.
    """
    print(f"Загрузка данных из {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #  Шаг 1: Группировка всех записей по (ID УЧАСТНИКА, ID ПОПЫТКИ)
    # Создаем словарь, где будем хранить все тесты, входящие в одну попытку одного человека.
    print("Группировка данных по (id_person, id_test_attempt)...")
    grouped_by_attempt = {} # NEW

    for record in data:
        person_id = record.get("id_person")
        attempt_id = record.get("id_test_attempt")
        test_name = record.get("test_name")

        key = (person_id, attempt_id)  # NEW

        # Если такой пары (person, attempt) еще нет в словаре, создаем
        if key not in grouped_by_attempt:
            grouped_by_attempt[key] = {
                "age": record.get("age"),
                "person_id": person_id,
                "records": []
            }

        # Добавляем запись о прохождении теста этой попытке
        grouped_by_attempt[key]["records"].append(record)

    # Шаг 2: Обработка всех тестов для каждой попытки
    print("Обработка всех попыток тестов...")
    all_tests_data = {name: [] for name in TEST_NAMES} # Словарь для хранения последовательностей каждого теста
    ages = [] # Список возрастов, соответствующий каждой попытке
    attempt_to_person_map = []  # Карта: индекс попытки -> ID участника

    # Сортируем попытки для консистентности
    sorted_keys = sorted(grouped_by_attempt.keys())

    # Проходим по каждой попытке
    for key in sorted_keys:
        attempt_data = grouped_by_attempt[key]
        attempt_age = attempt_data["age"]
        attempt_records = attempt_data["records"]

        # Создаем временное хранилище тестов одной попытки
        tests_in_attempt = {name: None for name in TEST_NAMES} # NEW

        for rec in attempt_records:
            test_name = rec["test_name"]

            # Обрабатываем только те тесты, которые нас интересуют
            if test_name in TEST_NAMES:
                features = TEST_CONFIG[test_name]

                if test_name == "TStroop":
                    tests_in_attempt[test_name] = prepare_TStroop(
                        rec, features, TStroop_way=STROOP_PROCESSING_WAY
                    )
                else:
                    tests_in_attempt[test_name] = prepare_test(rec, features)

        # Добавляем данные попытки в общий массив
        for test_name in TEST_NAMES:
            all_tests_data[test_name].append(tests_in_attempt[test_name])

        ages.append(attempt_age)
        attempt_to_person_map.append(attempt_data["person_id"])

    #  Шаг 3: Паддинг (выравнивание) и сохранение данных по каждому тесту
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

    # Шаг 4: Сохранение возрастов
    y = np.array(ages, dtype=float)

    output_path = os.path.join(OUTPUT_DIR, "y_aligned.npy")
    print(f"Сохранение выровненных возрастов в {output_path} с формой {y.shape}")
    np.save(output_path, y)

    # Шаг 5: Сохранение карты попыток -> участников
    # Это позволит правильно разделить данные по участникам, а не по попыткам
    np.save(os.path.join(OUTPUT_DIR, "attempt_to_person_map.npy"), np.array(attempt_to_person_map, dtype=object))


if __name__ == "__main__":

    run_preparation()

'''
import json
import numpy as np
import os



JSON_PATH = "calculator_9_2025-03-19_filter.json" 
OUTPUT_DIR = "." # Директория, куда будут сохранены обработанные .npy файлы.
FILLING_VALUE = -1.0 # Значение, которое будет использоваться для заполнения пропусков (NaN/None) и для выравнивания (padding).
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
        # Логика для создания структурированного 3D-массива 
        all_subtests = []
        max_subtest_len = 0
        
        # Шаг 1: Проходим по каждому из 4-х подтестов, извлекаем их данные
        # и находим максимальную длину подтеста у ДАННОГО испытуемого.
        for key in subtest_keys:
            subtest_seq = []
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    subtest_seq.append([row.get(f) for f in features])                #*  2d
            all_subtests.append(np.array(subtest_seq, dtype=float))  #3d

            # Обновляем максимальную длину подтеста     #*
            if len(all_subtests[-1]) > max_subtest_len:
                max_subtest_len = len(all_subtests[-1])

        # Шаг 2: Теперь, зная максимальную длину, выравниваем каждый из 4-х подтестов до нее.
        padded_subtests = []
        for sub_seq in all_subtests:
            pad_width = max_subtest_len - len(sub_seq)
            if pad_width > 0:
                # Создаем "подушку" из значений-заполнителей, двумерный массив
                padding = np.full((pad_width, num_features), FILLING_VALUE, dtype=float)
                # Добавляем ее к данным подтеста
                padded_subtests.append(np.vstack([sub_seq, padding])) #vstack объединяет массивы по строкам, снизу добавляет
            else:
                padded_subtests.append(sub_seq)

        # Шаг 3: Складываем 4 выровненных 2D-массива в один 3D-массив.
        return np.stack(padded_subtests)

    else: # TStroop_way == 2
        # Логика для простого объединения в 2D-массив 
        seq = []
        for key in subtest_keys:
            if key in rec["test_results"]:
                for row in rec["test_results"][key]:
                    values = []
                    for f in features:
                        values.append(row.get(f)) #*
                    seq.append(values)

        return np.array(seq, dtype=float)


def prepare_test(rec, features):
    """
    Извлекает данные для стандартных тестов (не TStroop) из одной записи.
    """
    seq = []
    for row in rec["test_results"]: #список словарей
        values = []
        for f in features:
            values.append(row.get(f))    #добавляем в values данные по нужным признакам (строка)     *
        seq.append(values)       #добавляем в seq строку
        
    return np.array(seq, dtype=float)


def run_preparation():
    """
    Основная функция, которая запускает весь процесс подготовки данных.
    """
    print(f"Загрузка данных из {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #  Шаг 1: Группировка всех записей по ID испытуемого
    # Создаем словарь, где будем хранить все попытки тестов для каждого человека.
    print("Группировка данных по ID испытуемого...")
    grouped_by_person = {} #словарь где значение это список
    for record in data:
        person_id = record.get("id_person")
        test_name = record.get("test_name")

        # Если такого person_id еще нет в словаре, создаем для него запись
        if person_id not in grouped_by_person:
            grouped_by_person[person_id] = {
                "age": record.get("age"),
                "records": []
            }
        # Добавляем запись о прохождении теста этому человеку
        grouped_by_person[person_id]["records"].append(record)

    # Шаг 2: Обработка всех попыток тестов для каждого испытуемого
    print("Обработка всех попыток тестов...")

    # Создаем структуру для хранения данных: для каждого участника и каждого теста
    # будем хранить все попытки
    all_tests_data = {person_id: {name: [] for name in TEST_NAMES} for person_id in grouped_by_person.keys()}
    ages = [] # Список возрастов для каждого участника
    person_ids_ordered = [] # Упорядоченный список ID участников

    # Сортируем ID испытуемых для консистентности
    sorted_person_ids = sorted(grouped_by_person.keys())

    # Проходим по каждому испытуемому
    for person_id in sorted_person_ids:
        person_data = grouped_by_person[person_id]
        person_age = person_data["age"] # Возраст испытуемого
        person_records = person_data["records"] # Все записи (попытки тестов) этого испытуемого

        # Сохраняем возраст и ID участника
        ages.append(person_age)
        person_ids_ordered.append(person_id)

        # Проходим по каждой записи (попытке теста) этого испытуемого
        for rec in person_records:
            test_name = rec["test_name"]

            # Обрабатываем только те тесты, которые нас интересуют
            if test_name in TEST_NAMES:
                features = TEST_CONFIG[test_name]

                # Обрабатываем запись (попытку)
                if test_name == "TStroop":
                    processed_seq = prepare_TStroop(rec, features, TStroop_way=STROOP_PROCESSING_WAY)
                else:
                    processed_seq = prepare_test(rec, features)

                # Добавляем обработанную последовательность к соответствующему тесту для этого участника
                all_tests_data[person_id][test_name].append(processed_seq)

    # Шаг 3: Агрегация данных для каждого теста из всех попыток каждого участника
    # Для каждого теста создаем массив, где для каждого участника будет одна последовательность
    # (например, первая попытка или агрегированная)
    aggregated_tests_data = {name: [] for name in TEST_NAMES}

    for person_id in sorted_person_ids:
        for test_name in TEST_NAMES:
            # Берем первую попытку для каждого теста от каждого участника
            # Если попыток нет, добавляем пустую последовательность
            if len(all_tests_data[person_id][test_name]) > 0:
                # Берем первую попытку
                selected_seq = all_tests_data[person_id][test_name][0]
            else:
                # Если нет попыток, создаем пустую последовательность с правильным числом признаков
                num_features = len(TEST_CONFIG[test_name])
                selected_seq = np.full((1, num_features), FILLING_VALUE, dtype=float)

            aggregated_tests_data[test_name].append(selected_seq)

    #  Шаг 4: Паддинг (выравнивание) и сохранение данных по каждому тесту
    print("Выравнивание и сохранение данных...")
    for test_name in TEST_NAMES:
        X_list = aggregated_tests_data[test_name]
        num_features = len(TEST_CONFIG[test_name])
        batch_size = len(X_list)

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

    # Шаг 5: Сохранение возрастов
    y = np.array(ages, dtype=float)

    output_path = os.path.join(OUTPUT_DIR, "y_aligned.npy")
    print(f"Сохранение выровненных возрастов в {output_path} с формой {y.shape}")
    np.save(output_path, y)

if __name__ == "__main__":
    run_preparation()
'''    