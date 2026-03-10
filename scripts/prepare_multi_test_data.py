import json
import numpy as np
import os

JSON_PATH = "calculator_9_2025-03-19_filter.json"
OUTPUT_DIR = "." # Директория для сохранения .npy файлов (корень проекта).
STROOP_PROCESSING_WAY = 2

TEST_CONFIG = {

    "T1back": { "col" : ["Stimul", "Goal", "Color", "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"],
                "length" : [38]
    },

    "T274": { "col" : ["Stimul", "Goal",  "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"],
            "length" : [15] },

    "T278": { "col" : ["Stimul", "Goal",  "SMR", "MR", "ERR_1", "ERR_2", "ERR_3"],
                "length" : [30]
    },
    "TStroop": {"col": ["True", "Error", "SMR", "MR", "Time"],
                "length": [29],  # Длина каждого подтеста
                "subtests": ["mono", "trueColor", "color", "trueText"]
                },
    "T258": {"col": ["Stimul", "H", "H+", "dH+", "H-", "dH-", 	"t+", 	"t-", 	"ERR", 	"ERR_LIM"],
                "length": [13]
    }
}

TEST_NAMES = list(TEST_CONFIG.keys())


def validate_test_lengths(data):
    """
    Проверяет, что длины всех тестов соответствуют указанным в конфиге.
    Вызывает ValueError при несоответствии.
    """
    print("Проверка длин тестов...")
    for record in data:
        test_name = record.get("test_name")

        expected_length = TEST_CONFIG[test_name]["length"][0]  
        person_id = record.get("id_person")
        attempt_id = record.get("id_test_attempt")
        context = f"id_person={person_id}, id_test_attempt={attempt_id}"

        if test_name == "TStroop":
            # Проверка длины каждого подтеста TStroop
            subtests = TEST_CONFIG[test_name].get("subtests")
            for subtest_key in subtests:
                actual_length = len(record["test_results"][subtest_key])
                if actual_length != expected_length:
                    raise ValueError(
                        f"Несоответствие длины подтеста {test_name}/{subtest_key}: "
                        f"ожидалось {expected_length}, получено {actual_length} ({context})"
                    )
        else:
            # Проверка длины обычного теста
            actual_length = len(record["test_results"])
            if actual_length != expected_length:
                raise ValueError(
                    f"Несоответствие длины теста {test_name}: "
                    f"ожидалось {expected_length}, получено {actual_length} ({context})"
                )

    print("Все тесты имеют корректную длину.")


def prepare_TStroop(rec, features, TStroop_way=2):
    
    subtest_keys = ["mono", "trueColor", "color", "trueText"]
    num_features = len(features)

    if TStroop_way == 1:
        pass
    else: 
        # TStroop_way == 2
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

    # Шаг 1: Проверка длин тестов
    validate_test_lengths(data)

    #  Шаг 2: Группировка всех записей по (ID УЧАСТНИКА, ID ПОПЫТКИ)
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

    #шаг 3 из формата «по попыткам» → в формат «по типам тестов»
    print("Обработка всех попыток тестов...")
    all_tests_data = {name: [] for name in TEST_NAMES}
    ages = []
    attempt_to_person_map = []

    sorted_keys = sorted(grouped_by_attempt.keys())

    for key in sorted_keys:
        attempt = grouped_by_attempt[key]
        records = attempt["records"]
        age = attempt["age"]
        person_id = attempt["person_id"]

        # Для каждой попытки обрабатываем все интересующие тесты
        for test_name in TEST_NAMES:
            # Ищем запись с нужным test_name
            record = next((rec for rec in records if rec["test_name"] == test_name), None)

            if record is None:
                # Тест не был пройден в этой попытке
                seq = None
            else:
                # Обрабатываем данные
                if test_name == "TStroop":
                    seq = prepare_TStroop(record, TEST_CONFIG[test_name]["col"], TStroop_way=STROOP_PROCESSING_WAY)
                else:
                    seq = prepare_test(record, TEST_CONFIG[test_name]["col"])

            all_tests_data[test_name].append(seq)

        # Сохраняем возраст и ID участника (один раз на попытку)
        ages.append(age)
        attempt_to_person_map.append(person_id)

    #  Шаг 3: Сохранение данных по каждому тесту
    FILLING_VALUE = -1.0  # Для замены NaN/None в данных
    print("Сохранение данных...")
    for test_name in TEST_NAMES:
        X_list = all_tests_data[test_name]

        # Собираем все последовательности в единый 3D-массив
        X = np.stack(X_list)  # Форма: (batch_size, length, num_features)
        # Заменяем все NaN на FILLING_VALUE
        X[np.isnan(X)] = FILLING_VALUE

        output_path = os.path.join(OUTPUT_DIR, f"X_{test_name}.npy")
        print(f"Сохранение {test_name} в {output_path} с формой {X.shape}")
        np.save(output_path, X)

    # Шаг 4: Сохранение возрастов
    y = np.array(ages, dtype=float)

    output_path = os.path.join(OUTPUT_DIR, "y_aligned.npy")
    print(f"Сохранение выровненных возрастов в {output_path} с формой {y.shape}")
    np.save(output_path, y)

    # Шаг 5: Сохранение карты попыток -> участников
    # Это позволит правильно разделить данные по участникам, а не по попыткам
    np.save(os.path.join(OUTPUT_DIR, "attempt_to_person_map.npy"), np.array(attempt_to_person_map, dtype=object))
    # массив, в котором на каждой позиции — ID человека, которому принадлежит эта попытка

if __name__ == "__main__":

    run_preparation()

