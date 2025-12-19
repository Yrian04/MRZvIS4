import json
import argparse
import sys
from pathlib import Path

from lab4.hopfield_network import HopfieldNetwork


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        sys.exit(1)


def get_image(image: list[int], width: int) -> str:
    rows = (image[i:i+width] for i in range(0, len(image), width))
    return '\n'.join(
        ''.join(
            '▓' if x == 1 else '░' 
            for x in row
        ) 
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser(
        description="Программа для работы с сетью Хопфильда"
    )
    
    # Пути к файлам
    parser.add_argument(
        'input_file', 
        type=Path, 
        help='Путь к JSON файлу с входным образом для распознавания'
    )
    parser.add_argument(
        '--train_file', 
        type=Path,
        default=Path('data/train.json'),
        help='Путь к JSON файлу с обучающими образами'
    )
    
    # Гиперпараметры
    parser.add_argument(
        '--beta', 
        type=float, 
        default=10.0, 
        help='Коэффициент усиления tanh (по умолчанию: 10.0)'
    )
    parser.add_argument(
        '--max_epoch', 
        type=int, 
        default=1000, 
        help='Макс. кол-во эпох (по умолчанию: 1000)'
    )
    parser.add_argument(
        '--tol', 
        type=float, 
        default=1e-5, 
        help='Порог сходимости (по умолчанию: 1e-5)'
    )
    parser.add_argument(
        '--image-width', 
        type=int, 
        default=6, 
        help='Ширина образа для вывода (по умолчанию: 6)'
    )

    args = parser.parse_args()

    # Загрузка данных
    train_patterns = load_json(args.train_file)
    input_image = load_json(args.input_file)

    # Инициализация и обучение
    model = HopfieldNetwork(
        beta=args.beta, 
        max_epoch=args.max_epoch, 
        tol=args.tol
    )
    model.fit(train_patterns)

    # Распознавание
    result, iterations = model.predict(input_image)

    # Вывод результатов
    print(f"Релаксация завершена за {iterations} итераций(ий).")
    print("Входной образ:")
    print(get_image(input_image, args.image_width))
    print("Результат:",)
    print(get_image(result, args.image_width))


if __name__ == "__main__":
    main()