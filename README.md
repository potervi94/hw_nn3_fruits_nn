# hw_nn3_fruits_nn

ДЗ: Створити нейромережу,  3-5 шари перший шар Flatten кількість нейронів у шарах має не збільшуватись використайте функції активації RELU або LeakyRELU

## Опис / Description

Цей проект містить реалізацію нейронної мережі для класифікації фруктів з наступними вимогами:
- 3-5 шарів
- Перший шар - Flatten
- Кількість нейронів не збільшується (зменшується: 256 → 128 → 64)
- Функції активації: RELU або LeakyRELU

This project contains a neural network implementation for fruits classification with the following requirements:
- 3-5 layers
- First layer is Flatten
- Number of neurons decreases (256 → 128 → 64)
- Activation functions: RELU or LeakyRELU

## Архітектура моделі / Model Architecture

### Версія з ReLU / ReLU Version
```
Layer 1: Flatten (input: 100x100x3)
Layer 2: Dense (256 neurons, ReLU activation)
Layer 3: Dense (128 neurons, ReLU activation)
Layer 4: Dense (64 neurons, ReLU activation)
Layer 5: Dense (5 neurons, Softmax activation)
```

### Версія з LeakyReLU / LeakyReLU Version
```
Layer 1: Flatten (input: 100x100x3)
Layer 2: Dense (256 neurons, LeakyReLU activation)
Layer 3: Dense (128 neurons, LeakyReLU activation)
Layer 4: Dense (64 neurons, LeakyReLU activation)
Layer 5: Dense (5 neurons, Softmax activation)
```

## Встановлення / Installation

```bash
pip install -r requirements.txt
```

## Використання / Usage

### Запуск моделі з ReLU / Run ReLU Model
```bash
python fruits_nn.py
```

### Запуск моделі з LeakyReLU / Run LeakyReLU Model
```bash
python fruits_nn_leaky_relu.py
```

## Файли проекту / Project Files

- `fruits_nn.py` - Основна реалізація з функцією активації ReLU
- `fruits_nn_leaky_relu.py` - Альтернативна реалізація з функцією активації LeakyReLU
- `requirements.txt` - Залежності проекту

## Вимоги / Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.23+
- Matplotlib 3.5+
