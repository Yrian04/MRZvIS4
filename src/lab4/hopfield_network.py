import numpy as np

class HopfieldNetwork:
    def __init__(self, beta=10.0, max_epoch=1000, tol=1e-5):
        """
        Args: 
            beta: Коэффициент усиления функции активации tanh(beta * x).
            max_iter: Максимальное количество полных итераций (эпох).
            tol: Порог сходимости для определения релаксации.
        """
        self.beta = beta
        self.max_epoch = max_epoch
        self.tol = tol
        self.W = None
        self.n_neurons = 0

    def fit(self, patterns):
        """
        Обучение сети методом проекций.
        Args:
            patterns: Массив образов формы (n_patterns, n_neurons). 
                      Образы должны быть 1 или -1.
        """
        patterns = np.array(patterns, dtype=float)
        self.n_neurons = patterns.shape[1]
        
        # Метод проекций: W = X * pinv(X)
        # pinv вычисляет псевдообратную матрицу Мура-Пенроуза
        X = patterns.T  # Образы в столбцах
        self.W = X @ np.linalg.pinv(X)
        
        # Обнуляем диагональ
        np.fill_diagonal(self.W, 0)

    def predict(self, input_pattern):
        """
        Восстановление образа (релаксация).
        Args:
            input_pattern: Входной зашумленный образ (1 или -1).
        Returns:
            (final_output, epoch): Восстановленный образ и 
                                   количество итераций.
        """
        state = np.array(input_pattern, dtype=float).copy()
        
        for epoch in range(self.max_epoch):
            old_state = state.copy()
            
            # Асинхронное обновление
            for i in range(self.n_neurons):
                # Вычисляем вход нейрона
                net_i = np.dot(self.W[i], state)
                # Обновляем состояние нейрона
                state[i] = np.tanh(self.beta * net_i)
            
            # Проверка на достижение состояния релаксации
            if np.linalg.norm(state - old_state) < self.tol:
                break
        
        # Бинаризуем выход
        final_output = np.where(state >= 0, 1, -1)
        
        return final_output, epoch
