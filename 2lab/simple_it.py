import numpy as np

def solve_by_iterations(A, B, epsilon=0.0001, max_iterations=1000):
    """
    Решение СЛАУ методом простых итераций
    
    Параметры:
    A - матрица коэффициентов
    B - вектор свободных членов
    epsilon - точность вычислений
    max_iterations - максимальное количество итераций
    
    Возвращает:
    X - решение
    iterations - количество выполненных итераций
    """
    
    n = len(B)  # размерность системы
    
    # Шаг 1: Проверка диагонального преобладания
    print("\n=== ПРОВЕРКА ДИАГОНАЛЬНОГО ПРЕОБЛАДАНИЯ ===")
    need_reorder = False
    for i in range(n):
        diagonal = abs(A[i][i])
        sum_without_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        
        print(f"Строка {i+1}: |{A[i][i]:.3f}| = {diagonal:.3f} > {sum_without_diag:.3f}? {diagonal > sum_without_diag}")
        
        if diagonal <= sum_without_diag:
            need_reorder = True
    
    if need_reorder:
        print("\n⚠️  Диагонального преобладания нет! Метод может не сойтись.")
        print("Рекомендуется переставить уравнения или использовать другой метод.")
        """ answer = input("Продолжить? (да/нет): ")
        if answer.lower() != 'да':
            return None, 0 """
    
    # Шаг 2: Приведение к итерационному виду X = C*X + D
    print("\n=== ПРИВЕДЕНИЕ К ИТЕРАЦИОННОМУ ВИДУ ===")
    C = np.zeros((n, n))
    D = np.zeros(n)
    
    for i in range(n):
        diag = A[i][i]
        if abs(diag) < 1e-10:
            raise ValueError(f"Диагональный элемент в строке {i+1} близок к нулю!")
        
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] / diag
        D[i] = B[i] / diag
        
        # Выводим полученное уравнение
        equation = f"x{i+1} = "
        terms = []
        for j in range(n):
            if i != j and abs(C[i][j]) > 1e-10:
                terms.append(f"{C[i][j]:.3f}*x{j+1}")
        if terms:
            equation += " + ".join(terms).replace("+ -", "- ")
        else:
            equation += "0"
        equation += f" + {D[i]:.3f}"
        print(equation)
    
    # Проверка условия сходимости
    print("\n=== ПРОВЕРКА УСЛОВИЯ СХОДИМОСТИ ===")
    norms = []
    for i in range(n):
        row_sum = sum(abs(C[i][j]) for j in range(n))
        norms.append(row_sum)
        print(f"Строка {i+1}: сумма |коэф.| = {row_sum:.3f} {'< 1' if row_sum < 1 else '>= 1'}")
    
    max_norm = max(norms)
    if max_norm >= 1:
        print(f"\n⚠️  Максимальная норма = {max_norm:.3f} >= 1. Сходимость не гарантирована!")
    else:
        print(f"\n✅ Максимальная норма = {max_norm:.3f} < 1. Метод должен сойтись.")
    
    # Шаг 3: Начальное приближение (берем D)
    X_old = D.copy()
    print(f"\n=== НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ ===")
    print("X(0) =", np.array2string(X_old, precision=4, suppress_small=True))
    
    # Шаг 4: Итерационный процесс
    print("\n=== ИТЕРАЦИИ ===")
    print("-" * 60)
    print(f"{'k':<3} | {'X1':<12} | {'X2':<12} | {'X3':<12} | {'Δ max':<12}")
    print("-" * 60)
    
    for iteration in range(max_iterations):
        # Вычисляем новые значения
        X_new = np.zeros(n)
        for i in range(n):
            X_new[i] = sum(C[i][j] * X_old[j] for j in range(n)) + D[i]
        
        # Вычисляем погрешность
        delta = max(abs(X_new[i] - X_old[i]) for i in range(n))
        
        # Выводим текущую итерацию
        print(f"{iteration+1:<3} | {X_new[0]:<12.6f} | {X_new[1]:<12.6f} | {X_new[2]:<12.6f} | {delta:<12.6f}")
        
        # Проверка условия остановки
        if delta < epsilon:
            print("-" * 60)
            print(f"\n✅ Достигнута требуемая точность ε = {epsilon}")
            print(f"Количество итераций: {iteration+1}")
            return X_new, iteration+1
        
        X_old = X_new.copy()
    
    print(f"\n⚠️  Достигнуто максимальное число итераций ({max_iterations})")
    return X_old, max_iterations

def input_matrix():
    """Функция для ввода матрицы и вектора"""
    print("Введите систему уравнений построчно.")
    print("Каждое уравнение вводите как коэффициенты при x1, x2, x3,... и свободный член")
    print("Пример для 3x3: 4 0.24 -0.08 8")
    
    # Ввод первой строки для определения размера
    first_line = input("\nВведите первую строку (коэф. и свободный член): ").strip().split()
    
    # Определяем размерность системы
    n = len(first_line) - 1
    
    # Инициализация матрицы и вектора
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Обработка первой строки
    for j in range(n):
        A[0][j] = float(first_line[j])
    B[0] = float(first_line[n])
    
    print(f"\nРазмерность системы: {n}x{n}")
    print("Введите остальные строки:")
    
    # Ввод остальных строк
    for i in range(1, n):
        while True:
            line = input(f"Строка {i+1}: ").strip().split()
            if len(line) == n + 1:
                break
            print(f"Ошибка! Нужно ввести {n+1} чисел (коэффициенты и свободный член)")
        
        for j in range(n):
            A[i][j] = float(line[j])
        B[i] = float(line[n])
    
    return A, B

# Основная программа
def main():
    print("=" * 60)
    print("МЕТОД ПРОСТЫХ ИТЕРАЦИЙ ДЛЯ РЕШЕНИЯ СЛАУ")
    print("=" * 60)
    
    while True:
        try:
            # Ввод данных
            A, B = input_matrix()
            
            print("\n=== ВВЕДЕННАЯ СИСТЕМА ===")
            for i in range(len(B)):
                row = [f"{A[i][j]:.3f}*x{j+1}" for j in range(len(B))]
                print(f"{' + '.join(row)} = {B[i]:.3f}")
            
            # Точность вычислений
            epsilon = 0.0001
            print(f"\nТочность вычислений: ε = {epsilon}")
            
            # Решение
            solution, iterations = solve_by_iterations(A, B, epsilon)
            
            if solution is not None:
                print("\n=== РЕШЕНИЕ ===")
                for i, val in enumerate(solution):
                    print(f"x{i+1} = {val:.6f}")
            
            break
                
        except Exception as e:
            print(f"Ошибка: {e}")
            print("Попробуйте снова...")
    
    print("\nПрограмма завершена.")

if __name__ == "__main__":
    main()
