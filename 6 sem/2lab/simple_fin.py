import numpy as np

def reorder_for_diagonal_dominance(A, B):
    """
    Переставляет уравнения (строки) для достижения диагонального преобладания
    Возвращает переупорядоченные матрицу A и вектор B
    """
    n = len(B)
    # Создаем список неиспользованных индексов строк
    available_rows = list(range(n))
    new_order = []
    
    print("\n=== ПЕРЕСТАНОВКА УРАВНЕНИЙ ===")
    
    # Для каждой позиции на диагонали ищем подходящее уравнение
    for col in range(n):  # col - номер диагонального элемента (какую переменную ищем)
        best_row = -1
        best_ratio = -1
        
        for row in available_rows:
            # Проверяем, может ли это уравнение дать диагональный элемент для col
            diag = abs(A[row][col])
            if diag == 0:
                continue
                
            # Сумма остальных коэффициентов в строке
            others = sum(abs(A[row][j]) for j in range(n) if j != col)
            
            # Проверяем условие преобладания
            if diag > others:
                # Нашли идеальную строку - берем её сразу
                best_row = row
                print(f"✓ Строка {row+1} подходит для x{col+1}: |{A[row][col]:.3f}| > {others:.3f}")
                break
            else:
                # Сохраняем лучшую по соотношению diag/others
                ratio = diag / max(others, 0.001)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_row = row
        
        if best_row == -1:
            print(f"❌ Не удалось найти подходящее уравнение для x{col+1}")
            return None, None
            
        # Добавляем выбранную строку в новый порядок
        new_order.append(best_row)
        available_rows.remove(best_row)
    
    # Создаем переупорядоченные матрицы
    A_new = A[new_order]
    B_new = B[new_order]
    
    print("\n✅ Новый порядок уравнений:", [i+1 for i in new_order])
    return A_new, B_new

def check_diagonal_dominance(A):
    """Проверяет диагональное преобладание"""
    n = len(A)
    dominant = True
    
    for i in range(n):
        diag = abs(A[i][i])
        others = sum(abs(A[i][j]) for j in range(n) if j != i)
        
        if diag <= others:
            print(f"✗ Строка {i+1}: |{A[i][i]:.3f}| ≤ {others:.3f} - нет преобладания")
            dominant = False
        else:
            print(f"✓ Строка {i+1}: |{A[i][i]:.3f}| > {others:.3f} - преобладание есть")
    
    return dominant

def solve_by_iterations(A, B, epsilon=0.0001, max_iterations=1000, auto_reorder=True):
    """
    Решение СЛАУ методом простых итераций с возможностью перестановки
    """
    n = len(B)
    original_A = A.copy()
    original_B = B.copy()
    
    # Шаг 1: Проверяем диагональное преобладание
    print("\n=== ПРОВЕРКА ДИАГОНАЛЬНОГО ПРЕОБЛАДАНИЯ ===")
    has_dominance = check_diagonal_dominance(A)
    
    # Шаг 2: Если нет преобладания и разрешена перестановка - пробуем переставить
    if not has_dominance and auto_reorder:
        print("\n⚠️  Диагонального преобладания нет. Пробуем переставить уравнения...")
        A_reordered, B_reordered = reorder_for_diagonal_dominance(A, B)
        
        if A_reordered is not None:
            print("\n=== ПРОВЕРКА ПОСЛЕ ПЕРЕСТАНОВКИ ===")
            if check_diagonal_dominance(A_reordered):
                print("✅ Перестановка помогла! Продолжаем с новой системой.")
                A, B = A_reordered, B_reordered
            else:
                print("⚠️  Перестановка не дала полного преобладания, но пробуем решить...")
                A, B = A_reordered, B_reordered
        else:
            print("❌ Не удалось достичь диагонального преобладания перестановкой.")
            answer = input("Продолжить с исходной системой? (да/нет): ")
            if answer.lower() != 'да':
                return None, 0
    elif not has_dominance and not auto_reorder:
        print("\n⚠️  Диагонального преобладания нет! Метод может не сойтись.")
        answer = input("Продолжить? (да/нет): ")
        if answer.lower() != 'да':
            return None, 0
    
    # Шаг 3: Приведение к итерационному виду
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
    
    # Шаг 4: Проверка условия сходимости
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
    
    # Шаг 5: Итерационный процесс
    X_old = D.copy()
    print(f"\n=== НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ ===")
    print("X(0) =", np.array2string(X_old, precision=4, suppress_small=True))
    
    print("\n=== ИТЕРАЦИИ ===")
    print("-" * 70)
    header = f"{'k':<3} | " + " | ".join([f"x{j+1}:<10" for j in range(min(n, 3))]) + " | Δ max"
    if n > 3:
        header = f"{'k':<3} | x₁:<10 | x₂:<10 | x₃:<10 | ... | Δ max"
    print(header)
    print("-" * 70)
    
    for iteration in range(max_iterations):
        X_new = np.zeros(n)
        for i in range(n):
            X_new[i] = sum(C[i][j] * X_old[j] for j in range(n)) + D[i]
        
        delta = max(abs(X_new[i] - X_old[i]) for i in range(n))
        
        # Выводим первые 3 значения (или все, если их мало)
        if n <= 3:
            values = [f"{X_new[j]:<10.6f}" for j in range(n)]
        else:
            values = [f"{X_new[0]:<10.6f}", f"{X_new[1]:<10.6f}", f"{X_new[2]:<10.6f}", "..."]
        
        print(f"{iteration+1:<3} | " + " | ".join(values) + f" | {delta:<10.6f}")
        
        if delta < epsilon:
            print("-" * 70)
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
    
    first_line = input("\nВведите первую строку (коэф. и свободный член): ").strip().split()
    n = len(first_line) - 1
    
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    for j in range(n):
        A[0][j] = float(first_line[j])
    B[0] = float(first_line[n])
    
    print(f"\nРазмерность системы: {n}x{n}")
    print("Введите остальные строки:")
    
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

def main():
    print("=" * 70)
    print("="*70)
    print("МЕТОД ПРОСТЫХ ИТЕРАЦИЙ С АВТОМАТИЧЕСКОЙ ПЕРЕСТАНОВКОЙ")
    print("=" * 70)
    print("="*70)
    
    while True:
        try:
            A, B = input_matrix()
            
            print("\n=== ИСХОДНАЯ СИСТЕМА ===")
            for i in range(len(B)):
                row = [f"{A[i][j]:.3f}*x{j+1}" for j in range(len(B))]
                print(f"{' + '.join(row)} = {B[i]:.3f}")
            
            epsilon = 0.0001
            print(f"\nТочность вычислений: ε = {epsilon}")
            
            # Решаем с автоматической перестановкой
            solution, iterations = solve_by_iterations(A, B, epsilon, auto_reorder=True)
            
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
