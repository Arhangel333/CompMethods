import numpy as np
import sys

def check_diagonal_dominance(A):
    """
    Проверка диагонального преобладания матрицы
    """
    n = len(A)
    violations = []
    
    for i in range(n):
        diagonal = abs(A[i][i])
        sum_row = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= sum_row:
            violations.append((i, diagonal, sum_row))
    
    return len(violations) == 0, violations

def reorder_rows_for_dominance(A, b):
    """
    ПЕРЕСТАНОВКА СТРОК для достижения диагонального преобладания
    Жадный алгоритм: на каждую позицию ставим строку с максимальным диагональным элементом
    """
    n = len(A)
    # Создаем список индексов строк
    rows = list(range(n))
    A_new = A.copy()
    b_new = b.copy()
    
    print("\n🔄 Пытаемся переставить строки для улучшения сходимости...")
    
    # Для каждой позиции i ищем строку с максимальным |A[i][i]|
    for i in range(n):
        max_row = i
        max_val = abs(A_new[rows[i]][i])
        
        for j in range(i + 1, n):
            curr_val = abs(A_new[rows[j]][i])
            if curr_val > max_val:
                max_val = curr_val
                max_row = j
        
        if max_row != i:
            # Меняем местами строки
            rows[i], rows[max_row] = rows[max_row], rows[i]
            print(f"  Меняем строку {i+1} со строкой {max_row+1}")
    
    # Перестраиваем матрицу и вектор
    A_reordered = A_new[rows]
    b_reordered = b_new[rows]
    
    # Проверяем, помогло ли
    has_dom, violations = check_diagonal_dominance(A_reordered)
    
    if has_dom:
        print("  ✓ После перестановки достигнуто диагональное преобладание!")
    else:
        print("  ✗ Перестановка не помогла достичь диагонального преобладания")
        print("  Нарушения остались в строках:")
        for row, diag, sum_row in violations:
            print(f"    Строка {row+1}: |{diag:.3f}| ≤ {sum_row:.3f}")
    
    return A_reordered, b_reordered, has_dom

def reorder_rows_and_columns(A, b):
    """
    ПЕРЕСТАНОВКА СТРОК И СТОЛБЦОВ (более мощный метод)
    Меняет местами и уравнения, и переменные
    """
    n = len(A)
    A_new = A.copy()
    b_new = b.copy()
    
    # Запоминаем соответствие переменных
    var_order = list(range(n))
    
    print("\n🔄 Пытаемся переставить строки И столбцы...")
    
    # Жадный алгоритм: ищем максимальный элемент в оставшейся подматрице
    for i in range(n):
        # Ищем максимальный элемент в подматрице [i:, i:]
        max_row, max_col = i, i
        max_val = abs(A_new[i][i])
        
        for r in range(i, n):
            for c in range(i, n):
                if abs(A_new[r][c]) > max_val:
                    max_val = abs(A_new[r][c])
                    max_row, max_col = r, c
        
        # Перестановка строк
        if max_row != i:
            A_new[[i, max_row]] = A_new[[max_row, i]]
            b_new[[i, max_row]] = b_new[[max_row, i]]
            print(f"  Меняем строку {i+1} со строкой {max_row+1}")
        
        # Перестановка столбцов
        if max_col != i:
            A_new[:, [i, max_col]] = A_new[:, [max_col, i]]
            var_order[i], var_order[max_col] = var_order[max_col], var_order[i]
            print(f"  Меняем переменную x{i+1} с x{max_col+1}")
    
    # Проверяем, помогло ли
    has_dom, violations = check_diagonal_dominance(A_new)
    
    return A_new, b_new, var_order, has_dom

def adaptive_sor_method(A, b, epsilon=1e-4, max_iterations=1000, 
                        omega_min=0.1, omega_max=1.5, omega_init=1.0,
                        adaptation_rate=0.2, verbose=False):
    """
    Адаптивный SOR метод с изменением ω на каждой итерации
    """
    n = len(A)
    
    # Проверка на нулевые диагональные элементы
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            print(f"Ошибка: Нулевой диагональный элемент в строке {i+1}")
            return None, 0, False, []
    
    x = np.zeros(n)
    omega = omega_init
    omega_history = [omega]
    
    prev_error = float('inf')
    
    if verbose:
        print(f"\n{' Итер ':^6} | {'ω':^8} | ", end="")
        for i in range(n):
            print(f"x{i+1}{' ':^13}", end=" | ")
        print(f"{'max|Δ|':^15}")
        print("-" * (35 + n * 20))
    
    for iteration in range(max_iterations):
        x_prev = x.copy()
        
        # SOR итерация
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x_gs = (b[i] - sum1 - sum2) / A[i][i]
            x[i] = omega * x_gs + (1 - omega) * x_prev[i]
            
            if abs(x[i]) > 1e100:
                return None, iteration + 1, False, omega_history
        
        current_error = max(abs(x - x_prev))
        
        # Адаптация ω
        if iteration > 0:
            # Если ошибка растет - уменьшаем ω
            if current_error > prev_error * 1.05:  # Рост > 5%
                omega = max(omega * 0.8, omega_min)
                action = "⚠ РОСТ ↓ω"
            # Если ошибка падает - можно немного увеличить ω
            elif current_error < prev_error * 0.95:  # Падение > 5%
                omega = min(omega * 1.05, omega_max)
                action = "✓ ПАДЕНИЕ ↑ω"
            else:
                action = "→ СТАБИЛЬНО"
        else:
            action = "СТАРТ"
        
        omega_history.append(omega)
        
        if verbose:
            print(f"{iteration+1:6d} | {omega:8.4f} | ", end="")
            for i in range(n):
                print(f"{x[i]:15.10f} | ", end="")
            print(f"{current_error:15.10f} | {action}")
        
        # Проверка сходимости
        if current_error < epsilon:
            if verbose:
                print("-" * (35 + n * 20))
                print(f"✓ СХОДИМОСТЬ на итерации {iteration+1}, ω = {omega:.4f}")
            return x, iteration + 1, True, omega_history
        
        prev_error = current_error
    
    return x, max_iterations, False, omega_history

def read_augmented_matrix_from_input():
    """
    Чтение расширенной матрицы из стандартного ввода
    """
    print("Введите расширенную матрицу системы:")
    print("Каждая строка: коэффициенты при переменных и последний элемент - правая часть")
    print("\nВведите первую строку (определит размерность системы):")
    
    first_line = list(map(float, input().split()))
    n = len(first_line) - 1
    
    print(f"Определен размер системы: {n} уравнений, {n} неизвестных")
    print(f"Введите остальные {n-1} строк:")
    
    A = []
    b = []
    
    A.append(first_line[:-1])
    b.append(first_line[-1])
    
    for i in range(n - 1):
        line = list(map(float, input().split()))
        if len(line) != n + 1:
            print(f"Ошибка: строка должна содержать {n + 1} элементов")
            return None, None
        A.append(line[:-1])
        b.append(line[-1])
    
    return np.array(A, dtype=float), np.array(b, dtype=float)

def read_augmented_matrix_from_file(filename):
    """
    Чтение расширенной матрицы из файла
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден")
        return None, None
    
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not lines:
        print("Ошибка: Файл пуст")
        return None, None
    
    first_line = list(map(float, lines[0].split()))
    n = len(first_line) - 1
    
    if n < 1:
        print("Ошибка: Некорректный размер системы")
        return None, None
    
    if len(lines) != n:
        print(f"Ошибка: Ожидалось {n} строк, получено {len(lines)}")
        return None, None
    
    A = []
    b = []
    
    for i, line in enumerate(lines):
        values = list(map(float, line.split()))
        if len(values) != n + 1:
            print(f"Ошибка в строке {i+1}: ожидалось {n + 1} элементов")
            return None, None
        A.append(values[:-1])
        b.append(values[-1])
    
    return np.array(A, dtype=float), np.array(b, dtype=float)

def print_augmented_matrix(A, b, title="РАСШИРЕННАЯ МАТРИЦА"):
    """
    Красивый вывод расширенной матрицы
    """
    print(f"\n{title}:")
    n = len(A)
    for i in range(n):
        row = ""
        for j in range(n):
            row += f"{A[i][j]:12.6f} "
        row += f"| {b[i]:12.6f}"
        print(row)

def solve_by_gauss(A, b, message="Метод Гаусса"):
    """
    Решение методом Гаусса через numpy.linalg.solve
    """
    try:
        x = np.linalg.solve(A, b)
        print(f"\n✓ {message} успешно решил систему:")
        for i, val in enumerate(x):
            print(f"  x{i+1} = {val:.10f}")
        return x, True
    except np.linalg.LinAlgError:
        print(f"\n✗ {message} не сработал (матрица вырождена)")
        return None, False

def main():
    """
    Основная функция программы с многоуровневой стратегией:
    1. Сначала пробуем SOR
    2. Если не работает - перестановка строк
    3. Если не помогает - перестановка строк и столбцов
    4. В крайнем случае - метод Гаусса
    """
    print("=" * 80)
    print("АДАПТИВНЫЙ SOR МЕТОД С ПЕРЕСТАНОВКОЙ СТРОК")
    print("=" * 80)
    
    # Чтение данных
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"\nЧтение из файла: {filename}")
        A, b = read_augmented_matrix_from_file(filename)
    else:
        print("\n" + "=" * 80)
        A, b = read_augmented_matrix_from_input()
    
    if A is None or b is None:
        print("Ошибка при чтении данных. Программа завершена.")
        return 1
    
    # Сохраняем исходную систему
    A_original = A.copy()
    b_original = b.copy()
    
    # Вывод исходной системы
    print_augmented_matrix(A, b, "ИСХОДНАЯ СИСТЕМА")
    
    # Параметры решения
    epsilon = 0.0001
    print(f"\nЗаданная точность: ε = {epsilon}")
    
    # -----------------------------------------------------------------
    # УРОВЕНЬ 1: Пробуем SOR на исходной матрице
    # -----------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("УРОВЕНЬ 1: Пробуем SOR на исходной матрице")
    print(f"{'=' * 80}")
    
    solution, iterations, converged, omega_history = adaptive_sor_method(
        A, b, 
        epsilon=epsilon,
        max_iterations=100,
        omega_min=0.1,
        omega_max=1.5,
        omega_init=0.5,  # Начинаем с маленького ω для безопасности
        adaptation_rate=0.2,
        verbose=True
    )
    
    if converged:
        print(f"\n✓ SOR МЕТОД УСПЕШНО СОШЕЛСЯ на исходной матрице!")
        print(f"  Итераций: {iterations}")
        print(f"  Итоговое ω: {omega_history[-1]:.4f}")
        
        print(f"\n{'=' * 80}")
        print("РЕЗУЛЬТАТ")
        print(f"{'=' * 80}")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.10f}")
        return 0
    
    print(f"\n✗ SOR не сошелся на исходной матрице за {iterations} итераций")
    
    # -----------------------------------------------------------------
    # УРОВЕНЬ 2: Перестановка строк
    # -----------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("УРОВЕНЬ 2: Переставляем строки и пробуем SOR снова")
    print(f"{'=' * 80}")
    
    A_reordered, b_reordered, dom_achieved = reorder_rows_for_dominance(A_original, b_original)
    print_augmented_matrix(A_reordered, b_reordered, "СИСТЕМА ПОСЛЕ ПЕРЕСТАНОВКИ СТРОК")
    
    print("\nЗапускаем SOR на переставленной матрице...")
    solution, iterations, converged, omega_history = adaptive_sor_method(
        A_reordered, b_reordered, 
        epsilon=epsilon,
        max_iterations=100,
        omega_min=0.1,
        omega_max=1.5,
        omega_init=0.5,
        adaptation_rate=0.2,
        verbose=True
    )
    
    if converged:
        print(f"\n✓ SOR МЕТОД УСПЕШНО СОШЕЛСЯ ПОСЛЕ ПЕРЕСТАНОВКИ СТРОК!")
        print(f"  Итераций: {iterations}")
        print(f"  Итоговое ω: {omega_history[-1]:.4f}")
        
        print(f"\n{'=' * 80}")
        print("РЕЗУЛЬТАТ")
        print(f"{'=' * 80}")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.10f}")
        return 0
    
    print(f"\n✗ SOR не сошелся даже после перестановки строк")
    
    # -----------------------------------------------------------------
    # УРОВЕНЬ 3: Перестановка строк И столбцов
    # -----------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("УРОВЕНЬ 3: Переставляем строки И столбцы")
    print(f"{'=' * 80}")
    
    A_reordered_cols, b_reordered_cols, var_order, dom_achieved = reorder_rows_and_columns(A_original, b_original)
    print_augmented_matrix(A_reordered_cols, b_reordered_cols, "СИСТЕМА ПОСЛЕ ПЕРЕСТАНОВКИ СТРОК И СТОЛБЦОВ")
    print(f"Порядок переменных: {[f'x{i+1}' for i in var_order]}")
    
    print("\nЗапускаем SOR на полностью переставленной матрице...")
    solution_reordered, iterations, converged, omega_history = adaptive_sor_method(
        A_reordered_cols, b_reordered_cols, 
        epsilon=epsilon,
        max_iterations=100,
        omega_min=0.1,
        omega_max=1.5,
        omega_init=0.5,
        adaptation_rate=0.2,
        verbose=True
    )
    
    if converged:
        print(f"\n✓ SOR МЕТОД УСПЕШНО СОШЕЛСЯ ПОСЛЕ ПЕРЕСТАНОВКИ СТРОК И СТОЛБЦОВ!")
        print(f"  Итераций: {iterations}")
        print(f"  Итоговое ω: {omega_history[-1]:.4f}")
        
        # Возвращаем переменные в исходный порядок
        solution = np.zeros_like(solution_reordered)
        for new_idx, old_idx in enumerate(var_order):
            solution[old_idx] = solution_reordered[new_idx]
        
        print(f"\n{'=' * 80}")
        print("РЕЗУЛЬТАТ (с восстановленным порядком переменных)")
        print(f"{'=' * 80}")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.10f}")
        return 0
    
    print(f"\n✗ SOR не сошелся даже после всех перестановок")
    
    # -----------------------------------------------------------------
    # УРОВЕНЬ 4: Метод Гаусса (последний шанс)
    # -----------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("УРОВЕНЬ 4: Используем метод Гаусса (прямой метод)")
    print(f"{'=' * 80}")
    
    solve_by_gauss(A_original, b_original, "Метод Гаусса на исходной матрице")
    
    return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)