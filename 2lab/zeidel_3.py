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

def adaptive_sor_method_fixed(A, b, epsilon=1e-4, max_iterations=1000, 
                              omega_min=0.1, omega_max=1.5, omega_init=1.0,
                              adaptation_rate=0.2, verbose=False):
    """
    Улучшенный адаптивный SOR метод
    """
    n = len(A)
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
            if abs(A[i][i]) < 1e-15:
                return None, iteration, False, omega_history
            
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x_gs = (b[i] - sum1 - sum2) / A[i][i]
            x[i] = omega * x_gs + (1 - omega) * x_prev[i]
        
        current_error = max(abs(x - x_prev))
        
        # КРИТИЧЕСКИ ВАЖНО: проверка на расходимость
        if current_error > 1e100:  # Переполнение
            print(f"\n⚠ ПЕРЕПОЛНЕНИЕ на итерации {iteration+1}")
            print("  Метод расходится! Уменьшаем ω и пробуем заново...")
            
            # Кардинальное уменьшение ω
            omega = max(omega * 0.5, omega_min)
            print(f"  Новое ω = {omega:.4f}")
            
            # Сбрасываем решение и начинаем заново
            x = np.zeros(n)
            prev_error = float('inf')
            continue
        
        # АДАПТАЦИЯ: специально для случая расходимости
        if iteration > 0:
            # Если ошибка РАСТЕТ - срочно уменьшаем ω!
            if current_error > prev_error * 1.1:  # Рост больше 10%
                omega = max(omega * 0.7, omega_min)  # Резко уменьшаем на 30%
                action = "⚠ РАСХОДИМОСТЬ! ω↓"
                
            # Если ошибка падает - можно немного увеличить ω
            elif current_error < prev_error * 0.9:  # Падение больше 10%
                omega = min(omega * 1.1, omega_max)  # Плавно увеличиваем
                action = "✓ СХОДИМОСТЬ! ω↑"
            else:
                action = "→ стабильно"
        else:
            action = "старт"
        
        omega_history.append(omega)
        
        if verbose:
            print(f"{iteration+1:6d} | {omega:8.4f} | ", end="")
            for i in range(n):
                print(f"{x[i]:15.10f} | ", end="")
            print(f"{current_error:15.10f} | {action}")
            if iteration > 0:
                print(f"    рост: {current_error/prev_error:.2f}x, тренд: {'📈' if current_error>prev_error else '📉'}")
        
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
    Каждая строка содержит коэффициенты и правую часть (последний элемент)
    """
    print("Введите расширенную матрицу системы:")
    print("Каждая строка: коэффициенты при переменных и последний элемент - правая часть")
    print("\nВведите первую строку (определит размерность системы):")
    
    # Читаем первую строку и определяем размерность
    first_line = list(map(float, input().split()))
    n = len(first_line) - 1  # размерность системы (без правой части)
    
    print(f"Определен размер системы: {n} уравнений, {n} неизвестных")
    print(f"Введите остальные {n-1} строк:")
    
    # Создаем матрицу A и вектор b
    A = []
    b = []
    
    # Добавляем первую строку
    A.append(first_line[:-1])
    b.append(first_line[-1])
    
    # Читаем остальные строки
    for i in range(n - 1):
        line = list(map(float, input().split()))
        if len(line) != n + 1:
            print(f"Ошибка: строка должна содержать {n + 1} элементов (коэффициенты + правая часть)")
            return None, None
        A.append(line[:-1])
        b.append(line[-1])
    
    return np.array(A, dtype=float), np.array(b, dtype=float)

def read_augmented_matrix_from_file(filename):
    """
    Чтение расширенной матрицы из файла
    Каждая строка содержит коэффициенты и правую часть (последний элемент)
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден")
        return None, None
    
    # Убираем пустые строки и комментарии
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not lines:
        print("Ошибка: Файл пуст")
        return None, None
    
    # Определяем размерность по первой строке
    first_line = list(map(float, lines[0].split()))
    n = len(first_line) - 1
    
    if n < 1:
        print("Ошибка: Некорректный размер системы")
        return None, None
    
    print(f"Определен размер системы: {n} уравнений, {n} неизвестных")
    
    # Проверяем количество строк
    if len(lines) != n:
        print(f"Ошибка: Ожидалось {n} строк, получено {len(lines)}")
        return None, None
    
    # Читаем все строки
    A = []
    b = []
    
    for i, line in enumerate(lines):
        values = list(map(float, line.split()))
        if len(values) != n + 1:
            print(f"Ошибка в строке {i+1}: ожидалось {n + 1} элементов, получено {len(values)}")
            return None, None
        A.append(values[:-1])
        b.append(values[-1])
    
    return np.array(A, dtype=float), np.array(b, dtype=float)

def print_augmented_matrix(A, b):
    """
    Красивый вывод расширенной матрицы
    """
    print("\nРасширенная матрица системы:")
    n = len(A)
    for i in range(n):
        row = ""
        for j in range(n):
            row += f"{A[i][j]:10.3f} "
        row += f"| {b[i]:10.3f}"
        print(row)
    
    print("\nСистема уравнений:")
    for i in range(n):
        equation = []
        for j in range(n):
            coeff = A[i][j]
            if j == 0:
                equation.append(f"{coeff:.3f}*x{j+1}")
            else:
                sign = "+" if coeff >= 0 else "-"
                equation.append(f"{sign} {abs(coeff):.3f}*x{j+1}")
        print(f"{' '.join(equation)} = {b[i]:.3f}")

def main():
    """
    Основная функция программы
    """
    print("=" * 70)
    print("АДАПТИВНЫЙ SOR МЕТОД (ω меняется на каждой итерации)")
    print("=" * 70)
    
    # Чтение данных
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"\nЧтение из файла: {filename}")
        A, b = read_augmented_matrix_from_file(filename)
    else:
        print("\n" + "=" * 70)
        A, b = read_augmented_matrix_from_input()
    
    if A is None or b is None:
        print("Ошибка при чтении данных. Программа завершена.")
        return 1
    
    # Вывод системы
    print("\n" + "=" * 70)
    print_augmented_matrix(A, b)
    
    # Проверка диагонального преобладания
    has_dominance, violations = check_diagonal_dominance(A)
    
    print(f"\n{'=' * 70}")
    print(f"{'АНАЛИЗ СХОДИМОСТИ':^70}")
    print(f"{'=' * 70}")
    
    if has_dominance:
        print("✓ Матрица имеет диагональное преобладание")
    else:
        print("⚠ Матрица НЕ имеет диагонального преобладания")
        print("  Нарушения в строках:")
        for row, diag, sum_row in violations:
            print(f"  Строка {row+1}: |{diag:.3f}| ≤ {sum_row:.3f}")
    
    # Параметры решения
    epsilon = 0.0001
    print(f"\nЗаданная точность: ε = {epsilon}")
    
    # Решение системы адаптивным SOR методом
    print(f"\n{'=' * 70}")
    print(f"{'РЕШЕНИЕ':^70}")
    print(f"{'=' * 70}")
    
    solution, iterations, converged, omega_history = adaptive_sor_method_fixed(
        A, b, 
        epsilon=epsilon,
        max_iterations=1000,
        omega_min=0.3,
        omega_max=1.9,
        omega_init=1.0,
        adaptation_rate=0.1,
        verbose=True
    )
    
    if not converged:
        print("\n⚠ Метод не сошелся за максимальное количество итераций")
        print("\nПробуем метод Гаусса как запасной вариант:")
        try:
            x_gauss = np.linalg.solve(A, b)
            print("✓ Метод Гаусса успешно решил систему:")
            for i, val in enumerate(x_gauss):
                print(f"  x{i+1} = {val:.10f}")
            return 0
        except np.linalg.LinAlgError:
            print("✗ Метод Гаусса не сработал (матрица вырождена)")
            return 1
    
    # Вывод результатов
    print(f"\n{'=' * 70}")
    print(f"{'РЕЗУЛЬТАТ':^70}")
    print(f"{'=' * 70}")
    
    print(f"\nКоличество итераций: {iterations}")
    print(f"Итоговое ω: {omega_history[-1]:.4f}")
    print(f"\nПолученное решение:")
    for i, val in enumerate(solution):
        print(f"  x{i+1} = {val:.10f}")
    
    # Проверка решения
    print(f"\n{'=' * 70}")
    print(f"{'ПРОВЕРКА':^70}")
    print(f"{'=' * 70}")
    
    max_error = 0
    for i in range(len(A)):
        result = sum(A[i][j] * solution[j] for j in range(len(A)))
        error = abs(result - b[i])
        max_error = max(max_error, error)
        print(f"Уравнение {i+1}: {result:.10f} = {b[i]:.3f}  (Δ = {error:.10f})")
    
    print(f"\nМаксимальная невязка: {max_error:.10f}")
    
    if max_error < epsilon:
        print("✓ Решение удовлетворяет заданной точности")
    else:
        print(f"⚠ Решение может быть неточным (невязка > {epsilon})")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)