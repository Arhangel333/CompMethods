import sys
import numpy as np

def read_system_from_stdin():
    """
    Чтение системы уравнений из стандартного ввода.
    Формат ввода: расширенная матрица (n строк по n+1 чисел)
    """
    lines = sys.stdin.read().strip().split('\n')
    
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        print("Ошибка: пустой входной файл")
        sys.exit(1)
    
    # Фильтруем пустые строки
    lines = [line.strip() for line in lines if line.strip()]
    
    # Определяем размерность по первой строке
    try:
        first_row = list(map(float, lines[0].split()))
    except ValueError:
        print("Ошибка: первая строка должна содержать числа")
        sys.exit(1)
    
    n = len(first_row) - 1  # количество неизвестных
    
    if n < 1:
        print("Ошибка: система должна содержать хотя бы одно уравнение")
        sys.exit(1)
    
    if len(lines) != n:
        print(f"Предупреждение: количество строк ({len(lines)}) не равно количеству неизвестных ({n})")
        print("Будут использованы первые", n, "строк")
        lines = lines[:n]
    
    # Чтение расширенной матрицы
    A = []
    b = []
    
    for i, line in enumerate(lines[:n]):
        numbers = line.split()
        if len(numbers) != n + 1:
            print(f"Ошибка в строке {i+1}: ожидается {n+1} чисел, получено {len(numbers)}")
            print("Строка:", line)
            sys.exit(1)
        
        try:
            row = list(map(float, numbers[:n]))
            free_term = float(numbers[n])
            A.append(row)
            b.append(free_term)
        except ValueError:
            print(f"Ошибка в строке {i+1}: некорректное число")
            sys.exit(1)
    
    return np.array(A), np.array(b), n

def extract_tridiagonal(A, n):
    """
    Извлечение трехдиагональной структуры из полной матрицы.
    Возвращает:
    a - нижняя диагональ (a[0] не используется)
    b - главная диагональ
    c - верхняя диагональ (c[n-1] не используется)
    """
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    non_zero_off_diag = []
    
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > 1e-10:  # не нулевой элемент
                if j == i - 1:  # нижняя диагональ
                    a[i] = A[i, j]
                elif j == i:  # главная диагональ
                    b[i] = A[i, j]
                elif j == i + 1:  # верхняя диагональ
                    c[i] = A[i, j]
                else:
                    # Запоминаем ненулевые элементы вне трех диагоналей
                    non_zero_off_diag.append((i+1, j+1, A[i, j]))
    
    # Проверка, что главная диагональ не содержит нулей
    zero_diag = []
    for i in range(n):
        if abs(b[i]) < 1e-10:
            zero_diag.append(i+1)
    
    if zero_diag:
        print(f"Ошибка: нулевые элементы на главной диагонали в позициях: {zero_diag}")
        print("Метод прогонки требует ненулевых диагональных элементов")
        sys.exit(1)
    
    # Если есть элементы вне трех диагоналей, выводим предупреждение
    if non_zero_off_diag:
        print("\nВНИМАНИЕ: Обнаружены ненулевые элементы вне трех диагоналей:")
        for i, j, val in non_zero_off_diag:
            print(f"  A[{i},{j}] = {val:.6f}")
        print("Эти элементы будут проигнорированы!")
        print("Метод прогонки будет применен только к трехдиагональной части матрицы.\n")
    
    return a, b, c

def solve_tridiagonal(a, b, c, d):
    """
    Решение трехдиагональной системы методом прогонки
    
    Параметры:
    a - нижняя диагональ (a[0] не используется, длина n)
    b - главная диагональ (длина n)
    c - верхняя диагональ (c[n-1] не используется, длина n)
    d - правая часть (длина n)
    
    Возвращает:
    x - вектор решения
    det - определитель матрицы
    """
    n = len(b)
    
    # Прямой ход - вычисление прогоночных коэффициентов
    P = np.zeros(n)
    Q = np.zeros(n)
    
    # Начальные коэффициенты
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    
    # Вычисляем знаменатели для определителя
    det_factors = [b[0]]
    
    # Основной цикл прямого хода
    for i in range(1, n-1):
        denominator = a[i] * P[i-1] + b[i]
        det_factors.append(denominator)
        
        if abs(denominator) < 1e-15:
            print(f"Ошибка: близкое к нулю значение знаменателя на шаге {i+1}")
            print("Матрица может быть вырожденной или плохо обусловленной")
            print(f"denominator = {denominator:.6e}")
            sys.exit(1)
        
        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i-1]) / denominator
    
    # Последний шаг прямого хода
    if n > 1:
        denominator = a[n-1] * P[n-2] + b[n-1]
        det_factors.append(denominator)
        
        if abs(denominator) < 1e-15:
            print("Ошибка: близкое к нулю значение знаменателя на последнем шаге")
            print("Матрица вырождена")
            sys.exit(1)
    
    # Вычисление определителя
    det = np.prod(det_factors)
    
    # Обратный ход
    x = np.zeros(n)
    
    if n == 1:
        x[0] = Q[0]
    else:
        x[n-1] = (d[n-1] - a[n-1] * Q[n-2]) / (a[n-1] * P[n-2] + b[n-1])
        
        for i in range(n-2, -1, -1):
            x[i] = P[i] * x[i+1] + Q[i]
    
    return x, det

def print_system(A, b, n):
    """Вывод системы уравнений"""
    print("ИСХОДНАЯ СИСТЕМА УРАВНЕНИЙ")
    print("-"*70)
    
    # Находим максимальную ширину для форматирования
    max_width = 0
    for i in range(n):
        for j in range(n):
            val = A[i, j]
            if abs(val) > 1e-10:
                width = len(f"{val:.3f}") + len(f"·x{j+1}")
                max_width = max(max_width, width)
    
    for i in range(n):
        equation = ""
        first = True
        for j in range(n):
            coeff = A[i, j]
            if abs(coeff) > 1e-10:
                if first:
                    equation += f"{coeff:8.3f}·x{j+1}"
                    first = False
                else:
                    sign = "+" if coeff > 0 else ""
                    equation += f" {sign}{coeff:7.3f}·x{j+1}"
            else:
                if first:
                    equation += " " * 10
        # Добавляем правую часть
        equation += f" = {b[i]:8.3f}"
        print(equation)
    print()

def print_tridiagonal(a, b, c, d, n):
    """Вывод трехдиагональной структуры"""
    print("\nТРЕХДИАГОНАЛЬНАЯ СТРУКТУРА:")
    print("-"*70)
    print("  i  │    a[i]    │    b[i]    │    c[i]    │    d[i]    ")
    print("-"*70)
    for i in range(n):
        a_str = f"{a[i]:10.6f}" if i > 0 else "     —     "
        c_str = f"{c[i]:10.6f}" if i < n-1 else "     —     "
        print(f"  {i+1:2d} │ {a_str:10} │ {b[i]:10.6f} │ {c_str:10} │ {d[i]:10.6f}")
    print()

def main():
    # Чтение системы из стандартного ввода
    A, b, n = read_system_from_stdin()
    
    # Вывод исходной системы
    print_system(A, b, n)
    
    # Извлечение трехдиагональной структуры
    a, b_diag, c = extract_tridiagonal(A, n)
    
    # Вывод трехдиагональной структуры
    print_tridiagonal(a, b_diag, c, b, n)
    
    # Решение методом прогонки
    print("="*70)
    print("="*70)
    print("РЕШЕНИЕ МЕТОДОМ ПРОГОНКИ:")
    print("="*70)
    print("="*70)
    
    x, det = solve_tridiagonal(a, b_diag, c, b)
    
    # Вывод решения
    print("\nНайденное решение:")
    for i, val in enumerate(x):
        print(f"  x{i+1} = {val:.10f}")
    
    print(f"\nОПРЕДЕЛИТЕЛЬ МАТРИЦЫ: {det:.10e}")
    
    # Проверка решения (полная матрица)
    print("\n" + "="*70)
    print("ПРОВЕРКА РЕШЕНИЯ (A·x - b)")
    print("="*70)
    
    residual = A @ x - b
    max_residual = np.max(np.abs(residual))
    
    for i in range(n):
        status = "✓" if abs(residual[i]) < 1e-8 else "⚠"
        print(f"{status} Ур. {i+1}: {residual[i]:.2e}")
    
    print(f"\nМаксимальная невязка: {max_residual:.2e}")
    
    if max_residual < 1e-8:
        print("\n✓ РЕШЕНИЕ ВЕРНО (невязка близка к нулю)")
    elif max_residual < 1e-4:
        print("\n⚠ РЕШЕНИЕ ПРИЕМЛЕМО (невязка мала)")
    else:
        print("\n✗ РЕШЕНИЕ МОЖЕТ БЫТЬ НЕТОЧНЫМ (большая невязка)")
        print("  Возможно, матрица не является трехдиагональной или плохо обусловлена")

if __name__ == "__main__":
    main()
