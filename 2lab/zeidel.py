import numpy as np

def is_diagonally_dominant(A):
    """
    Проверка диагонального преобладания матрицы
    """
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i][i])
        sum_row = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= sum_row:
            return False
    return True

def seidel_method(A, b, epsilon=1e-4, max_iterations=1000):
    """
    Решение СЛАУ методом Зейделя
    
    Параметры:
    A - матрица коэффициентов
    b - вектор правых частей
    epsilon - точность вычислений
    max_iterations - максимальное количество итераций
    
    Возвращает:
    x - вектор решения
    iterations - количество выполненных итераций
    """
    n = len(A)
    
    # Проверка диагонального преобладания
    if not is_diagonally_dominant(A):
        print("Предупреждение: Матрица не имеет диагонального преобладания.")
        print("Метод Зейделя может не сходиться или сходиться медленно.")
    
    # Начальное приближение (нулевой вектор)
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    for iteration in range(max_iterations):
        # Копируем текущее приближение для проверки сходимости
        x_prev = x.copy()
        
        # Одна итерация метода Зейделя
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))  # уже обновленные
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))  # еще старые
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Копируем новые значения в x для следующей итерации
        x = x_new.copy()
        
        # Проверка условия сходимости
        if max(abs(x - x_prev)) < epsilon:
            return x, iteration + 1
    
    print(f"Достигнуто максимальное количество итераций ({max_iterations})")
    return x, max_iterations

def read_augmented_matrix_from_input():
    """
    Чтение расширенной матрицы из стандартного ввода
    
    Формат ввода:
    Каждая строка содержит коэффициенты при переменных И последний элемент - правую часть
    Количество элементов в первой строке определяет размерность + 1
    """
    print("Введите расширенную матрицу системы (каждая строка - уравнение):")
    print("Последний элемент каждой строки - правая часть уравнения")
    print("\nВведите первую строку (коэффициенты и правую часть через пробел):")
    
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
    
    return np.array(A), np.array(b)

def print_augmented_matrix(A, b):
    """
    Красивый вывод расширенной матрицы
    """
    print("\nРасширенная матрица системы:")
    n = len(A)
    for i in range(n):
        row = ""
        for j in range(n):
            row += f"{A[i][j]:8.3f} "
        row += f"| {b[i]:8.3f}"
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
    print("РЕШЕНИЕ СЛАУ МЕТОДОМ ЗЕЙДЕЛЯ")
    print("(Вектор правых частей - последний элемент каждой строки)")
    print("=" * 70)
    
    # Чтение расширенной матрицы из ввода
    A, b = read_augmented_matrix_from_input()
    
    if A is None or b is None:
        print("Ошибка при чтении данных. Программа завершена.")
        return
    
    # Вывод системы
    print_augmented_matrix(A, b)
    
    # Задание точности
    epsilon = 0.0001
    print(f"\nЗаданная точность: ε = {epsilon}")
    
    # Решение системы
    print("\n" + "=" * 70)
    print("РЕШЕНИЕ")
    print("=" * 70)
    
    try:
        solution, iterations = seidel_method(A, b, epsilon)
        
        print(f"\nКоличество итераций: {iterations}")
        print("\nПолученное решение:")
        for i, val in enumerate(solution):
            print(f"x{i+1} = {val:.10f}")
        
        # Проверка решения
        print("\n" + "=" * 70)
        print("ПРОВЕРКА")
        print("=" * 70)
        print("Подстановка в исходную систему:")
        
        max_error = 0
        for i in range(len(A)):
            result = sum(A[i][j] * solution[j] for j in range(len(A)))
            error = abs(result - b[i])
            max_error = max(max_error, error)
            print(f"Уравнение {i+1}: {result:.10f} = {b[i]:.3f} (разница: {error:.10f})")
        
        print(f"\nМаксимальная невязка: {max_error:.10f}")
        
        # Дополнительная информация о сходимости
        print("\n" + "=" * 70)
        print("ИНФОРМАЦИЯ О СХОДИМОСТИ")
        print("=" * 70)
        if is_diagonally_dominant(A):
            print("✓ Матрица имеет диагональное преобладание - метод должен сходиться")
        else:
            print("⚠ Матрица НЕ имеет диагонального преобладания")
            print("  Сходимость не гарантирована")
        
    except np.linalg.LinAlgError as e:
        print(f"Ошибка при решении системы: {e}")
    except ZeroDivisionError:
        print("Ошибка: Обнаружен нулевой диагональный элемент")
        print("Метод Зейделя требует ненулевые диагональные элементы")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")

if __name__ == "__main__":
    main()
