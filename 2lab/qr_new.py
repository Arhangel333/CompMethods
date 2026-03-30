import numpy as np

# ==================== QR-РАЗЛОЖЕНИЕ МЕТОДОМ ХАУСХОЛДЕРА ====================
def householder_qr(A):
    """Возвращает Q и R для матрицы A методом Хаусхолдера."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)

        if norm_x == 0 or len(x) == 1:
            continue

        if x[0] >= 0:
            u = x + norm_x * np.eye(len(x))[:, 0]
        else:
            u = x - norm_x * np.eye(len(x))[:, 0]

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-12:
            v = u / u_norm
        else:
            v = u

        for j in range(k, n):
            dot = 2 * np.dot(v, R[k:, j])
            R[k:, j] -= dot * v

        for j in range(m):
            dot = 2 * np.dot(v, Q[k:, j])
            Q[k:, j] -= dot * v

    return Q.T, R


# ==================== ОПРЕДЕЛЕНИЕ ТИПА МАТРИЦЫ ====================
def has_complex_eigenvalues(A, tolerance=1e-10):
    """
    Определяет, есть ли у матрицы комплексные собственные значения.
    Для матриц 2x2 использует точную формулу дискриминанта.
    Для больших матриц использует QR-итерации для обнаружения блоков 2x2.
    """
    n = A.shape[0]

    # Точная формула для матрицы 2x2
    if n == 2:
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        trace = a + d
        det = a*d - b*c
        discriminant = trace**2 - 4*det
        return discriminant < -tolerance

    # Для матриц большего размера: делаем несколько QR-итераций
    A_test = A.copy()
    for _ in range(10):  # 10 итераций достаточно для выявления структуры
        Q, R = householder_qr(A_test)
        A_test = R @ Q

    # Ищем блоки 2x2 (признак комплексных корней)
    for i in range(n - 1):
        # Если поддиагональный элемент ненулевой — возможный блок 2x2
        if abs(A_test[i + 1, i]) > tolerance:
            a, b = A_test[i, i], A_test[i, i + 1]
            c, d = A_test[i + 1, i], A_test[i + 1, i + 1]
            trace = a + d
            det_block = a*d - b*c
            discriminant = trace**2 - 4*det_block
            if discriminant < -tolerance:
                return True

    return False


# ==================== QR-АЛГОРИТМ С АДАПТИВНЫМ КРИТЕРИЕМ ====================
def qr_algorithm_adaptive(A, epsilon=0.0001, max_iter=1000):
    """
    QR-алгоритм с автоматическим выбором критерия сходимости.
    - Для симметричных матриц: проверяем все внедиагональные элементы.
    - Для несимметричных: проверяем элементы ниже второй поддиагонали (|i-j| ≥ 2).
    """
    n = A.shape[0]
    A_k = A.copy().astype(float)

    print("=" * 70)
    print("QR-АЛГОРИТМ ДЛЯ ПОИСКА СОБСТВЕННЫХ ЗНАЧЕНИЙ")
    print("=" * 70)

    # Арифметическая проверка исходной матрицы
    trace_A = np.trace(A)
    det_A = np.linalg.det(A)
    print(f"\nАРИФМЕТИЧЕСКАЯ ПРОВЕРКА ИСХОДНОЙ МАТРИЦЫ:")
    print(f"След матрицы A: {trace_A:.6f}")
    print(f"Определитель матрицы A: {det_A:.6f}")

    # Определяем тип матрицы
    has_complex = has_complex_eigenvalues(A)
    print(f"\nТИП МАТРИЦЫ: {'НЕСИММЕТРИЧНАЯ (есть комплексные корни)' if has_complex else 'СИММЕТРИЧНАЯ (только вещественные корни)'}")
    print(f"Критерий сходимости: {'квазидиагональная форма (|i-j| ≥ 2)' if has_complex else 'диагональная форма (все внедиаг < ε)'}")

    for iteration in range(max_iter):
        Q, R = householder_qr(A_k)
        A_next = R @ Q

        if has_complex:
            # Для несимметричных: проверяем элементы где i - j >= 2 (вторая поддиагональ и ниже)
            max_below_second = 0
            for i in range(n):
                for j in range(i - 1):  # j <= i-2
                    max_below_second = max(max_below_second, abs(A_next[i, j]))

            if iteration % 10 == 0:
                print(f"Итерация {iteration:4d}, max ниже 2-й поддиаг = {max_below_second:.6f}")

            if max_below_second < epsilon:
                print(f"\nСОШЛОСЬ на итерации {iteration} (квазидиагональная форма)")
                final_matrix = A_next
                break
        else:
            # Для симметричных: проверяем все внедиагональные элементы
            off_diag = A_next.copy()
            np.fill_diagonal(off_diag, 0)
            max_off_diag = np.max(np.abs(off_diag))

            if iteration % 10 == 0:
                print(f"Итерация {iteration:4d}, max внедиаг = {max_off_diag:.6f}")

            if max_off_diag < epsilon:
                print(f"\nСОШЛОСЬ на итерации {iteration} (диагональная форма)")
                final_matrix = A_next
                break

        A_k = A_next
    else:
        print(f"\nДостигнуто максимальное число итераций ({max_iter})")
        final_matrix = A_k

    return iteration, final_matrix, has_complex


# ==================== ИЗВЛЕЧЕНИЕ СОБСТВЕННЫХ ЗНАЧЕНИЙ ====================
def extract_eigenvalues_from_quasi_diagonal(A, epsilon=1e-10):
    """
    Извлекает собственные значения из квазидиагональной матрицы.
    Блоки 2x2 дают комплексно-сопряженные пары, блоки 1x1 — вещественные.
    """
    n = A.shape[0]
    eigenvalues = []
    i = 0

    print("\nИЗВЛЕЧЕНИЕ СОБСТВЕННЫХ ЗНАЧЕНИЙ ИЗ КВАЗИДИАГОНАЛЬНОЙ ФОРМЫ:")

    while i < n:
        if i < n - 1 and abs(A[i + 1, i]) > epsilon:
            # Блок 2x2
            a, b = A[i, i], A[i, i + 1]
            c, d = A[i + 1, i], A[i + 1, i + 1]

            print(f"\nБлок 2x2 в позиции ({i},{i}):")
            print(f"[{a:8.4f} {b:8.4f}]")
            print(f"[{c:8.4f} {d:8.4f}]")

            trace = a + d
            det_block = a * d - b * c
            discriminant = trace**2 - 4 * det_block

            if discriminant >= -epsilon and discriminant < 0:
                discriminant = 0

            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                lambda1 = (trace + sqrt_disc) / 2
                lambda2 = (trace - sqrt_disc) / 2
                eigenvalues.append(lambda1)
                eigenvalues.append(lambda2)
                print(f"  λ = {lambda1:.6f}, {lambda2:.6f}")
            else:
                real_part = trace / 2
                imag_part = np.sqrt(-discriminant) / 2
                lambda1 = complex(real_part, imag_part)
                lambda2 = complex(real_part, -imag_part)
                eigenvalues.append(lambda1)
                eigenvalues.append(lambda2)
                print(f"  λ = {lambda1:.6f}, {lambda2:.6f}")

            i += 2
        else:
            # Блок 1x1
            print(f"\nБлок 1x1 в позиции {i}: {A[i, i]:.6f}")
            eigenvalues.append(A[i, i])
            i += 1

    return np.array(eigenvalues)


# ==================== ПРОВЕРКА СОБСТВЕННЫХ ЗНАЧЕНИЙ ====================
def verify_eigenvalues(A, eigenvalues):
    """Проверяет найденные собственные значения через след и определитель."""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА СОБСТВЕННЫХ ЗНАЧЕНИЙ")
    print("=" * 70)

    trace_A = np.trace(A)
    sum_eigenvals = np.sum(eigenvalues)

    print(f"\n1. ПРОВЕРКА ЧЕРЕЗ СЛЕД МАТРИЦЫ:")
    print(f"   След матрицы A: {trace_A:.6f}")
    print(f"   Сумма собственных значений: {sum_eigenvals:.6f}")
    print(f"   Разница: {abs(trace_A - sum_eigenvals):.2e}")

    det_A = np.linalg.det(A)
    prod_eigenvals = np.prod(eigenvalues)

    print(f"\n2. ПРОВЕРКА ЧЕРЕЗ ОПРЕДЕЛИТЕЛЬ:")
    print(f"   Определитель матрицы A: {det_A:.6f}")
    print(f"   Произведение собственных значений: {prod_eigenvals:.6f}")
    print(f"   Разница: {abs(det_A - prod_eigenvals):.2e}")

    # Вывод комплексных пар
    complex_pairs = []
    real_vals = []
    for val in eigenvalues:
        if isinstance(val, complex) or np.iscomplexobj(val):
            if abs(val.imag) > 1e-10:
                complex_pairs.append(val)
            else:
                real_vals.append(val.real)
        else:
            real_vals.append(val)

    if complex_pairs:
        print(f"\n3. КОМПЛЕКСНЫЕ СОБСТВЕННЫЕ ЗНАЧЕНИЯ (сопряженные пары):")
        for k in range(0, len(complex_pairs), 2):
            if k + 1 < len(complex_pairs):
                print(f"   Пара {k//2 + 1}: {complex_pairs[k]:.6f} и {complex_pairs[k+1]:.6f}")

    return trace_A, sum_eigenvals, det_A, prod_eigenvals


def input_matrix():
    """Функция для ввода матрицы и вектора"""
    print("Введите систему уравнений построчно.")
    print("Каждое уравнение вводите как коэффициенты при x1, x2, x3,... и свободный член")
    print("Пример для 3x3: 4 0.24 -0.08 8")
    
    first_line = input("\nВведите первую строку: ").strip().split()
    n = len(first_line)
    
    A = np.zeros((n, n))
    
    for j in range(n):
        A[0][j] = float(first_line[j])
    
    print(f"\nРазмерность матрицы: {n}x{n}")
    print("Введите остальные строки:")
    
    for i in range(1, n):
        while True:
            line = input(f"Строка {i+1}: ").strip().split()
            if len(line) == n:
                break
            print(f"Ошибка! Нужно ввести {n} чисел")
        
        for j in range(n):
            A[i][j] = float(line[j])
    
    return A

# ==================== ОСНОВНАЯ ПРОГРАММА ====================
if __name__ == "__main__":    

    A = input_matrix()

    print("\nИсходная матрица A:")
    print(A)

    # Запуск QR-алгоритма
    iterations, final_matrix, has_complex = qr_algorithm_adaptive(A, epsilon=0.0001)

    print(f"\nКоличество итераций: {iterations}")
    print("\nФинальная матрица:")
    for row in final_matrix:
        print(" ".join(f"{val:10.6f}" for val in row))

    # Извлечение собственных значений
    if has_complex:
        eigenvalues = extract_eigenvalues_from_quasi_diagonal(final_matrix)
    else:
        eigenvalues = np.diag(final_matrix)
        print("\nСОБСТВЕННЫЕ ЗНАЧЕНИЯ (диагональ финальной матрицы):")
        for i, val in enumerate(eigenvalues):
            print(f"λ{i+1} = {val:.6f}")

    print("\n" + "=" * 70)
    print("НАЙДЕННЫЕ СОБСТВЕННЫЕ ЗНАЧЕНИЯ:")
    print("=" * 70)
    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1} = {val:.6f}")

    # Проверка через след и определитель
    verify_eigenvalues(A, eigenvalues)

    # Сравнение с numpy (для проверки)
    numpy_vals = np.linalg.eigvals(A)
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ С NUMPY (ДЛЯ ПРОВЕРКИ):")
    print("=" * 70)
    print("Собственные значения от numpy:")
    for i, val in enumerate(sorted(numpy_vals, key=lambda x: x.real)):
        print(f"λ{i+1} = {val:.6f}")