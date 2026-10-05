import numpy as np

# ============================================================================
# Задание 4. Метод прогонки для трёхдиагональной системы (вариант 11 из табл.2)
# Система задаётся матрицей A и вектором правой части b.
# ============================================================================

def solve_tridiagonal(a, b, c, d):
    """
    Решает систему с трёхдиагональной матрицей методом прогонки.
    a - поддиагональ (длина n-1, a[0] соответствует элементу (2,1))
    b - главная диагональ (длина n)
    c - наддиагональ (длина n-1, c[0] соответствует элементу (1,2))
    d - правая часть (длина n)
    Возвращает вектор x и определитель матрицы.
    """
    n = len(b)
    alpha = np.zeros(n-1)
    beta = np.zeros(n)
    # Прямая прогонка
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    for i in range(1, n-1):
        denominator = b[i] + a[i-1] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i-1] * beta[i-1]) / denominator
    # Последнее уравнение
    beta[n-1] = (d[n-1] - a[n-2] * beta[n-2]) / (b[n-1] + a[n-2] * alpha[n-2])
    # Обратная прогонка
    x = np.zeros(n)
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    # Вычисление определителя через рекуррентное соотношение
    det = np.zeros(n+1)
    det[0] = 1
    det[1] = b[0]
    for i in range(2, n+1):
        det[i] = b[i-1] * det[i-1] - a[i-2] * c[i-2] * det[i-2]
    return x, det[n]

# Задаём систему в виде матрицы A и вектора b (как в задании 5-6)
A = np.array([
    [5,  2,  0,  0,  0,  0,  0,  0],
    [3,  7, -2,  0,  0,  0,  0,  0],
    [0,  2, -4, -1,  0,  0,  0,  0],
    [0,  0, -2,  8, -3,  0,  0,  0],
    [0,  0,  0,  3,  7,  1,  0,  0],
    [0,  0,  0,  0, -5, 10,  4,  0],
    [0,  0,  0,  0,  0,  3, -6, -2],
    [0,  0,  0,  0,  0,  0,  3,  4]
], dtype=float)
b = np.array([6, -43, -21, -40, -28, 28, -3, -6], dtype=float)

print("Задание 4 (метод прогонки, вариант 11)")
print("Входная матрица A:")
print(A)
print("Вектор правой части b =", b)
print()

# Извлекаем диагонали
b_diag = np.diag(A)                # главная диагональ
c_diag = np.diag(A, k=1)           # наддиагональ
a_diag = np.diag(A, k=-1)          # поддиагональ

# Проверка, что матрица действительно трёхдиагональная
if not np.allclose(A, np.diag(b_diag) + np.diag(c_diag, 1) + np.diag(a_diag, -1)):
    print("Внимание: матрица не является трёхдиагональной! Результат может быть неверным.")

x, det = solve_tridiagonal(a_diag, b_diag, c_diag, b)
print("Решение x =", x)
print("Определитель матрицы (вычисленный прогонкой) =", det)
print("Определитель (numpy) =", np.linalg.det(A))
print()

# ============================================================================
# Задание 5 и 6. Методы простых итераций и Зейделя (вариант 11 из табл.3)
# ============================================================================

# Система из таблицы 3, вариант 11
A = np.array([
    [-5, 15, -2, 7],
    [16, 0, -8, 4],
    [4, -8, 3, 17],
    [7, -1, 15, 6]
], dtype=float)
b = np.array([75, 96, 34, -55], dtype=float)

print("Задание 5 и 6 (методы простых итераций и Зейделя, вариант 11)\n")
print("Входная матрица A:")
print(A)
print("\nВектор правой части b =", b)
print()

# Приведение к виду x = C x + d с диагональным преобладанием
# Выражаем переменные из уравнений с максимальным диагональным коэффициентом:
# x1 из 3-го (коэф. 16), x2 из 2-го (15), x3 из 1-го (15), x4 из 4-го (17)
C = np.zeros((4, 4))
d_vec = np.zeros(4)

# x1 = (96 + 8x3 - 4x4)/16
C[0, 2] = 8/16      # 0.5
C[0, 3] = -4/16     # -0.25
d_vec[0] = 96/16    # 6

# x2 = (75 + 5x1 + 2x3 - 7x4)/15
C[1, 0] = 5/15      # 1/3 ≈ 0.333333
C[1, 2] = 2/15      # ≈ 0.133333
C[1, 3] = -7/15     # ≈ -0.466667
d_vec[1] = 75/15    # 5

# x3 = (-55 -7x1 + x2 -6x4)/15
C[2, 0] = -7/15     # ≈ -0.466667
C[2, 1] = 1/15      # ≈ 0.066667
C[2, 3] = -6/15     # -0.4
d_vec[2] = -55/15   # ≈ -3.666667

# x4 = (34 -4x1 +8x2 -3x3)/17
C[3, 0] = -4/17     # ≈ -0.235294
C[3, 1] = 8/17      # ≈ 0.470588
C[3, 2] = -3/17     # ≈ -0.176471
d_vec[3] = 34/17    # 2

print("Матрица C (преобразованная для итераций):")
print(C)
print("\nВектор d =", d_vec)
print()

# Проверка сходимости: норма матрицы C должна быть меньше 1
normC = np.max(np.sum(np.abs(C), axis=1))
print("Норма матрицы C =", normC, "(должна быть < 1 для сходимости)\n")

def simple_iteration(C, d, eps=1e-4, max_iter=1000):
    x = np.zeros_like(d)
    for k in range(max_iter):
        x_new = C @ x + d
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k+1
        x = x_new
    return x, max_iter

def seidel(C, d, eps=1e-4, max_iter=1000):
    n = len(d)
    x = np.zeros(n)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += C[i, j] * x[j]
            x[i] = s + d[i]
        if np.max(np.abs(x - x_old)) < eps:
            return x, k+1
    return x, max_iter

eps = 1e-4
x_simple, iter_simple = simple_iteration(C, d_vec, eps)
x_seidel, iter_seidel = seidel(C, d_vec, eps)

print("Задание 5 (метод простых итераций):")
print("Решение:", x_simple)
print("Количество итераций:", iter_simple)
print()
print("Задание 6 (метод Зейделя):")
print("Решение:", x_seidel)
print("Количество итераций:", iter_seidel)
print()

# ============================================================================
# Задание 7.1. Собственные значения симметричной матрицы (вариант 11 из табл.4)
# ============================================================================

# Формируем симметричную матрицу 5x5 по данным таблицы 4 вариант 11
A_sym = np.array([
    [-3, -5, -4,  0, -3],
    [-5,  7,  1,  2,  2],
    [-4,  1, -1,  6,  5],
    [ 0,  2,  6,  1,  0],
    [-3,  2,  5,  0, -2]
], dtype=float)

print("Задание 7.1. Собственные значения симметричной матрицы (вариант 11)")
print("Входная матрица A (симметричная):")
print(A_sym)
print()

def jacobi_eigen(A, eps=1e-4, max_iter=1000):
    """
    Метод Якоби для нахождения всех собственных значений симметричной матрицы.
    Возвращает диагональную матрицу и количество итераций.
    """
    n = A.shape[0]
    V = np.eye(n)
    for k in range(max_iter):
        # Поиск максимального внедиагонального элемента
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        if max_val < eps:
            return np.diag(A), k+1, V
        # Вычисление угла поворота
        if A[p, p] == A[q, q]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        c = np.cos(theta)
        s = np.sin(theta)
        # Обновление матрицы A
        new_pp = c**2 * A[p, p] + s**2 * A[q, q] - 2 * c * s * A[p, q]
        new_qq = s**2 * A[p, p] + c**2 * A[q, q] + 2 * c * s * A[p, q]
        new_pq = (c**2 - s**2) * A[p, q] + c * s * (A[p, p] - A[q, q])
        # Сохраняем старые значения
        old_p = A[p, :].copy()
        old_q = A[q, :].copy()
        # Обновляем строки и столбцы p и q
        A[p, p] = new_pp
        A[q, q] = new_qq
        A[p, q] = 0.0
        A[q, p] = 0.0
        for i in range(n):
            if i != p and i != q:
                A[i, p] = c * old_p[i] - s * old_q[i]
                A[p, i] = A[i, p]
                A[i, q] = s * old_p[i] + c * old_q[i]
                A[q, i] = A[i, q]
        # Обновляем матрицу собственных векторов (не обязательно для ответа)
        for i in range(n):
            V[i, p], V[i, q] = c * V[i, p] - s * V[i, q], s * V[i, p] + c * V[i, q]
    return np.diag(A), max_iter, V

def qr_algorithm(A, eps=1e-4, max_iter=1000):
    """
    QR-алгоритм для нахождения собственных значений.
    Возвращает диагональную или квазидиагональную матрицу и число итераций.
    """
    n = A.shape[0]
    T = A.copy()
    for k in range(max_iter):
        # Проверка на сходимость (все поддиагональные элементы близки к нулю)
        if np.all(np.abs(np.tril(T, -1)) < eps):
            return T, k+1
        Q, R = np.linalg.qr(T)
        T = R @ Q
    return T, max_iter

eps_eig = 1e-4
# Метод Якоби
eig_jacobi, iter_jacobi, V = jacobi_eigen(A_sym.copy(), eps_eig)
print("Метод вращения Якоби:")
print("Собственные значения:", eig_jacobi)
print("Количество итераций:", iter_jacobi)
print(V)

A_sym = np.array([
    [-3, -5, -4,  0, -3],
    [11,  7,  1,  2,  2],
    [-2,  0, -1,  6,  5],
    [ 3,  5,  -4,  1,  0],
    [4,  3,  -5,  3, -2]
], dtype=float)

print("Задание 7.1. Собственные значения симметричной матрицы (вариант 11)")
print("Входная матрица A:")
print(A_sym)
print()

# QR-алгоритм
T_qr, iter_qr = qr_algorithm(A_sym.copy(), eps_eig)
print("\nQR-алгоритм:")
print("Финальная (квази)диагональная матрица:\n", T_qr)
print("Собственные значения (диагональные элементы):", np.diag(T_qr))
print("Количество итераций:", iter_qr)

# Для проверки
print("\nДля проверки (numpy.linalg.eigvalsh):", np.linalg.eigvalsh(A_sym))

print("\n" + "="*50)
print("АРИФМЕТИЧЕСКАЯ ПРОВЕРКА")
print("="*50)

# След матрицы
trace_A = np.trace(A_sym)
sum_eigenvals = np.sum(np.diag(T_qr))
print(f"След матрицы A: {trace_A:.6f}")
print(f"Сумма собственных значений: {sum_eigenvals}")
print(f"Разница (модуль): {abs(trace_A - sum_eigenvals):.2e}")

""" # Определитель и произведение
det_A = np.linalg.det(A_sym)
prod_eigenvals = np.prod(np.diag(T_qr))
print(f"\nОпределитель матрицы A: {det_A:.6f}")
print(f"Произведение собственных значений: {prod_eigenvals}")
print(f"Разница (модуль): {abs(det_A - prod_eigenvals):.2e}") """