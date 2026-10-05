import numpy as np

# ============================================================================
# Задание 7.1. Собственные значения симметричной матрицы (вариант 11)
# ============================================================================

eps_eig = 1e-4   # <--- добавлено определение переменной

# Симметричная матрица 5x5 из таблицы 4, вариант 11
A_sym = np.array([
    [2, -5, -4, 6, 1],
    [4, 3, 1, -5, 2],
    [-2, 3, 4, 2, -5],
    [ 2, 0, -4, 1, 3],
    [1, 3, -5, -7, -2]
], dtype=float)

print("Задание 7.1. Собственные значения симметричной матрицы (вариант 11)")
print("Входная матрица A (симметричная):")
print(A_sym)
print()

# ---------- 1. Метод вращения Якоби ----------
def jacobi_eigen(A, eps=1e-4, max_iter=1000):
    n = A.shape[0]
    for k in range(max_iter):
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        if max_val < eps:
            return np.diag(A), k+1
        if A[p, p] == A[q, q]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        c = np.cos(theta)
        s = np.sin(theta)
        new_pp = c**2 * A[p, p] + s**2 * A[q, q] - 2 * c * s * A[p, q]
        new_qq = s**2 * A[p, p] + c**2 * A[q, q] + 2 * c * s * A[p, q]
        old_p = A[p, :].copy()
        old_q = A[q, :].copy()
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
    return np.diag(A), max_iter

print("Метод вращения Якоби:")
eig_jacobi, iter_jacobi = jacobi_eigen(A_sym.copy(), eps_eig)
print("Финальная диагональная матрица (собственные значения):")
print(np.diag(eig_jacobi))
print("Собственные значения:", eig_jacobi)
print("Количество итераций:", iter_jacobi)
print()

# ---------- 2. QR-алгоритм с остановкой по поддиагонали и изменению диагонали ----------
def qr_algorithm(A, eps=1e-4, max_iter=1000):
    n = A.shape[0]
    T = A.copy()
    prev_diag = np.diag(T)
    for k in range(max_iter):
        off_diag = np.abs(np.tril(T, -1))
        max_off = np.max(off_diag) if off_diag.size > 0 else 0.0
        curr_diag = np.diag(T)
        diag_change = np.max(np.abs(curr_diag - prev_diag)) if k > 0 else np.inf
        if max_off < eps and diag_change < eps:
            return T, k+1
        Q, R = np.linalg.qr(T)
        T = R @ Q
        prev_diag = curr_diag
    return T, max_iter

print("QR-алгоритм:")
T_qr, iter_qr = qr_algorithm(A_sym.copy(), eps_eig)
print("Финальная (квази)диагональная матрица:")
print(T_qr)
print("Собственные значения (диагональные элементы):", np.diag(T_qr))
print("Количество итераций:", iter_qr)
print()

# Проверка с помощью numpy
eig_numpy = np.linalg.eigvalsh(A_sym)
print("Для проверки (numpy.linalg.eigvalsh):", eig_numpy)
print("Все собственные значения вещественны (симметричная матрица).")