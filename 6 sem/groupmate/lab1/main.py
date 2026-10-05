def lu_decomposition(A):
    """
    LU-разложение с частичным выбором главного элемента по столбцу.
    Возвращает (L, U, P), где:
        L - нижняя треугольная с единицами на диагонали,
        U - верхняя треугольная,
        P - список перестановок (для умножения справа: b_perm = [b[P[i]] for i in range(n)]).
    """
    n = len(A)
    # Создаем копии для L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[A[i][j] for j in range(n)] for i in range(n)]
    
    # Инициализируем диагональ L единицами
    for i in range(n):
        L[i][i] = 1.0
    
    # Вектор перестановок (изначально строки не переставлены)
    P = list(range(n))
    
    for k in range(n - 1):
        # Поиск главного элемента в столбце k (начиная с k-й строки)
        max_val = abs(U[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i
        
        if max_val < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Если главный элемент не на текущей строке, меняем строки местами
        if max_row != k:
            # Меняем строки в U
            U[k], U[max_row] = U[max_row], U[k]
            # Меняем строки в L (только для столбцов до k-1, так как дальше они нули)
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            # Обновляем вектор перестановок
            P[k], P[max_row] = P[max_row], P[k]
        
        # Исключение (заполнение L и преобразование U)
        for i in range(k + 1, n):
            # Множитель
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            # Вычитаем из строки i строку k, умноженную на factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
    
    return L, U, P

def solve_lu(L, U, P, b):
    """
    Решение системы LU x = P^T b? На самом деле: PA = LU, значит A = P^T L U.
    Исходная система: A x = b  =>  P A x = P b  =>  L U x = P b.
    Решаем L y = P b, затем U x = y.
    """
    n = len(b)
    # Применяем перестановки к правой части
    b_perm = [b[P[i]] for i in range(n)]
    
    # Прямая подстановка для L y = b_perm
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = b_perm[i] - s   # L[i][i] = 1, поэтому деление не нужно
    
    # Обратная подстановка для U x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    
    return x

def determinant(L, U, P):
    """
    Определитель исходной матрицы A.
    det(A) = det(P^T) * det(L) * det(U) = (-1)^s * prod(diag(U)),
    где s - число перестановок.
    Число перестановок можно восстановить по вектору P.
    """
    # Определяем чётность перестановки P
    # Простейший способ: подсчитать количество инверсий
    n = len(P)
    visited = [False] * n
    swaps = 0
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len > 1:
                swaps += (cycle_len - 1)
    sign = -1 if swaps % 2 else 1
    
    det = sign
    for i in range(n):
        det *= U[i][i]
    return det

def inverse_matrix(A):
    """
    Построение обратной матрицы A^{-1} через LU-разложение с частичным выбором.
    """
    n = len(A)
    L, U, P = lu_decomposition(A)
    
    # Единичная матрица
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Для каждого столбца j решаем систему A x = e_j
    inv_cols = []
    for j in range(n):
        e = [I[i][j] for i in range(n)]
        x = solve_lu(L, U, P, e)
        inv_cols.append(x)  # пока это столбец
    
    # Преобразуем список столбцов в матрицу (по строкам)
    inv_matrix = [[inv_cols[j][i] for j in range(n)] for i in range(n)]
    return inv_matrix



if __name__ == "__main__":
    # Исходная матрица A и вектор b (можно взять ту же, что и раньше)
    A = [
        [-6, -5, -4, 7, 1],
        [4, 3, 5, 9, 2],
        [-2, 3, 5, 2, 4],
        [2, 5, -4, 1, 3],
        [1, 3, -5, 1, -2]
    ]
    b = [50, 53, -44, 46, 38]
    
    print("Матрица A:")
    for row in A:
        print(row)
    print("\nВектор b:", b)
    
    # LU-разложение
    L, U, P = lu_decomposition(A)
    print("\nМатрица L (нижняя треугольная с единицами на диагонали):")
    for row in L:
        print([round(x, 6) for x in row])
    print("\nМатрица U (верхняя треугольная):")
    for row in U:
        print([round(x, 6) for x in row])
    print("\nВектор перестановок P (исходная строка i стала новой строкой P[i]):", P)
    
    # Проверка: PA = LU
    n = len(A)
    PA = [[A[P[i]][j] for j in range(n)] for i in range(n)]
    LU = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += L[i][k] * U[k][j]
            LU[i][j] = s
    
    print("\nПроизведение L * U (должно совпадать с PA):")
    for row in LU:
        print([round(x, 6) for x in row])
    print("\nPA (переставленная A):")
    for row in PA:
        print([round(x, 6) for x in row])
    
    # Решение СЛАУ
    x = solve_lu(L, U, P, b)
    print("\nРешение системы Ax = b:", [round(v, 6) for v in x])
    
    # Определитель
    det = determinant(L, U, P)
    print("\nОпределитель матрицы A:", round(det, 6))
    
    # Обратная матрица
    A_inv = inverse_matrix(A)
    print("\nОбратная матрица A^{-1}:")
    for row in A_inv:
        print([round(x, 6) for x in row])
    
    # Проверка A * A^{-1} ≈ I
    I_calc = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * A_inv[k][j]
            I_calc[i][j] = s
    
    print("\nПроизведение A * A^{-1} (должно быть близко к единичной):")
    for row in I_calc:
        print([round(x, 6) for x in row])

