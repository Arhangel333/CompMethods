import numpy as np

# --- Функция QR-разложения методом Хаусхолдера (из предыдущего объяснения) ---
def householder_qr(A):
    """Возвращает Q и R для матрицы A методом Хаусхолдера."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        # Вектор x = текущий столбец от k-й строки до конца
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        
        if norm_x == 0 or len(x) == 1:
            continue
            
        # Строим вектор отражения u
        # Чтобы избежать потери точности, используем знак первого элемента
        if x[0] >= 0:
            u = x + norm_x * np.eye(len(x))[:, 0]
        else:
            u = x - norm_x * np.eye(len(x))[:, 0]
            
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-12:
            v = u / u_norm
        else:
            v = u
        
        # Обновляем R: R = (I - 2*v*v^T) @ R
        # Но эффективнее делать так:
        for j in range(k, n):
            dot = 2 * np.dot(v, R[k:, j])
            R[k:, j] -= dot * v
            
        # Обновляем Q: Q = Q @ (I - 2*v*v^T) 
        # Но мы накапливаем произведение, поэтому удобнее:
        for j in range(m):
            dot = 2 * np.dot(v, Q[k:, j])
            Q[k:, j] -= dot * v
    
    return Q.T, R

# --- Основной QR-алгоритм для поиска собственных значений ---
def qr_algorithm(A, epsilon=0.0001, max_iter=1000):
    """
    Находит все собственные значения симметричной матрицы A
    с помощью QR-алгоритма (через разложение Хаусхолдера).
    
    Возвращает:
        iterations: число итераций
        final_matrix: матрица, ставшая почти диагональной
        eigenvalues: диагональные элементы (собственные значения)
    """
    n = A.shape[0]
    A_k = A.copy().astype(float)
    
    for iteration in range(max_iter):
        # Шаг 1: QR-разложение текущей матрицы методом Хаусхолдера
        Q, R = householder_qr(A_k)
        
        # Шаг 2: Новая матрица = R * Q
        A_next = R @ Q
        
        # Проверка сходимости: максимальный внедиагональный элемент
        # Создаём маску для внедиагональных элементов
        off_diag = A_next.copy()
        np.fill_diagonal(off_diag, 0)  # Обнуляем диагональ
        max_off_diag = np.max(np.abs(off_diag))
        
        # Для печати процесса (можно убрать)
        if iteration % 5 == 0:
            print(f"Итерация {iteration}: max|внедиаг| = {max_off_diag:.6f}")
        
        # Если достигли нужной точности — выходим
        if max_off_diag < epsilon:
            print(f"Сошлось на итерации {iteration} с точностью {epsilon}")
            return iteration, A_next, np.diag(A_next)
        
        A_k = A_next
    
    print("Достигнуто максимальное число итераций")
    return max_iter, A_k, np.diag(A_k)

def jacobi_eigenvalues(A, epsilon=0.0001, max_iter=1000):
    """
    Находит собственные значения симметричной матрицы A
    методом вращений (Якоби).
    
    Возвращает:
        iterations: число итераций
        eigenvalues: массив собственных значений
        V: матрица собственных векторов (опционально)
    """
    n = A.shape[0]
    A_k = A.copy().astype(float)
    V = np.eye(n)  # Для накопления собственных векторов
    
    for iteration in range(max_iter):
        # Шаг 1: Найти максимальный внедиагональный элемент
        max_val = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A_k[i, j]) > max_val:
                    max_val = abs(A_k[i, j])
                    p, q = i, j
        
        # Шаг 2: Проверка сходимости
        if max_val < epsilon:
            print(f"Метод Якоби сошелся за {iteration} итераций")
            return iteration, A_k, np.diag(A_k)
        
        # Шаг 3: Вычисление угла поворота φ
        # Для симметричной матрицы формула: tan(2φ) = 2*A[p,q] / (A[p,p] - A[q,q])
        if A_k[p, p] == A_k[q, q]:
            # Если диагональные элементы равны, угол = π/4
            phi = np.pi / 4
        else:
            tau = (A_k[q, q] - A_k[p, p]) / (2 * A_k[p, q])
            if tau >= 0:
                t = 1 / (tau + np.sqrt(1 + tau**2))
            else:
                t = -1 / (-tau + np.sqrt(1 + tau**2))
            phi = np.arctan(t)
        
        c = np.cos(phi)
        s = np.sin(phi)
        
        # Шаг 4: Обновление матрицы A_k (вращение)
        # Сохраняем старые значения
        app = A_k[p, p]
        aqq = A_k[q, q]
        apq = A_k[p, q]
        
        # Новые диагональные элементы
        A_k[p, p] = c**2 * app - 2 * c * s * apq + s**2 * aqq
        A_k[q, q] = s**2 * app + 2 * c * s * apq + c**2 * aqq
        
        # Обнуляем внедиагональные элементы
        A_k[p, q] = 0
        A_k[q, p] = 0
        
        # Обновляем остальные элементы в строках p и q
        for i in range(n):
            if i != p and i != q:
                aip = A_k[i, p]
                aiq = A_k[i, q]
                A_k[i, p] = c * aip - s * aiq
                A_k[p, i] = A_k[i, p]  # симметрия
                A_k[i, q] = s * aip + c * aiq
                A_k[q, i] = A_k[i, q]  # симметрия
        
        # Шаг 5: Обновление матрицы собственных векторов V
        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq
        
        # Печатаем прогресс каждые 10 итераций
        if iteration % 10 == 0:
            print(f"Итерация {iteration}, max|внедиаг| = {max_val:.6f}")
    
    print("Метод Якоби не сошелся за максимальное число итераций")
    return max_iter, A_k, np.diag(A_k)

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

# Функция для извлечения собственных значений из квазидиагональной матрицы
def extract_eigenvalues_from_quasi_diag(A, epsilon=1e-6):
    n = A.shape[0]
    eigenvalues = []
    i = 0
    while i < n:
        if i < n-1 and abs(A[i+1, i]) > epsilon:  # нашли блок 2×2
            # вычисляем комплексную пару
            a, b = A[i, i], A[i, i+1]
            c, d = A[i+1, i], A[i+1, i+1]
            trace = (a + d) / 2
            det = a*d - b*c
            disc = trace**2 - det
            if disc < 0:  # комплексные
                sqrt_disc = np.sqrt(-disc) * 1j
                eigenvalues.append(trace + sqrt_disc)
                eigenvalues.append(trace - sqrt_disc)
            else:  # вещественные
                sqrt_disc = np.sqrt(disc)
                eigenvalues.append(trace + sqrt_disc)
                eigenvalues.append(trace - sqrt_disc)
            i += 2
        else:  # блок 1×1
            eigenvalues.append(A[i, i])
            i += 1
    return np.array(eigenvalues)

# --- ТЕСТ НА ПРИМЕРЕ 4x4 ---
if __name__ == "__main__":
    # Матрица из примера (симметричная)
    A = np.array([
        [4.0, 1.0, 0.0, 2.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 5.0, 1.0],
        [2.0, 0.0, 1.0, 4.0]
    ])

    A = input_matrix()
    
    print("Исходная матрица A:")
    print(A)
    print("-" * 50)
    """ 
    print("="*70)
    print("="*70)
    print("ПОИСК СОБСТВЕННЫХ ЗНАЧЕНИЙ МЕТОДОМ ВРАЩЕНИЙ"*70)
    print("="*70)
    print("="*70)

    # Запускаем Якоби-алгоритм
    iterations, final_matrix, eigenvalues = jacobi_eigenvalues(A, epsilon=0.0001)

    print("-" * 50)
    print(f"Количество итераций: {iterations}")
    print("Финальная (почти диагональная) матрица:")
    # Печатаем с 4 знаками после запятой
    for row in final_matrix:
        print(" ".join(f"{val:8.4f}" for val in row))
    
    print("\nСобственные значения (диагональ финальной матрицы):")
    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1} = {val:.6f}")
    
    # Проверка: сравниваем с numpy.linalg.eigvals
    true_vals = np.linalg.eigvals(A)
    true_vals_sorted = np.sort(true_vals)
    our_vals_sorted = np.sort(eigenvalues)
    
    print("\n--- ПРОВЕРКА ---")
    print("Наши собственные значения (отсортированы):", our_vals_sorted)
    print("Собственные значения от numpy:", true_vals_sorted)
    print("Разница:", np.abs(our_vals_sorted - true_vals_sorted))



    ##-----------------------------------------------------
    """
    print("="*70)
    print("="*70)
    print("ПОИСК СОБСТВЕННЫХ ЗНАЧЕНИЙ QR-АЛГОРИТМОМ")
    print("="*70)
    print("="*70)

    # Запускаем QR-алгоритм
    iterations, final_matrix, eigenvalues = qr_algorithm(A, epsilon=0.0001)
    
    print("-" * 50)
    print(f"Количество итераций: {iterations}")
    print("Финальная (почти диагональная) матрица:")
    # Печатаем с 4 знаками после запятой
    for row in final_matrix:
        print(" ".join(f"{val:8.4f}" for val in row))
    
    print("\nСобственные значения (диагональ финальной матрицы):")
    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1} = {val:.6f}")
    
    # Проверка: сравниваем с numpy.linalg.eigvals
    true_vals = np.linalg.eigvals(A)
    true_vals_sorted = np.sort(true_vals)
    our_vals_sorted = np.sort(eigenvalues)
    
    print("\n--- ПРОВЕРКА ---")
    print("Наши собственные значения (отсортированы):", our_vals_sorted)
    print("Собственные значения от numpy:", true_vals_sorted)
    print("Разница:", np.abs(our_vals_sorted - true_vals_sorted))

    eigenvalues = extract_eigenvalues_from_quasi_diag(final_matrix)

    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1} = {val:.6f}")
    
    # Проверка: сравниваем с numpy.linalg.eigvals
    true_vals = np.linalg.eigvals(A)
    true_vals_sorted = np.sort(true_vals)
    our_vals_sorted = np.sort(eigenvalues)

    print("Наши собственные значения для мнимых решений (отсортированы):", our_vals_sorted)
    print("Собственные значения от numpy:", true_vals_sorted)
    print("Разница:", np.abs(our_vals_sorted - true_vals_sorted))

    print("\n" + "="*50)
print("АРИФМЕТИЧЕСКАЯ ПРОВЕРКА")
print("="*50)

# След матрицы
trace_A = np.trace(A)
sum_eigenvals = np.sum(eigenvalues)
print(f"След матрицы A: {trace_A:.6f}")
print(f"Сумма собственных значений: {sum_eigenvals}")
print(f"Разница (модуль): {abs(trace_A - sum_eigenvals):.2e}")

# Определитель и произведение
det_A = np.linalg.det(A)
prod_eigenvals = np.prod(eigenvalues)
print(f"\nОпределитель матрицы A: {det_A:.6f}")
print(f"Произведение собственных значений: {prod_eigenvals}")
print(f"Разница (модуль): {abs(det_A - prod_eigenvals):.2e}")
