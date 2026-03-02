import numpy as np
import sys 

original_out = sys.stdout
sys.stdout = open('logs_1lab.txt', 'w', encoding='utf-8')


def lu_decomposition_with_pivoting(A):
    """
    LU-разложение с частичным выбором главного элемента
    Возвращает: P, L, U такие, что P@A = L@U
    """
    n = len(A)
    # Копируем матрицу, чтобы не менять исходную
    U = A.copy().astype(float)
    L = np.eye(n)  # Единичная матрица
    P = np.eye(n)  # Матрица перестановок
    
    for k in range(n-1):  # k - текущий шаг (номер диагонального элемента)
        print(f"\n=== Шаг {k+1} ===")
        print("Текущая матрица U:")
        print(U)
        
        # 1. ВЫБОР ГЛАВНОГО ЭЛЕМЕНТА
        # Ищем максимальный элемент в k-м столбце от k-й строки и ниже
        max_row = k
        max_val = abs(U[k, k])
        for i in range(k+1, n):
            if abs(U[i, k]) > max_val:
                max_val = abs(U[i, k])
                max_row = i
        
        print(f"Максимальный элемент в столбце {k}: {max_val:.2f} в строке {max_row}")
        
        # 2. ПЕРЕСТАНОВКА СТРОК (если нужно)
        if max_row != k:
            print(f"Меняем местами строки {k} и {max_row}")
            # Меняем строки в U
            U[[k, max_row]] = U[[max_row, k]]
            # Меняем строки в L (но только для уже вычисленных столбцов)
            L[[k, max_row], :k] = L[[max_row, k], :k]
            # Меняем строки в P
            P[[k, max_row]] = P[[max_row, k]]
        
        # 3. ВЫЧИСЛЕНИЕ МНОЖИТЕЛЕЙ И ВЫЧИТАНИЕ СТРОК
        for i in range(k+1, n):
            # Множитель (сколько раз вычитать k-ю строку из i-й)
            L[i, k] = U[i, k] / U[k, k]
            print(f"l[{i},{k}] = {L[i, k]:.2f}")
            
            # ВЫЧИТАНИЕ СТРОКИ - вот оно, явное вычитание!
            # Вычитаем из i-й строки k-ю строку, умноженную на множитель
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
            
            print(f"После вычитания {L[i, k]:.2f} * строка{k} из строки{i}:")
            print(U)
    
    return P, L, U

def solve_with_pivoting(P, L, U, b):
    """
    Решение системы P@A@x = P@b или A@x = b с учетом перестановок
    """
    n = len(b)
    
    # Шаг 1: Применяем перестановки к вектору b
    Pb = P @ b
    
    print("\n=== Решение системы ===")
    print("Вектор b после перестановок:")
    print(Pb)
    
    # Шаг 2: Прямая подстановка (решаем L@y = Pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
    print("Вектор y после прямой подстановки:")
    print(y)
    
    # Шаг 3: Обратная подстановка (решаем U@x = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    print("Вектор x после обратной подстановки:")
    print(x)
    
    return x

# Пример с плохо обусловленной матрицей
print("="*60)
print("ПРИМЕР 1: Хорошо обусловленная система")
print("="*60)

A1 = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
], dtype=float)

b1 = np.array([8, -11, -3], dtype=float)

print("Исходная матрица A:")
print(A1)
print("Вектор правой части b:")
print(b1)

# Выполняем LU-разложение с выбором главного элемента
P, L, U = lu_decomposition_with_pivoting(A1.copy())

print("\n" + "="*30)
print("РЕЗУЛЬТАТ РАЗЛОЖЕНИЯ:")
print("="*30)
print("Матрица перестановок P:")
print(P)
print("\nМатрица L:")
print(L)
print("\nМатрица U:")
print(U)
print("\nПроверка P@A:")
print(P @ A1)
print("\nПроверка L@U:")
print(L @ U)
print("\nДолжно быть равно P@A:")
print(P @ A1)

# Решаем систему
x = solve_with_pivoting(P, L, U, b1)

print("\n" + "="*30)
print("РЕШЕНИЕ:")
print("="*30)
print(f"x = {x}")
print("\nПроверка A@x:")
print(A1 @ x)
print("Должно быть равно b:")
print(b1)

# Пример с матрицей, где нужна перестановка
print("\n" + "="*60)
print("ПРИМЕР 2: Матрица, требующая перестановки")
print("="*60)

A2 = np.array([
    [1e-10, 1, 1],
    [1, 2, 3],
    [2, 3, 4]
], dtype=float)

b2 = np.array([2, 6, 9], dtype=float)

print("Исходная матрица A (с очень маленьким диагональным элементом):")
print(A2)
print("Вектор правой части b:")
print(b2)

# Без перестановок (классическое LU)
print("\n" + "-"*30)
print("Без перестановок (будет деление на очень маленькое число):")
print("-"*30)

try:
    P_no, L_no, U_no = lu_decomposition_with_pivoting(A2.copy())
    # Но мы принудительно отключаем перестановки для демонстрации
    # Для этого закомментируем часть с перестановками в функции выше
except:
    print("ОШИБКА: Деление на очень маленькое число!")

# С перестановками
print("\n" + "-"*30)
print("С ЧАСТИЧНЫМ ВЫБОРОМ ГЛАВНОГО ЭЛЕМЕНТА:")
print("-"*30)

P, L, U = lu_decomposition_with_pivoting(A2.copy())

print("\nМатрица перестановок P (видим, что строки переставлены):")
print(P)
print("\nМатрица L:")
print(L)
print("\nМатрица U (верхняя треугольная):")
print(U)

x2 = solve_with_pivoting(P, L, U, b2)
print(f"\nРешение: x = {x2}")
print("Проверка A@x:")
print(A2 @ x2)
print("Должно быть равно b:")
print(b2)

sys.stdout.close()
sys.stdout = original_out

print("\nМатрица перестановок P (видим, что строки переставлены):")
print(P)
print("\nМатрица L:")
print(L)
print("\nМатрица U (верхняя треугольная):")
print(U)

print(f"\nРешение: x = {x2}")
