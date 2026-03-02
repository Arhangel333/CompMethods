import numpy as np
import sys

def lu_decomposition_with_pivoting(A, log_file):
    """
    LU-разложение с частичным выбором главного элемента
    Возвращает: P, L, U такие, что P@A = L@U
    """
    n = len(A)
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("ПРОЦЕСС LU-РАЗЛОЖЕНИЯ\n")
    log_file.write("="*60 + "\n")
    
    for k in range(n-1):
        log_file.write(f"\n--- Шаг {k+1} ---\n")
        log_file.write(f"Текущая матрица U:\n{U}\n")
        
        # Выбор главного элемента
        max_row = k
        max_val = abs(U[k, k])
        for i in range(k+1, n):
            if abs(U[i, k]) > max_val:
                max_val = abs(U[i, k])
                max_row = i
        
        log_file.write(f"Максимальный элемент в столбце {k}: {max_val:.6f} в строке {max_row}\n")
        
        # Перестановка строк
        if max_row != k:
            log_file.write(f"Меняем местами строки {k} и {max_row}\n")
            U[[k, max_row]] = U[[max_row, k]]
            L[[k, max_row], :k] = L[[max_row, k], :k]
            P[[k, max_row]] = P[[max_row, k]]
            log_file.write(f"U после перестановки:\n{U}\n")
        
        # Вычисление множителей и вычитание строк
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            log_file.write(f"  l[{i},{k}] = {L[i, k]:.6f}\n")
            
            old_row = U[i, :].copy()
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
            
            log_file.write(f"  Строка {i} после вычитания:\n    Было: {old_row}\n    Стало: {U[i, :]}\n")
    
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("LU-РАЗЛОЖЕНИЕ ЗАВЕРШЕНО\n")
    log_file.write("="*60 + "\n")
    
    return P, L, U


def solve_system(L, U, P, b, log_file):
    """
    Решение системы P@A@x = P@b
    """
    n = len(b)
    
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("РЕШЕНИЕ СИСТЕМЫ\n")
    log_file.write("="*60 + "\n")
    
    # Применяем перестановки к b
    b_permuted = P @ b
    log_file.write(f"Исходный b: {b}\n")
    log_file.write(f"b после перестановок (P@b): {b_permuted}\n")
    
    # Прямая подстановка (Ly = b_permuted)
    y = np.zeros(n)
    log_file.write("\n--- Прямая подстановка (Ly = P@b) ---\n")
    for i in range(n):
        y[i] = b_permuted[i]
        log_file.write(f"  y[{i}] начинает с {y[i]:.6f}\n")
        for j in range(i):
            y[i] -= L[i, j] * y[j]
            log_file.write(f"    вычитаем L[{i},{j}]×y[{j}] = {L[i, j]:.6f}×{y[j]:.6f}\n")
        log_file.write(f"  y[{i}] = {y[i]:.6f}\n")
    
    log_file.write(f"\nВектор y: {y}\n")
    
    # Обратная подстановка (Ux = y)
    x = np.zeros(n)
    log_file.write("\n--- Обратная подстановка (Ux = y) ---\n")
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        log_file.write(f"  x[{i}] начинает с {x[i]:.6f}\n")
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
            log_file.write(f"    вычитаем U[{i},{j}]×x[{j}] = {U[i, j]:.6f}×{x[j]:.6f}\n")
        x[i] /= U[i, i]
        log_file.write(f"  делим на U[{i},{i}] = {U[i, i]:.6f}\n")
        log_file.write(f"  x[{i}] = {x[i]:.6f}\n")
    
    log_file.write(f"\nВектор x: {x}\n")
    
    return x, y, b_permuted


def calculate_determinant(P, U, log_file):
    """
    Вычисление определителя матрицы A
    det(A) = det(P⁻¹) × det(U), det(L)=1
    """
    n = len(P)
    
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("ВЫЧИСЛЕНИЕ ОПРЕДЕЛИТЕЛЯ\n")
    log_file.write("="*60 + "\n")
    
    # Вычисляем определитель P через число перестановок
    I = np.eye(n)
    P_copy = P.copy()
    permutations = 0
    
    log_file.write("Анализ матрицы перестановок P:\n")
    for i in range(n):
        if not np.array_equal(P_copy[i], I[i]):
            for j in range(i+1, n):
                if np.array_equal(P_copy[j], I[i]):
                    log_file.write(f"  Меняем строки {i} и {j}\n")
                    P_copy[[i, j]] = P_copy[[j, i]]
                    permutations += 1
                    break
    
    det_P = (-1) ** permutations
    log_file.write(f"Число перестановок: {permutations}\n")
    log_file.write(f"det(P) = (-1)^{permutations} = {det_P}\n")
    
    # Определитель U - произведение диагональных элементов
    det_U = 1.0
    log_file.write("\nПроизведение диагональных элементов U:\n")
    for i in range(n):
        log_file.write(f"  U[{i},{i}] = {U[i, i]:.6f}\n")
        det_U *= U[i, i]
    
    log_file.write(f"det(U) = {det_U:.6f}\n")
    
    det_A = det_P * det_U
    log_file.write(f"\n🔷 det(A) = det(P) × det(U) = {det_P} × {det_U:.6f} = {det_A:.6f}\n")
    
    return det_A, det_U, det_P, permutations


def inverse_matrix(P, L, U, log_file):
    """
    Построение обратной матрицы A⁻¹
    Решаем A × X = I для каждого столбца
    """
    n = len(P)
    A_inv = np.zeros((n, n))
    I = np.eye(n)
    
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("ПОСТРОЕНИЕ ОБРАТНОЙ МАТРИЦЫ\n")
    log_file.write("="*60 + "\n")
    
    for col in range(n):
        log_file.write(f"\n--- Столбец {col+1} ---\n")
        b = I[:, col]
        log_file.write(f"Правая часть: столбец {col+1} единичной матрицы = {b}\n")
        
        # Применяем перестановки
        b_permuted = P @ b
        log_file.write(f"После перестановок: {b_permuted}\n")
        
        # Прямая подстановка
        y = np.zeros(n)
        for i in range(n):
            y[i] = b_permuted[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
        
        # Обратная подстановка
        x_col = np.zeros(n)
        for i in range(n-1, -1, -1):
            x_col[i] = y[i]
            for j in range(i+1, n):
                x_col[i] -= U[i, j] * x_col[j]
            x_col[i] /= U[i, i]
        
        A_inv[:, col] = x_col
        log_file.write(f"Полученный столбец: {x_col}\n")
    
    return A_inv


def print_result_to_console(A, b, P, L, U, x, det_A, A_inv):
    """Вывод только итоговых результатов на экран"""
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ")
    print("="*70)
    
    print("\n📌 Исходные данные:")
    print("Матрица A:")
    print(A)
    print("\nВектор b:")
    print(b)
    
    print("\n📌 Результаты LU-разложения:")
    print("Матрица перестановок P:")
    print(P)
    print("\nНижняя треугольная матрица L:")
    print(L)
    print("\nВерхняя треугольная матрица U:")
    print(U)
    
    print("\n📌 Решение системы A·x = b:")
    print(f"x = {x}")
    
    print("\n📌 Проверка решения (A·x):")
    print(A @ x)
    print("Должно быть равно b:")
    print(b)
    
    print(f"\n📌 Определитель матрицы A:")
    print(f"det(A) = {det_A:.6f}")
    
    print("\n📌 Обратная матрица A⁻¹:")
    print(A_inv)
    
    print("\n📌 Проверка (A · A⁻¹):")
    print(A @ A_inv)
    print("Должна быть единичная матрица:")
    print(np.eye(len(A)))
    
    print("\n" + "="*70)
    print("Подробный ход вычислений сохранен в файл 'lu_log.txt'")
    print("="*70)


def read_matrix_from_file():
    """
    Считывание матрицы A и вектора b из стандартного ввода.
    
    Формат ввода:
    - Строки с числами через пробел (матрица A)
    - Пустая строка (разделитель)
    - Строка с числами через пробел (вектор b)
    
    Для завершения ввода нажмите Ctrl+D (Linux/Mac) или Ctrl+Z (Windows)
    
    Пример ввода:
    2 1 -1
    -3 -1 2
    -2 1 2
    
    8 -11 -3
    """
    import sys
    
    print("Введите матрицу A (строки через пробел):")
    print("После последней строки матрицы оставьте пустую строку")
    print("Затем введите вектор b")
    print("(Для завершения ввода нажмите Ctrl+D или Ctrl+Z)")
    print("-" * 50)
    
    try:
        # Читаем все строки из stdin
        lines = []
        for line in sys.stdin:
            lines.append(line.rstrip())
        
        # Разделяем на две части по пустой строке
        matrix_lines = []
        vector_lines = []
        
        current_part = matrix_lines
        for line in lines:
            if (line == "" or line == "\n")and current_part == matrix_lines:
                # Пустая строка - переключаемся на вектор
                current_part = vector_lines
                continue
            elif line == "":
                # Игнорируем лишние пустые строки
                continue
            current_part.append(line)
        
        # Проверяем, что обе части не пустые
        if not matrix_lines:
            raise ValueError("Не введена матрица A")
        if not vector_lines:
            raise ValueError("Не введен вектор b")
        
        # Парсим матрицу A
        A_rows = []
        for i, line in enumerate(matrix_lines):
            try:
                row = list(map(float, line.split()))
                if not row:
                    raise ValueError(f"Строка {i+1} матрицы пуста")
                A_rows.append(row)
            except ValueError as e:
                raise ValueError(f"Ошибка в строке {i+1} матрицы: {line} — {e}")
        
        # Проверяем, что матрица квадратная
        n = len(A_rows)
        for i, row in enumerate(A_rows):
            if len(row) != n:
                raise ValueError(f"Строка {i+1} матрицы содержит {len(row)} элементов, а должно быть {n}")
        
        A = np.array(A_rows, dtype=float)
        
        # Парсим вектор b
        if len(vector_lines) > 1:
            print(f"\n⚠️ Внимание: найдено {len(vector_lines)} строк для вектора. Использую первую:")
            print(f"   {vector_lines[0]}")
            print(f"   Остальные строки игнорируются: {vector_lines[1:]}")
        
        b_line = vector_lines[0]
        try:
            b = np.array(list(map(float, b_line.split())), dtype=float)
        except ValueError as e:
            raise ValueError(f"Ошибка в векторе b: {b_line} — {e}")
        
        if len(b) != n:
            raise ValueError(f"Вектор b содержит {len(b)} элементов, а должно быть {n}")
        
        print("\n" + "-" * 50)
        print("✅ Данные успешно прочитаны")
        print(f"   Размерность матрицы: {n}×{n}")
        print(f"   Матрица A:")
        for row in A:
            print(f"     {row}")
        print(f"   Вектор b: {b}")
        print("-" * 50)
        
        return A, b, n
        
    except EOFError:
        print("\n⚠️ Ввод прерван (EOF)")
        return None, None, None
    except KeyboardInterrupt:
        print("\n⚠️ Ввод прерван пользователем")
        return None, None, None
    except ValueError as e:
        print(f"\n❌ Ошибка в формате ввода: {e}")
        return None, None, None
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        return None, None, None


def main():
    # Открываем файл для логов
    with open('lu_log.txt', 'w', encoding='utf-8') as log_file:
        log_file.write("ЛАБОРАТОРНАЯ РАБОТА: LU-РАЗЛОЖЕНИЕ\n")
        log_file.write("="*70 + "\n")
        
        # ========== ПРИМЕР 1 ==========
        log_file.write("\n\n" + "🔥"*35 + "\n")
        log_file.write("ПРИМЕР 1: Хорошо обусловленная система\n")
        log_file.write("🔥"*35 + "\n")

        filename = "input.txt"
        A1, b1, n = read_matrix_from_file()
        


        
        
        log_file.write("\nИсходная матрица A1:\n")
        log_file.write(str(A1) + "\n")
        log_file.write("\nВектор b1:\n")
        log_file.write(str(b1) + "\n")
        
        # LU-разложение
        P1, L1, U1 = lu_decomposition_with_pivoting(A1.copy(), log_file)
        
        # Решение системы
        x1, y1, b_perm1 = solve_system(L1, U1, P1, b1, log_file)
        
        # Определитель
        det1, det_U1, det_P1, perm_count1 = calculate_determinant(P1, U1, log_file)
        
        # Обратная матрица
        A1_inv = inverse_matrix(P1, L1, U1, log_file)
        
        # Вывод результатов для примера 1 на экран
        print_result_to_console(A1, b1, P1, L1, U1, x1, det1, A1_inv)
        
        log_file.write("\n" + "="*70 + "\n")
        log_file.write("РАБОТА ПРОГРАММЫ ЗАВЕРШЕНА\n")
        log_file.write("="*70 + "\n")
    
    # После закрытия файла с логами
    print("\n✅ Все вычисления выполнены. Логи сохранены в 'lu_log.txt'")


if __name__ == "__main__":
    main()
