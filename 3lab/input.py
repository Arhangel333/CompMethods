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

# 12 variant
# L_and_N.py

x_values = [0, 1, 2, 3, 4]
y_values = [1, 4, 3, 5, 2]
x_star = 2.5

x_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y_values = [-0.4935, -0.0745, -0.0197, -0.2890, 0.1293, 1.5128, 1.7135, 2.5618, 2.8339]
x_star  = 1.839

#cub_spline.py

x_values_spline = [0, 1, 2, 3, 4]
y_values_spline = [1, 4, 3, 5, 2]
x_star_spline = 2.5


x_values_spline = [ 0.00, 0.28, 0.655, 0.98, 1.40, 1.855, 2.17, 2.555, 2.975, 3.22, 3.50 ]
y_values_spline = [ -5.539, -2.614, -1.028, -2.873, -3.718, -4.837, -1.729, -1.376, -2.794, -3.972, -3.473 ]
x_star_spline = 2.712

#MNK.py

x_values_mnk = [0, 1, 2, 3, 4, 5, 6]     # ← замените на свои x
y_values_mnk = [1.2, 2.8, 4.1, 5.3, 6.2, 7.5, 8.9]  # ← замените на свои y
x_star_mnk = 3.5                          # ← замените на свою точку x*

x_values_mnk = [ -3.0,	-2.5,	-2.0,	-1.5,	-1.0,	-0.5,	0.0,	0.5,	1.0,	1.5]
y_values_mnk = [ 1.5142,	0.7347,	0.5018,	0.4466,	0.7617,	1.1625,	1.9950,	2.5418,	2.8169,	2.6793]
x_star_mnk = 0.783
