import input
import math

# ------------------------------------------------------------
# 1. Ввод данных
# ------------------------------------------------------------
print("Вычисление определенного интеграла")
print("----------------------------------")


n = input.n  # начальное число отрезков (чётное для Симпсона)
max_iter = input.max_iter   # защита от бесконечного цикла

# Ввод подынтегральной функции (как строку)
#f_str = input("Введите функцию f(x) на языке Python (например, 'x**2', 'math.sin(x)', 'math.exp(-x**2)'): ")
f_str = input.f_str

# Ввод пределов интегрирования
#a = float(input("Введите нижний предел a: "))
#b = float(input("Введите верхний предел b: "))
a = input.a
b = input.b

# Ввод точности
#eps = float(input("Введите требуемую точность (например, 1e-4, 1e-6): "))
eps = input.eps

# Превращаем строку в настоящую функцию
def f(x):
    return eval(f_str)




# ------------------------------------------------------------
# 2. Методы интегрирования
# ------------------------------------------------------------

def method_rectangles(f, a, b, n):
    """
    Метод средних прямоугольников.
    n – число отрезков (должно быть чётным для удобства, но не обязательно)
    """
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5) * h   # середина i-го отрезка
        s += f(x_mid)
    return h * s

def method_trapezoids(f, a, b, n):
    """
    Метод трапеций.
    n – число отрезков
    """
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in range(1, n):
        x = a + i * h
        s += f(x)
    return h * s

def method_simpson(f, a, b, n):
    """
    Метод Симпсона (парабол).
    n – число отрезков, ДОЛЖНО БЫТЬ ЧЁТНЫМ.
    """
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона n должно быть чётным!")
    h = (b - a) / n
    s = f(a) + f(b)
    # Нечётные индексы (1,3,5,..., n-1) – коэффициент 4
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    # Чётные индексы (2,4,6,..., n-2) – коэффициент 2
    for i in range(2, n, 2):
        s += 2 * f(a + i * h)
    return (h / 3) * s

# ------------------------------------------------------------
# 3. Уточнение по Рунге–Ромбергу
# ------------------------------------------------------------
def runge_romberg(I_h, I_half, p):
    """
    Уточнение по Рунге–Ромбергу.
    I_h    – значение с шагом h
    I_half – значение с шагом h/2
    p      – порядок точности метода (2 для прямоуг/трап, 4 для Симпсона)
    """
    return I_half + (I_half - I_h) / ((2**p) - 1)

# ------------------------------------------------------------
# 4. Основной цикл подбора шага
# ------------------------------------------------------------
print("\nФункция:")
print(f"F = ∫_{a}^{b}", f_str.replace("math.", ""))
print("\nНачинаем подбор шага...\n")

n_rect = None
n_trap = None
n_simp = None

iter = 0

for iteration in range(max_iter):
    iter +=1
    h = (b - a) / n

    # Вычисляем интегралы для текущего n
    I_rect = method_rectangles(f, a, b, n)
    I_trap = method_trapezoids(f, a, b, n)
    I_simp = method_simpson(f, a, b, n)

    # Удваиваем n и считаем заново
    n2 = n * 2
    h2 = (b - a) / n2
    I_rect2 = method_rectangles(f, a, b, n2)
    I_trap2 = method_trapezoids(f, a, b, n2)
    I_simp2 = method_simpson(f, a, b, n2)

    # Оценка погрешности по Рунге (разница между двумя приближениями)
    err_rect = abs(I_rect2 - I_rect)
    err_trap = abs(I_trap2 - I_trap)
    err_simp = abs(I_simp2 - I_simp)

    # Уточнённые значения по Рунге–Ромбергу
    I_rect_fine = runge_romberg(I_rect, I_rect2, p=2)
    I_trap_fine = runge_romberg(I_trap, I_trap2, p=2)
    I_simp_fine = runge_romberg(I_simp, I_simp2, p=4)

    # Вывод на экран текущей итерации
    print(f"n = {n:4d}  h = {h:.6f} iter = {iter}")
    print(f"  Прямоугольники:  I={I_rect:.8f}  разница={err_rect:.2e}  уточнённое={I_rect_fine:.8f}")
    print(f"  Трапеции:        I={I_trap:.8f}  разница={err_trap:.2e}  уточнённое={I_trap_fine:.8f}")
    print(f"  Симпсон:         I={I_simp:.8f}  разница={err_simp:.2e}  уточнённое={I_simp_fine:.8f}")
    print()

    # Проверяем, достигнута ли точность (по уточнённому значению или по разнице)
    # Используем критерий: погрешность ≈ |I_half - I_h| / (2^p - 1) < eps
    # Для Симпсона p=4 → знаменатель 15
    # Для прямоугольников и трапеций p=2 → знаменатель 3
    stop_rect = (err_rect / 3) < eps
    stop_trap = (err_trap / 3) < eps
    stop_simp = (err_simp / 15) < eps

    if stop_rect and n_rect is None:
        n_rect = n
    if stop_trap and n_trap is None:
        n_trap = n
    if stop_simp and n_simp is None:
        n_simp = n



    if stop_rect and stop_trap and stop_simp:
        print("Точность достигнута для всех методов!")
        print(f"Метод средних прямоугольников: ",n_rect)
        print("Метод трапеций: ", n_trap)
        print("Метод Симпсона: ", n_simp)
        break

    # Иначе удваиваем n и продолжаем
    n = n2

else:
    print("Внимание: достигнут лимит итераций. Точность могла не быть достигнута.")

# ------------------------------------------------------------
# 5. Финальный вывод
# ------------------------------------------------------------
print("\n" + "="*50)
print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (уточнённый по Рунге–Ромбергу):")
print("="*50)
print(f"Метод средних прямоугольников:  {I_rect_fine:.10f}")
print(f"Метод трапеций:                 {I_trap_fine:.10f}")
print(f"Метод Симпсона:                 {I_simp_fine:.10f}")
print("="*50)