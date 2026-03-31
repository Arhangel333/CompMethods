import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================
# ЗАДАНИЕ 1: Интерполяция полиномами Лагранжа и Ньютона
# ============================================================

# Исходные данные таблицы 1, вариант 11
x1 = np.array([0.21, 0.42, 0.63, 0.84, 1.05, 1.26, 1.47, 1.68, 1.89])
y1 = np.array([6.5964, 6.3731, 4.7526, 1.2715, 0.7438, 2.5864, 3.6853, 4.8782, 4.6284])
x_star = 1.383

print("="*70)
print("ЗАДАНИЕ 1: Интерполяция")
print("\nИсходные данные (таблица 1):")
print(" i |    x_i    |    y_i")
for i in range(len(x1)):
    print(f"{i:2d} | {x1[i]:8.4f} | {y1[i]:8.4f}")
print(f"\nТочка интерполяции x* = {x_star}")

# Выбор узлов для полиномов 2-й и 3-й степени (ближайшие к x*)
# 2-я степень: узлы 5,6,7 (1.26, 1.47, 1.68)
x2 = x1[[5,6,7]]
y2 = y1[[5,6,7]]
# 3-я степень: узлы 4,5,6,7 (1.05, 1.26, 1.47, 1.68)
x3 = x1[[4,5,6,7]]
y3 = y1[[4,5,6,7]]

# Функция Лагранжа
def lagrange(x, xi, yi):
    n = len(xi)
    res = 0.0
    for i in range(n):
        term = yi[i]
        for j in range(n):
            if j != i:
                term *= (x - xi[j]) / (xi[i] - xi[j])
        res += term
    return res

# Функция Ньютона (разделённые разности)
def newton(x, xi, yi):
    n = len(xi)
    F = np.zeros((n, n))
    F[:,0] = yi
    for j in range(1, n):
        for i in range(n - j):
            F[i,j] = (F[i+1,j-1] - F[i,j-1]) / (xi[i+j] - xi[i])
    res = F[0,0]
    prod = 1.0
    for k in range(1, n):
        prod *= (x - xi[k-1])
        res += F[0,k] * prod
    return res

L2 = lagrange(x_star, x2, y2)
L3 = lagrange(x_star, x3, y3)
N2 = newton(x_star, x2, y2)
N3 = newton(x_star, x3, y3)

# Оценка погрешности
# 2-я степень: разделённая разность 3-го порядка на 4 точках (x3)
F3 = np.zeros((4,4))
F3[:,0] = y3
for j in range(1,4):
    for i in range(4-j):
        F3[i,j] = (F3[i+1,j-1] - F3[i,j-1]) / (x3[i+j] - x3[i])
omega2 = (x_star - x3[0]) * (x_star - x3[1]) * (x_star - x3[2])
err2 = abs(F3[0,3] * omega2)

# 3-я степень: разделённая разность 4-го порядка на 5 точках (добавляем узел 3: x=0.84)
x4 = x1[[3,4,5,6,7]]   # 0.84,1.05,1.26,1.47,1.68
y4 = y1[[3,4,5,6,7]]
F4 = np.zeros((5,5))
F4[:,0] = y4
for j in range(1,5):
    for i in range(5-j):
        F4[i,j] = (F4[i+1,j-1] - F4[i,j-1]) / (x4[i+j] - x4[i])
omega3 = (x_star - x4[0]) * (x_star - x4[1]) * (x_star - x4[2]) * (x_star - x4[3])
err3 = abs(F4[0,4] * omega3)

print("\nРезультаты интерполяции:")
print(f"Лагранж 2-й степени: L2({x_star}) = {L2:.6f}")
print(f"Лагранж 3-й степени: L3({x_star}) = {L3:.6f}")
print(f"Ньютон 2-й степени:  N2({x_star}) = {N2:.6f}")
print(f"Ньютон 3-й степени:  N3({x_star}) = {N3:.6f}")
print(f"Оценка погрешности 2-й степени: ~ {err2:.6f}")
print(f"Оценка погрешности 3-й степени: ~ {err3:.6f}")

# ============================================================
# ЗАДАНИЕ 2: Естественный кубический сплайн
# ============================================================

x_spl = np.array([-3.00, -2.545, -2.025, -1.31, -0.725, 0.12, 0.90, 1.68, 2.525, 2.98, 3.50])
y_spl = np.array([-6.382, -4.973, -1.254, -0.187, 0.928, 1.813, 1.054, 0.372, -0.876, -2.972, -3.645])
x_star2 = 0.524

print("\n" + "="*70)
print("ЗАДАНИЕ 2: Естественный кубический сплайн")
print("\nИсходные данные (таблица 2):")
print(" i |    x_i    |    y_i")
for i in range(len(x_spl)):
    print(f"{i:2d} | {x_spl[i]:8.4f} | {y_spl[i]:8.4f}")
print(f"\nТочка x* = {x_star2}")

# Создаём сплайн с естественными граничными условиями
cs = CubicSpline(x_spl, y_spl, bc_type='natural')
y_spl_star = cs(x_star2)

# Коэффициенты сплайна
coeffs = cs.c   # shape (4, n-1): [a, b, c, d] на каждом отрезке (в степенях (x-x_i))
print(f"\nS({x_star2}) = {y_spl_star:.6f}")
print("\nКоэффициенты сплайна S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3:")
print("  i |    x_i    |      a_i      |      b_i      |      c_i      |      d_i")
for i in range(len(x_spl)-1):
    print(f"{i:3d} | {x_spl[i]:8.4f} | {coeffs[0,i]:12.6f} | {coeffs[1,i]:12.6f} | {coeffs[2,i]:12.6f} | {coeffs[3,i]:12.6f}")

# Построение графиков сплайна и его производных
x_plot_spl = np.linspace(x_spl[0], x_spl[-1], 500)
y_plot_spl = cs(x_plot_spl)
y_plot_spl_der1 = cs.derivative()(x_plot_spl)
y_plot_spl_der2 = cs.derivative(2)(x_plot_spl)

plt.figure(figsize=(12, 10))

# График сплайна
plt.subplot(3,1,1)
plt.plot(x_plot_spl, y_plot_spl, 'b-', label='Кубический сплайн')
plt.scatter(x_spl, y_spl, color='red', zorder=5, label='Узлы')
plt.scatter(x_star2, y_spl_star, color='green', s=100, zorder=5, marker='*', label=f'x* = {x_star2}')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.title('Естественный кубический сплайн (дефект 1)')
plt.legend()
plt.grid(True)

# График первой производной
plt.subplot(3,1,2)
plt.plot(x_plot_spl, y_plot_spl_der1, 'g-', label='S\'(x)')
plt.xlabel('x')
plt.ylabel('S\'(x)')
plt.title('Первая производная сплайна')
plt.legend()
plt.grid(True)

# График второй производной
plt.subplot(3,1,3)
plt.plot(x_plot_spl, y_plot_spl_der2, 'r-', label='S\'\'(x)')
plt.xlabel('x')
plt.ylabel('S\'\'(x)')
plt.title('Вторая производная сплайна')
plt.legend()
plt.grid(True)

plt.tight_layout()

# ============================================================
# ЗАДАНИЕ 3: Аппроксимация МНК
# ============================================================

x_lsq = np.array([-5.48, -4.73, -3.98, -3.23, -2.48, -1.73, -0.98, -0.23, 0.52, 1.27, 2.02])
y_lsq = np.array([3.4931, 3.6844, 3.4329, 2.3856, 1.6693, 1.4538, 1.7219, 2.5137, 2.7812, 2.2754, 1.1749])
x_star3 = 0.024

print("\n" + "="*70)
print("ЗАДАНИЕ 3: Аппроксимация методом наименьших квадратов")
print("\nИсходные данные (таблица 3):")
print(" i |    x_i    |    y_i")
for i in range(len(x_lsq)):
    print(f"{i:2d} | {x_lsq[i]:8.4f} | {y_lsq[i]:8.4f}")
print(f"\nТочка x* = {x_star3}")

degrees = [1, 2, 3]
polys = []
sse_list = []
values = []

for deg in degrees:
    coef = np.polyfit(x_lsq, y_lsq, deg)
    p = np.poly1d(coef)
    polys.append(p)
    val = p(x_star3)
    values.append(val)
    y_pred = p(x_lsq)
    sse = np.sum((y_lsq - y_pred)**2)
    sse_list.append(sse)
    print(f"\nМногочлен степени {deg}:")
    print(f"  Коэффициенты: {coef}")
    print(f"  P({x_star3}) = {val:.6f}")
    print(f"  Сумма квадратов ошибок SSE = {sse:.6f}")

# График аппроксимации
plt.figure(figsize=(10,6))
plt.scatter(x_lsq, y_lsq, color='red', label='Исходные данные', zorder=5)
x_plot = np.linspace(min(x_lsq), max(x_lsq), 200)
colors = ['blue', 'green', 'orange']
for i, deg in enumerate(degrees):
    plt.plot(x_plot, polys[i](x_plot), color=colors[i], label=f'Степень {deg}, SSE={sse_list[i]:.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация МНК (вариант 11)')
plt.legend()
plt.grid(True)

# Отображение всех графиков
plt.tight_layout()
plt.show()