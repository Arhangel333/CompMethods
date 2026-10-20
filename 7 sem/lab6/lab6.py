import numpy as np
import matplotlib.pyplot as plt


L = np.pi #координатный промежуток всей сетки

T = 1.0     #временной промежуток всей сетки
a = 1.0     #коэфф из уравнения


M = 50         #длина сетки по координате
h = L/M     #шаг по координате
tau = 0.9 * h/a       #шаг по времени
N = int(T/tau) #длина сетки по времени


tau = T / N


x = np.linspace(0, L, M + 1)
t = np.linspace(0, T, N + 1)







def u_exact(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)


def progonka(a_sub, b_main, c_sup, d):
    """
    Решает трехдиагональную систему методом прогонки.
    a_sub: нижняя диагональ (длина N-1)
    b_main: главная диагональ (длина N)
    c_sup: верхняя диагональ (длина N-1)
    d: правая часть (длина N)
    """
    n = len(d)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)

    # Прямой ход
    alpha[0] = -c_sup[0] / b_main[0]
    beta[0] = d[0] / b_main[0]
    
    for i in range(1, n):
        denom = b_main[i] + a_sub[i-1] * alpha[i-1]
        alpha[i] = -c_sup[i] / denom if i < n-1 else 0
        beta[i] = (d[i] - a_sub[i-1] * beta[i-1]) / denom

    # Обратный ход
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
        
    return x

def neayvn(x, tau, h, N, M, a):
    u = np.zeros((N + 1, M + 1))
    
    # Начальные условия (такие же, как в явной схеме)
    u[0, :] = np.sin(x) + np.cos(x)
    psi = -a * (np.sin(x) + np.cos(x))
    u[1, :] = u[0, :] + tau * psi + (tau**2 / 2) * a**2 * (-u[0, :])
    
    sigma = (a * tau / h)**2
    
    for n in range(1, N):
        size = M - 1  # Количество внутренних узлов
        
        # 1. Формируем трехдиагональную матрицу
        a_sub = np.full(size - 1, -sigma)          # Нижняя диагональ
        b_main = np.full(size, 1 + 2 * sigma)      # Главная диагональ
        c_sup = np.full(size - 1, -sigma)          # Верхняя диагональ
        
        # Правая часть
        d = 2 * u[n, 1:-1] - u[n-1, 1:-1]
        
        # 2. Встраиваем граничные условия 2-го порядка в матрицу
        # Левая граница: u_0 = (4*u_1 - u_2) / (3 + 2*h)
        # Подставляем в 1-е уравнение: -sigma*u_0 + (1+2*sigma)*u_1 - sigma*u_2 = d_0
        b_main[0] = 1 + 2 * sigma - 4 * sigma / (3 + 2 * h)
        c_sup[0] = -sigma + sigma / (3 + 2 * h)
        
        # Правая граница: u_M = (4*u_{M-1} - u_{M-2}) / (3 - 2*h)
        # Подставляем в последнее уравнение: -sigma*u_{M-2} + (1+2*sigma)*u_{M-1} - sigma*u_M = d_{M-1}
        a_sub[-1] = -sigma + sigma / (3 - 2 * h)
        b_main[-1] = 1 + 2 * sigma - 4 * sigma / (3 - 2 * h)
        
        # 3. Решаем систему методом прогонки
        u_internal = progonka(a_sub, b_main, c_sup, d)
        
        # 4. Записываем результаты
        u[n+1, 1:-1] = u_internal
        
        # 5. Вычисляем значения на границах по тем же формулам 2-го порядка
        u[n+1, 0] = (4 * u[n+1, 1] - u[n+1, 2]) / (3 + 2 * h)
        u[n+1, -1] = (4 * u[n+1, -2] - u[n+1, -3]) / (3 - 2 * h)

    return u
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def yavn(x, tau, h, N, M, a):


    u = np.zeros((N + 1, M + 1))
    
    # Нулевой слой по времени
    u[0, :] = np.sin(x) + np.cos(x)

    # Начальная скорость
    psi = -a * (np.sin(x) + np.cos(x))

    # Коэффициент схемы
    sigma = (a * tau / h)**2


    #   РАССЧЕТ СЕТКИ==============================================
    #получаем искомую сеточную функцию со вторым порядком точности

    u[1, :] = u[0, :] + tau * psi + (tau**2 / 2) * a**2 * (-u[0, :])

    for n in range(1,N):
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + sigma * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        
        
        u[n+1, 0] = (4 * u[n+1, 1] - u[n+1, 2]) / (3 + 2 * h)
        u[n+1, -1] = (4 * u[n+1, -2] - u[n+1, -3]) / (3 - 2 * h)

    return u


def yavn_bc12(x, tau, h, N, M, a, bc=1):
    """1-й порядок, двухточечная аппроксимация (направленные разности)"""
    u = np.zeros((N + 1, M + 1))
    u[0, :] = np.sin(x) + np.cos(x)
    psi = -a * (np.sin(x) + np.cos(x))
    u[1, :] = u[0, :] + tau * psi + (tau**2 / 2) * a**2 * (-u[0, :])
    
    sigma = (a * tau / h)**2
    for n in range(1, N):
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + sigma * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        # Граничные условия 1-го порядка
        if bc == 1:
            u[n+1, 0] = u[n+1, 1] / (1 + h)
            u[n+1, -1] = u[n+1, -2] / (1 - h)
        elif bc ==2 :
            u[n+1, 0] = (4 * u[n+1, 1] - u[n+1, 2]) / (3 + 2 * h)
            u[n+1, -1] = (4 * u[n+1, -2] - u[n+1, -3]) / (3 - 2 * h)
    return u

def yavn_bc2(x, tau, h, N, M, a):
    """2-й порядок, трехточечная аппроксимация"""
    u = np.zeros((N + 1, M + 1))
    u[0, :] = np.sin(x) + np.cos(x)
    psi = -a * (np.sin(x) + np.cos(x))
    u[1, :] = u[0, :] + tau * psi + (tau**2 / 2) * a**2 * (-u[0, :])
    
    sigma = (a * tau / h)**2
    for n in range(1, N):
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + sigma * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        # Граничные условия 2-го порядка (трехточечные)
        u[n+1, 0] = (4 * u[n+1, 1] - u[n+1, 2]) / (3 + 2 * h)
        u[n+1, -1] = (4 * u[n+1, -2] - u[n+1, -3]) / (3 - 2 * h)
    return u

def yavn_bc3(x, tau, h, N, M, a):
    """Явная схема с двухточечной аппроксимацией граничных условий 2-го порядка (через Тейлора и уравнение)"""
    u = np.zeros((N + 1, M + 1))
    u[0, :] = np.sin(x) + np.cos(x)
    psi = -a * (np.sin(x) + np.cos(x))
    u[1, :] = u[0, :] + tau * psi + (tau**2 / 2) * a**2 * (-u[0, :])
    
    sigma = (a * tau / h)**2
    
    for n in range(1, N):
        # 1. Внутренние узлы (стандартная схема "крест")
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + sigma * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        
        # 2. Граничные условия 2-го порядка (двухточечные)
        # Нам нужно значение u_tt на границе. Берем его с предыдущего временного слоя n.
        if n == 1:
            # Для n=1 используем точное значение из начального условия: u_tt = -a^2 * u
            u_tt_0 = -a**2 * u[1, 0]
            u_tt_M = -a**2 * u[1, -1]
        else:
            # Для n > 1 используем центральную разность по времени на границе
            u_tt_0 = (u[n, 0] - 2*u[n-1, 0] + u[n-2, 0]) / (tau**2)
            u_tt_M = (u[n, -1] - 2*u[n-1, -1] + u[n-2, -1]) / (tau**2)
        
        # Левая граница (x = 0)
        u[n+1, 0] = (u[n+1, 1] - (h**2 / (2 * a**2)) * u_tt_0) / (1 + h)
        
        # Правая граница (x = L)
        u[n+1, -1] = (u[n+1, -2] - (h**2 / (2 * a**2)) * u_tt_M) / (1 - h)

    return u



u = yavn(x, tau, h, N, M, a)




# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================
# Выбираем несколько моментов времени для сравнения
t_plot_indices = [0, N // 4, N // 2, 3 * N // 4, N - 1]

plt.figure(figsize=(12, 8))

for idx in t_plot_indices:
    t_val = t[idx]
    # Численное решение
    u_num = u[idx, :]
    # Точное решение
    u_an = u_exact(x, t_val, a)
    
    plt.plot(x, u_num, 'o-', label=f'Численное, t={t_val:.2f}', markersize=3)
    plt.plot(x, u_an, '--', label=f'Точное, t={t_val:.2f}', linewidth=2)

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Сравнение численного и аналитического решений')
plt.legend()
plt.grid(True)
plt.savefig("6_analit_vs_count.png", dpi=300, bbox_inches='tight')


# ============================================================
# ВЫЧИСЛЕНИЕ ПОГРЕШНОСТИ
# ============================================================
# Максимальная погрешность (C-норма) на каждом временном слое
error_max = np.zeros(N + 1)
for n in range(N + 1):
    u_an = u_exact(x, t[n], a)
    error_max[n] = np.max(np.abs(u[n, :] - u_an))

# График погрешности от времени
plt.figure(figsize=(10, 6))
plt.plot(t, error_max, 'b-', linewidth=2)
plt.xlabel('Время t')
plt.ylabel('Максимальная погрешность')
plt.title('Зависимость погрешности от времени')
plt.grid(True)
plt.savefig("6_error_from_time.png", dpi=300, bbox_inches='tight')

print(f"Максимальная погрешность за всё время: {np.max(error_max):.6e}")
print(f"Погрешность в конечный момент t={T}: {error_max[N]:.6e}")






# ============================================================
# ИССЛЕДОВАНИЕ СХОДИМОСТИ (ЗАВИСИМОСТЬ ПОГРЕШНОСТИ ОТ ШАГА)
# ============================================================
print("\n--- Исследование сходимости схемы ---")

M_values = [25, 50, 100, 200]  # Разные количества узлов по пространству
errors = []
h_values = []

for M_test in M_values:
    h_test = L / M_test
    tau_est = 0.9 * h_test / a
    N_test = int(T / tau_est)
    tau_test = T / N_test  # Пересчет для точного попадания в T
    
    x_test = np.linspace(0, L, M_test + 1)
    t_test = np.linspace(0, T, N_test + 1)
    
    # Вызываем нашу готовую функцию!
    u_test = yavn(x_test, tau_test, h_test, N_test, M_test, a)
    
    # Считаем максимальную погрешность за всё время
    max_err = 0.0
    for n in range(N_test + 1):
        u_an = u_exact(x_test, t_test[n], a)
        err = np.max(np.abs(u_test[n, :] - u_an))
        if err > max_err:
            max_err = err
            
    errors.append(max_err)
    h_values.append(h_test)
    print(f"M = {M_test:4d}, h = {h_test:.5f}, Макс. погрешность = {max_err:.6e}")

# Построение графика в логарифмическом масштабе
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'ro-', linewidth=2, markersize=8, label='Погрешность схемы')

# Эталонная линия для 2-го порядка O(h^2)
ref_line = [errors[0] * (h / h_values[0])**2 for h in h_values]
plt.loglog(h_values, ref_line, 'b--', linewidth=2, label='Эталон O(h^2)')
plt.gca().invert_xaxis()
plt.xlabel('Шаг по пространству h (логарифмический масштаб)')
plt.ylabel('Максимальная погрешность (логарифмический масштаб)')
plt.title('Исследование порядка аппроксимации схемы')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("6_convergence.png", dpi=300, bbox_inches='tight')


# ============================================================
# СРАВНЕНИЕ ЯВНОЙ И НЕЯВНОЙ СХЕМ
# ============================================================
# Считаем неявную схему
u_impl = neayvn(x, tau, h, N, M, a)

# Считаем погрешность неявной схемы
error_impl = np.zeros(N + 1)
for n in range(N + 1):
    u_an = u_exact(x, t[n], a)
    error_impl[n] = np.max(np.abs(u_impl[n, :] - u_an))

print(f"Макс. погрешность явной схемы:   {np.max(error_max):.6e}")
print(f"Макс. погрешность неявной схемы: {np.max(error_impl):.6e}")

# График сравнения погрешностей
plt.figure(figsize=(10, 6))
plt.plot(t, error_max, 'b-', linewidth=2, label='Явная схема')
plt.plot(t, error_impl, 'r--', linewidth=2, label='Неявная схема')
plt.xlabel('Время t')
plt.ylabel('Максимальная погрешность')
plt.title('Сравнение погрешности явной и неявной схем')
plt.legend()
plt.grid(True)
plt.savefig("6_neyavn_err.png", dpi=300, bbox_inches='tight')

# ============================================================
# ПРОВЕРКА УСТОЙЧИВОСТИ (НАРУШЕНИЕ УСЛОВИЯ КУРАНТА)
# ============================================================
# Намеренно берем большой шаг по времени (нарушаем условие tau <= h/a)
tau_big = 1.5 * h / a 
N_big = int(T / tau_big)
tau_big = T / N_big # пересчет для точного попадания в T

print(f"\n--- Тест устойчивости (tau = {tau_big:.4f}, h = {h:.4f}) ---")

# Явная схема "взрывается"
u_exp_big = yavn(x, tau_big, h, N_big, M, a)
err_exp_big = np.max([np.max(np.abs(u_exp_big[n, :] - u_exact(x, t[n], a))) for n in range(N_big + 1)])

# Неявная схема остается стабильной
u_impl_big = neayvn(x, tau_big, h, N_big, M, a)
err_impl_big = np.max([np.max(np.abs(u_impl_big[n, :] - u_exact(x, t[n], a))) for n in range(N_big + 1)])

print(f"Погрешность явной схемы (большой tau):   {err_exp_big:.6e}  <-- ВЗРЫВ!")
print(f"Погрешность неявной схемы (большой tau): {err_impl_big:.6e} <-- Стабильно")

# Визуализация "взрыва"
plt.figure(figsize=(10, 6))
plt.plot(x, u_exp_big[N_big, :], 'r-', linewidth=2, label='Явная (нестабильная)')
plt.plot(x, u_impl_big[N_big, :], 'b--', linewidth=2, label='Неявная (стабильная)')
plt.plot(x, u_exact(x, T, a), 'g:', linewidth=2, label='Точное решение')
plt.title('Поведение схем при нарушении условия Куранта')
plt.legend()
plt.grid(True)
plt.savefig("6_blow_up.png", dpi=300, bbox_inches='tight')


# ============================================================
# СРАВНЕНИЕ ТРЕХ ВИДОВ АППРОКСИМАЦИИ ГРАНИЧНЫХ УСЛОВИЙ
# ============================================================
print("\n--- Сравнение трех видов аппроксимации граничных условий ---")

# Считаем все три варианта (функции yavn_bc1 и yavn_bc2 должны быть в твоем коде)
u_bc1 = yavn_bc12(x, tau, h, N, M, a) # 1-й порядок, двухточечная
u_bc2 = yavn_bc12(x, tau, h, N, M, a, 2) # 2-й порядок, трехточечная
u_bc3 = yavn_bc3(x, tau, h, N, M, a) # 2-й порядок, двухточечная (Тейлор)

# Считаем максимальные погрешности
err_bc1 = np.max([np.max(np.abs(u_bc1[n, :] - u_exact(x, t[n], a))) for n in range(N + 1)])
err_bc2 = np.max([np.max(np.abs(u_bc2[n, :] - u_exact(x, t[n], a))) for n in range(N + 1)])
err_bc3 = np.max([np.max(np.abs(u_bc3[n, :] - u_exact(x, t[n], a))) for n in range(N + 1)])

print(f"1. Двухточечная, 1-й порядок:     {err_bc1:.6e}")
print(f"2. Трехточечная, 2-й порядок:     {err_bc2:.6e}")
print(f"3. Двухточечная, 2-й порядок:     {err_bc3:.6e}")