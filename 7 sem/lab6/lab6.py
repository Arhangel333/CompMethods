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