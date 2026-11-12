import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# НАСТРОЙКИ ЗАДАЧИ
# =============================================================================
class ProblemConfig:
    """Конфигурация краевой задачи. Измените эти значения для другого варианта."""
    
    # Область определения
    x_min, x_max = 0.0, np.pi
    y_min, y_max = 0.0, 1.0
    
    # Шаги сетки (для исследования зависимости погрешности можно менять)
    hx_default = np.pi / 20   # шаг по x
    hy_default = 1.0 / 20     # шаг по y
    
    # Граничные условия Дирихле (значения функции)
    @staticmethod
    def u_bottom(x):      # y = y_min
        return np.sin(x)
    
    @staticmethod
    def u_top(x):         # y = y_max
        return np.e * np.sin(x)
    
    # Граничные условия Неймана (производные по нормали)
    # Для вертикальных границ нормаль совпадает с осью x
    @staticmethod
    def du_dx_left(y):    # x = x_min, du/dx
        return np.exp(y)
    
    @staticmethod
    def du_dx_right(y):   # x = x_max, du/dx
        return -np.exp(y)
    
    # Аналитическое решение для контроля точности
    @staticmethod
    def exact_solution(x, y):
        return np.sin(x) * np.exp(y)

# =============================================================================
# ПАРАМЕТРЫ ИТЕРАЦИОННЫХ МЕТОДОВ
# =============================================================================
ITER_PARAMS = {
    'tolerance': 1e-8,       # Точность остановки итераций
    'max_iter': 50000,       # Максимальное число итераций
    'omega_sor': 1.7,        # Параметр релаксации для SOR (1 < omega < 2)
}

# =============================================================================
# РЕАЛИЗАЦИЯ ЧИСЛЕННЫХ МЕТОДОВ
# =============================================================================
def build_grid(config, hx=None, hy=None):
    """Построение разностной сетки и задание граничных условий."""
    hx = hx or config.hx_default
    hy = hy or config.hy_default
    
    Nx = int(round((config.x_max - config.x_min) / hx))
    Ny = int(round((config.y_max - config.y_min) / hy))
    hx = (config.x_max - config.x_min) / Nx
    hy = (config.y_max - config.y_min) / Ny
    
    x = np.linspace(config.x_min, config.x_max, Nx + 1)
    y = np.linspace(config.y_min, config.y_max, Ny + 1)
    
    # Начальное приближение: линейная интерполяция по y между нижней и верхней границами
    U = np.zeros((Nx + 1, Ny + 1))
    for i in range(Nx + 1):
        bottom = config.u_bottom(x[i])
        top = config.u_top(x[i])
        U[i, :] = bottom + (top - bottom) * (y - config.y_min) / (config.y_max - config.y_min)
    
    # Жестко фиксируем условия Дирихле
    U[:, 0] = config.u_bottom(x)       # нижняя граница
    U[:, Ny] = config.u_top(x)          # верхняя граница
    
    return x, y, U, Nx, Ny, hx, hy


def apply_neumann_boundaries(U, x, y, config, hx, hy):
    """
    Аппроксимация условий Неймана методом фиктивных узлов.
    
    
    Корректно работает при ЛЮБЫХ hx и hy согласно п. 5.1.3 и Примеру 5.3 методички.
    Сохраняет порядок O(hx^2 + hy^2).
    """
    Nx, Ny = U.shape[0] - 1, U.shape[1] - 1
    
    # Коэффициенты разностного оператора Лапласа
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    denom = 2.0 * (cx + cy)
    
    # === Левая граница x=0 ===
    # Фиктивный узел: u_{-1,j} = u_{1,j} - 2*hx*phi_left(y_j)
    # Подстановка в уравнение Лапласа: 
    # cx*(u_{1}-2u_0+u_{-1}) + cy*(u_{j+1}-2u_0+u_{j-1}) = 0
    # => 2*cx*u_{1} + cy*(u_{j+1}+u_{j-1}) - 2*cx*hx*phi = denom*u_0
    for j in range(1, Ny):
        phi = config.du_dx_left(y[j])
        U[0, j] = (2 * cx * U[1, j] + cy * (U[0, j+1] + U[0, j-1]) - 2 * cx * hx * phi) / denom
    
    # === Правая граница x=pi ===
    # Фиктивный узел: u_{N+1,j} = u_{N-1,j} + 2*hx*phi_right(y_j)
    for j in range(1, Ny):
        phi = config.du_dx_right(y[j])
        U[Nx, j] = (2 * cx * U[Nx-1, j] + cy * (U[Nx, j+1] + U[Nx, j-1]) + 2 * cx * hx * phi) / denom
    
    # === Угловые узлы (смешанные условия) ===
    # Нижний левый (0,0): Нейман слева + Дирихле снизу
    phi_l = config.du_dx_left(y[0])
    U[0, 0] = (2 * cx * U[1, 0] + cy * (U[0, 1] + config.u_bottom(x[0])) - 2 * cx * hx * phi_l) / denom
    
    # Верхний левый (0,Ny): Нейман слева + Дирихле сверху
    phi_l = config.du_dx_left(y[Ny])
    U[0, Ny] = (2 * cx * U[1, Ny] + cy * (config.u_top(x[0]) + U[0, Ny-1]) - 2 * cx * hx * phi_l) / denom
    
    # Нижний правый (Nx,0): Нейман справа + Дирихле снизу
    phi_r = config.du_dx_right(y[0])
    U[Nx, 0] = (2 * cx * U[Nx-1, 0] + cy * (U[Nx, 1] + config.u_bottom(x[Nx])) + 2 * cx * hx * phi_r) / denom
    
    # Верхний правый (Nx,Ny): Нейман справа + Дирихле сверху
    phi_r = config.du_dx_right(y[Ny])
    U[Nx, Ny] = (2 * cx * U[Nx-1, Ny] + cy * (config.u_top(x[Nx]) + U[Nx, Ny-1]) + 2 * cx * hx * phi_r) / denom

def solve_libman(config, hx=None, hy=None):
    """Метод простых итераций (Либман)."""
    x, y, U, Nx, Ny, hx, hy = build_grid(config, hx, hy)
    tol = ITER_PARAMS['tolerance']
    max_iter = ITER_PARAMS['max_iter']
    
    # Коэффициенты для уравнения Лапласа при разных шагах
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    denom = 2.0 * (cx + cy)
    
    for k in range(max_iter):
        U_old = U.copy()
        
        # Обновляем внутренние узлы с учетом разных шагов
        for i in range(1, Nx):
            for j in range(1, Ny):
                U[i, j] = (cx * (U_old[i+1, j] + U_old[i-1, j]) + 
                           cy * (U_old[i, j+1] + U_old[i, j-1])) / denom
        
        # Применяем условия Неймана на вертикальных границах
        apply_neumann_boundaries(U, x, y, config, hx, hy)
        
        # Проверка сходимости
        err = np.max(np.abs(U - U_old))
        if err < tol:
            print(f"  Либман: сошелся за {k+1} итераций, невязка={err:.2e}")
            break
    else:
        print(f"  Либман: НЕ сошелся за {max_iter} итераций")
    
    return x, y, U


def solve_seidel(config, hx=None, hy=None):
    """Метод Зейделя (использует уже обновленные значения текущей итерации)."""
    x, y, U, Nx, Ny, hx, hy = build_grid(config, hx, hy)
    tol = ITER_PARAMS['tolerance']
    max_iter = ITER_PARAMS['max_iter']
    
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    denom = 2.0 * (cx + cy)
    
    for k in range(max_iter):
        U_old = U.copy()
        
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Используем U[i-1,j] и U[i,j-1] — уже обновленные на этой итерации
                U[i, j] = (cx * (U[i+1, j] + U[i-1, j]) + 
                           cy * (U[i, j+1] + U[i, j-1])) / denom
        
        apply_neumann_boundaries(U, x, y, config, hx, hy)
        
        err = np.max(np.abs(U - U_old))
        if err < tol:
            print(f"  Зейдель: сошелся за {k+1} итераций, невязка={err:.2e}")
            break
    else:
        print(f"  Зейдель: НЕ сошелся за {max_iter} итераций")
    
    return x, y, U


def solve_sor(config, hx=None, hy=None):
    """Метод верхней релаксации (SOR)."""
    x, y, U, Nx, Ny, hx, hy = build_grid(config, hx, hy)
    tol = ITER_PARAMS['tolerance']
    max_iter = ITER_PARAMS['max_iter']
    omega = ITER_PARAMS['omega_sor']
    
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    denom = 2.0 * (cx + cy)
    
    for k in range(max_iter):
        U_old = U.copy()
        
        for i in range(1, Nx):
            for j in range(1, Ny):
                gauss_seidel_val = (cx * (U[i+1, j] + U[i-1, j]) + 
                                    cy * (U[i, j+1] + U[i, j-1])) / denom
                # Релаксация: усиливаем поправку
                U[i, j] = U[i, j] + omega * (gauss_seidel_val - U[i, j])
        
        apply_neumann_boundaries(U, x, y, config, hx, hy)
        
        err = np.max(np.abs(U - U_old))
        if err < tol:
            print(f"  SOR (ω={omega}): сошелся за {k+1} итераций, невязка={err:.2e}")
            break
    else:
        print(f"  SOR (ω={omega}): НЕ сошелся за {max_iter} итераций")
    
    return x, y, U


# =============================================================================
# АНАЛИЗ ПОГРЕШНОСТИ И СХОДИМОСТИ
# =============================================================================
def compute_error(U, x, y, config):
    """Вычисление максимальной погрешности относительно аналитического решения."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = config.exact_solution(X, Y)
    return np.max(np.abs(U - U_exact))


def convergence_study(config):
    """Исследование зависимости погрешности от шагов сетки."""
    print("\n" + "="*70)
    print("ИССЛЕДОВАНИЕ СХОДИМОСТИ")
    print("="*70)
    
    # Набор шагов для исследования (уменьшаем в 2 раза каждый раз)
    scale_factors = [1, 2, 4]
    
    results = {'hx': [], 'hy': [], 'libman_err': [], 'seidel_err': [], 'sor_err': []}
    
    for sf in scale_factors:
        hx = config.hx_default / sf
        hy = config.hy_default / sf
        
        print(f"\n--- Сетка: hx={hx:.5f}, hy={hy:.5f} ---")
        
        _, _, U_l = solve_libman(config, hx, hy)
        _, _, U_s = solve_seidel(config, hx, hy)
        _, _, U_o = solve_sor(config, hx, hy)
        
        e_l = compute_error(U_l, np.linspace(config.x_min, config.x_max, U_l.shape[0]),
                            np.linspace(config.y_min, config.y_max, U_l.shape[1]), config)
        e_s = compute_error(U_s, np.linspace(config.x_min, config.x_max, U_s.shape[0]),
                            np.linspace(config.y_min, config.y_max, U_s.shape[1]), config)
        e_o = compute_error(U_o, np.linspace(config.x_min, config.x_max, U_o.shape[0]),
                            np.linspace(config.y_min, config.y_max, U_o.shape[1]), config)
        
        results['hx'].append(hx)
        results['hy'].append(hy)
        results['libman_err'].append(e_l)
        results['seidel_err'].append(e_s)
        results['sor_err'].append(e_o)
        
        print(f"  Погрешность Либман: {e_l:.6e}")
        print(f"  Погрешность Зейдель: {e_s:.6e}")
        print(f"  Погрешность SOR:     {e_o:.6e}")
    
    # Построение графика сходимости
    plt.figure(figsize=(10, 6))
    plt.loglog(results['hx'], results['libman_err'], 'bo-', label='Либман', linewidth=2)
    plt.loglog(results['hx'], results['seidel_err'], 'rs-', label='Зейдель', linewidth=2)
    plt.loglog(results['hx'], results['sor_err'], 'g^-', label=f'SOR (ω={ITER_PARAMS["omega_sor"]})', linewidth=2)
    
    # Эталонная линия O(h^2)
    ref_h = np.array(results['hx'])
    ref_err = results['libman_err'][0] * (ref_h / results['hx'][0])**2
    plt.loglog(ref_h, ref_err, 'k--', alpha=0.5, label='O(h²) эталон')
    
    plt.xlabel('Шаг сетки h', fontsize=12)
    plt.ylabel('Максимальная погрешность', fontsize=12)
    plt.title('Зависимость погрешности от шага сетки', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    
     # === РАСЧЕТ НАКЛОНА (ПОРЯДКА ТОЧНОСТИ) ===
    print("\n" + "="*70)
    print("РАСЧЕТ ПОРЯДКА СХОДИМОСТИ (НАКЛОНА ПРЯМОЙ)")
    print("="*70)
    
    for i in range(len(results['hx']) - 1):
        h1, h2 = results['hx'][i], results['hx'][i+1]
        e_l1, e_l2 = results['libman_err'][i], results['libman_err'][i+1]
        e_s1, e_s2 = results['seidel_err'][i], results['seidel_err'][i+1]
        e_o1, e_o2 = results['sor_err'][i], results['sor_err'][i+1]
        
        # Формула наклона: log(E1/E2) / log(h1/h2)
        p_libman = np.log(e_l1 / e_l2) / np.log(h1 / h2)
        p_seidel = np.log(e_s1 / e_s2) / np.log(h1 / h2)
        p_sor = np.log(e_o1 / e_o2) / np.log(h1 / h2)
        
        print(f"Переход h={h1:.5f} -> h={h2:.5f}:")
        print(f"  Наклон (порядок) Либман:  {p_libman:.2f}")
        print(f"  Наклон (порядок) Зейдель: {p_seidel:.2f}")
        print(f"  Наклон (порядок) SOR:     {p_sor:.2f}")
    
    return results


# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ
# =============================================================================
def plot_solution(x, y, U, config, method_name):
    """Построение поверхности численного решения и ошибки."""
    # Создаем координатные матрицы
    X, Y = np.meshgrid(x, y, indexing='xy')  # 'xy' — стандартная ориентация
    
    # Вычисляем аналитическое решение
    U_exact = config.exact_solution(X, Y)
    
    # Транспонируем U, чтобы согласовать с X и Y
    U_plot = U.T
    U_exact_plot = U_exact.T
    Error_plot = np.abs(U_plot - U_exact_plot.T)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Численное решение
    im1 = axes[0].pcolormesh(X, Y, U_plot, shading='auto', cmap='hot')
    axes[0].set_title(f'{method_name}: численное решение', fontsize=12)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Аналитическое решение
    im2 = axes[1].pcolormesh(X, Y, U_exact_plot.T, shading='auto', cmap='hot')
    axes[1].set_title('Аналитическое решение', fontsize=12)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Погрешность
    im3 = axes[2].pcolormesh(X, Y, Error_plot, shading='auto', cmap='inferno')
    max_err = np.max(Error_plot)
    axes[2].set_title(f'Погрешность (max={max_err:.2e})', fontsize=12)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Результаты метода: {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'solution_{method_name}.png', dpi=150)


# =============================================================================
# ГЛАВНЫЙ ЗАПУСК
# =============================================================================
if __name__ == '__main__':
    config = ProblemConfig()
    
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА 3: Краевая задача для уравнения Лапласа")
    print("="*70)
    print(f"Область: x∈[{config.x_min}, {config.x_max}], y∈[{config.y_min}, {config.y_max}]")
    print(f"Сетка по умолчанию: hx={config.hx_default:.4f}, hy={config.hy_default:.4f}")
    print(f"Точность итераций: {ITER_PARAMS['tolerance']}")
    print(f"Параметр SOR: ω={ITER_PARAMS['omega_sor']}")
    
    # 1. Решаем тремя методами на базовой сетке
    print("\n--- Решение на базовой сетке ---")
    x, y, U_libman = solve_libman(config)
    x, y, U_seidel = solve_seidel(config)
    x, y, U_sor = solve_sor(config)
    
    # Визуализация для SOR (как наиболее точного/быстрого)
    plot_solution(x, y, U_sor, config, "SOR")
    
    # 2. Исследование сходимости
    convergence_study(config)
    
    print("\n✅ Все расчёты завершены.")