import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# 1. ПАРАМЕТРЫ ЗАДАЧИ И ТОЧНОЕ РЕШЕНИЕ
# =============================================================================
A    = 1.0            
XMAX = np.pi / 4      
YMAX = np.log(2.0)    
T    = 0.1            

def u_exact(x, y, t):  return np.cos(2*x)*np.cosh(y)*np.exp(-3*A*t)
def bc_left(y, t):     return np.cosh(y)*np.exp(-3*A*t)
def bc_right(y, t):    return np.zeros_like(y)
def bc_bottom(x, t):   return np.cos(2*x)*np.exp(-3*A*t)
def flux_top(x, t):    return 0.75*np.cos(2*x)*np.exp(-3*A*t)

def set_dirichlet(u, xs, ys, t):
    u[0, :], u[-1, :], u[:, 0] = bc_left(ys, t), bc_right(ys, t), bc_bottom(xs, t)

# =============================================================================
# 2. ПРОГОНКА (ТОЛЬКО ДЛЯ НЕЯВНЫХ МЕТОДОВ)
# =============================================================================
def thomas_batch(sub, diag, sup, rhs):
    n, m = diag.size, rhs.shape[1]
    c = np.zeros((n, m)); d = np.zeros((n, m))
    c[0] = sup[0]/diag[0]; d[0] = rhs[0]/diag[0]
    for i in range(1, n):
        den  = diag[i] - sub[i]*c[i-1]
        c[i] = sup[i]/den if i < n-1 else 0.0
        d[i] = (rhs[i] - sub[i]*d[i-1])/den
    x = np.zeros_like(d); x[-1] = d[-1]
    for i in range(n-2, -1, -1):
        x[i] = d[i] - c[i]*x[i+1]
    return x

def make_matrices(mx, my, sx, sy):
    subx = np.full(mx, -sx); subx[0] = 0.0
    supx = np.full(mx, -sx); supx[-1] = 0.0
    suby = np.full(my, -sy); suby[0] = 0.0; suby[-1] = -2*sy
    supy = np.full(my, -sy); supy[-1] = 0.0
    return ((subx, np.full(mx, 1+2*sx), supx), (suby, np.full(my, 1+2*sy), supy))

# =============================================================================
# 3. ТРИ МЕТОДА: МПН, МДШ, ЯВНЫЙ
# =============================================================================

# --- Метод переменных направлений (МПН) ---
def step_adi(u, xs, ys, hx, hy, tau, t):
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    mx, my = Nx-1, Ny
    sx, sy = A*tau/(2*hx**2), A*tau/(2*hy**2)
    Mx, My = make_matrices(mx, my, sx, sy)
    th, tn = t + tau/2, t + tau

    uh = u.copy()
    set_dirichlet(uh, xs, ys, th)
    rhs = np.empty((mx, my))
    rhs[:, :my-1] = u[1:Nx, 1:Ny] + sy*(u[1:Nx, 0:Ny-1] - 2*u[1:Nx, 1:Ny] + u[1:Nx, 2:Ny+1])
    ghost_k = u[1:Nx, Ny-1] + 2*hy*flux_top(xs[1:Nx], t)
    rhs[:, my-1] = u[1:Nx, Ny] + sy*(u[1:Nx, Ny-1] - 2*u[1:Nx, Ny] + ghost_k)
    rhs[0, :]  += sx * uh[0, 1:Ny+1]
    rhs[-1, :] += sx * uh[Nx, 1:Ny+1]
    uh[1:Nx, 1:Ny+1] = thomas_batch(*Mx, rhs)

    r2 = np.empty((my, mx))
    r2[:my-1, :] = uh[1:Nx, 1:Ny].T + sx*(uh[0:Nx-1, 1:Ny].T - 2*uh[1:Nx, 1:Ny].T + uh[2:Nx+1, 1:Ny].T)
    r2[0, :] += sy * bc_bottom(xs[1:Nx], tn)
    r2[my-1, :] = uh[1:Nx, Ny].T + sx*(uh[0:Nx-1, Ny].T - 2*uh[1:Nx, Ny].T + uh[2:Nx+1, Ny].T) + 2*sy*hy*flux_top(xs[1:Nx], tn)
    
    un = u.copy()
    un[1:Nx, 1:Ny+1] = thomas_batch(*My, r2).T
    set_dirichlet(un, xs, ys, tn)
    return un

# --- Метод дробных шагов (МДШ) ---
def step_fds(u, xs, ys, hx, hy, tau, t):
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    mx, my = Nx-1, Ny
    rx, ry = A*tau/hx**2, A*tau/hy**2
    Mx, My = make_matrices(mx, my, rx, ry)
    tn = t + tau

    uh = u.copy()
    rhs = u[1:Nx, 1:Ny+1].copy()
    rhs[0, :]  += rx * bc_left(ys[1:Ny+1], tn)
    rhs[-1, :] += rx * bc_right(ys[1:Ny+1], tn)
    uh[1:Nx, 1:Ny+1] = thomas_batch(*Mx, rhs)
    set_dirichlet(uh, xs, ys, tn)

    r2 = uh[1:Nx, 1:Ny+1].T.copy()
    r2[0, :]    += ry * bc_bottom(xs[1:Nx], tn)
    r2[my-1, :] += 2*ry*hy*flux_top(xs[1:Nx], tn)
    
    un = u.copy()
    un[1:Nx, 1:Ny+1] = thomas_batch(*My, r2).T
    set_dirichlet(un, xs, ys, tn)
    return un

# --- Явный метод ---
def step_explicit(u, xs, ys, hx, hy, tau, t):
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    rx, ry = A*tau/hx**2, A*tau/hy**2
    tn = t + tau
    
    un = u.copy()
    
    # 1. Внутренние узлы (стандартная явная схема)
    un[1:Nx, 1:Ny] = u[1:Nx, 1:Ny] + \
                     rx * (u[0:Nx-1, 1:Ny] - 2*u[1:Nx, 1:Ny] + u[2:Nx+1, 1:Ny]) + \
                     ry * (u[1:Nx, 0:Ny-1] - 2*u[1:Nx, 1:Ny] + u[1:Nx, 2:Ny+1])
                     
    # 2. Верхняя граница (Нейман) через ФИКТИВНЫЙ УЗЕЛ (2-й порядок!)
    # Центр. разность: (u[Ny+1] - u[Ny-1]) / (2*hy) = flux
    # => u[Ny+1] = u[Ny-1] + 2*hy*flux
    # Подставляем в уравнение для j=Ny:
    # u_new[Ny] = u[Ny] + rx*(...) + ry*(u[Ny-1] - 2*u[Ny] + u[Ny+1])
    # После подстановки u[Ny+1]: ry*(2*u[Ny-1] - 2*u[Ny] + 2*hy*flux)
    flux_vals = flux_top(xs[1:Nx], tn)
    un[1:Nx, Ny] = u[1:Nx, Ny] + \
                   rx * (u[0:Nx-1, Ny] - 2*u[1:Nx, Ny] + u[2:Nx+1, Ny]) + \
                   ry * (2*u[1:Nx, Ny-1] - 2*u[1:Nx, Ny] + 2*hy*flux_vals)
    
    set_dirichlet(un, xs, ys, tn)
    return un

# =============================================================================
# 4. МАРШ ПО ВРЕМЕНИ С ЗАМЕРОМ ВРЕМЕНИ И КОНТРОЛЕМ УСТОЙЧИВОСТИ
# =============================================================================
def run(method, Nx, Ny, tau, T):
    xs = np.linspace(0, XMAX, Nx+1); ys = np.linspace(0, YMAX, Ny+1)
    hx, hy = xs[1]-xs[0], ys[1]-ys[0]
    u = u_exact(xs[:, None], ys[None, :], 0.0)
    
    if method == 'adi': step = step_adi
    elif method == 'fds': step = step_fds
    elif method == 'exp': step = step_explicit
        
    K = int(round(T/tau))
    
    start_time = time.perf_counter()
    for k in range(K):
        u = step(u, xs, ys, hx, hy, tau, k*tau)
        # Проверка на "взрыв" (потерю устойчивости)
        if np.any(np.isnan(u)) or np.any(np.isinf(u)) or np.max(np.abs(u)) > 1e10:
            end_time = time.perf_counter()
            return {'end': np.inf, 'time': end_time - start_time, 'unstable': True}
            
    end_time = time.perf_counter()
    err = np.max(np.abs(u - u_exact(xs[:, None], ys[None, :], T)))
    return {'end': err, 'time': end_time - start_time, 'unstable': False}

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ (3D, ТЕПЛОВЫЕ КАРТЫ, ОШИБКИ)
# =============================================================================
def plot_solution_3d(xs, ys, u, title, ax):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ax.plot_surface(X, Y, u, cmap='viridis', alpha=0.9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_title(title)

def plot_heatmap(xs, ys, u, title, ax):
    im = ax.pcolormesh(xs, ys, u.T, cmap='hot', shading='auto')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)
    return im

def plot_solutions_comparison():
    Nx, Ny = 32, 32
    # Для явного метода берем шаг из условия устойчивости tau <= h^2 / (4A)
    hx = XMAX / Nx
    tau_exp = (hx**2) / (6 * A)
    
    # Для неявных берем адекватный шаг
    tau_imp = T / 100 
    
    snapshots_imp = [int(T/tau_imp)]
    snapshots_exp = [int(T/tau_exp)]
    
    print("\nРасчёт снимков решения...")
    xs, ys, snaps_adi = run_with_snapshots('adi', Nx, Ny, tau_imp, T, snapshots_imp)
    _, _, snaps_fds   = run_with_snapshots('fds', Nx, Ny, tau_imp, T, snapshots_imp)
    _, _, snaps_exp   = run_with_snapshots('exp', Nx, Ny, tau_exp, T, snapshots_exp)
    
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_analytic = u_exact(X, Y, T)

    fig = plt.figure(figsize=(24, 5))
    data_list = [(u_analytic, 'Точное'), (snaps_adi[snapshots_imp[-1]], 'МПН'), 
                 (snaps_fds[snapshots_imp[-1]], 'МДШ'), (snaps_exp[snapshots_exp[-1]], 'Явный')]
    for idx, (data, title) in enumerate(data_list):
        ax = fig.add_subplot(1, 4, idx+1, projection='3d')
        plot_solution_3d(xs, ys, data, title, ax)
    plt.suptitle(f'Сравнение 3D решений при t={T}', fontsize=16, y=1.05)
    plt.tight_layout(); plt.savefig('comparison_3d.png', dpi=150, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for idx, (data, title) in enumerate(data_list):
        im = plot_heatmap(xs, ys, data, title, axes[idx])
        plt.colorbar(im, ax=axes[idx])
    plt.suptitle(f'Тепловые карты при t={T}', fontsize=16, y=1.02)
    plt.tight_layout(); plt.savefig('comparison_heatmaps.png', dpi=150, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    err_adi = np.abs(snaps_adi[snapshots_imp[-1]] - u_analytic)
    err_fds = np.abs(snaps_fds[snapshots_imp[-1]] - u_analytic)
    err_exp = np.abs(snaps_exp[snapshots_exp[-1]] - u_analytic)
    vmax = max(np.max(err_adi), np.max(err_fds), np.max(err_exp))
    
    for idx, (err, title) in enumerate([(err_adi, 'Ошибка МПН'), (err_fds, 'Ошибка МДШ'), (err_exp, 'Ошибка Явного')]):
        im = axes[idx].pcolormesh(xs, ys, err.T, cmap='Reds', shading='auto', vmin=0, vmax=vmax)
        axes[idx].set_title(f'{title} (max={np.max(err):.3e})')
        plt.colorbar(im, ax=axes[idx])
    plt.tight_layout(); plt.savefig('errors.png', dpi=150, bbox_inches='tight'); plt.close()

def run_with_snapshots(method, Nx, Ny, tau, T, snapshots):
    xs = np.linspace(0, XMAX, Nx+1); ys = np.linspace(0, YMAX, Ny+1)
    hx, hy = xs[1]-xs[0], ys[1]-ys[0]
    u = u_exact(xs[:, None], ys[None, :], 0.0)
    if method == 'adi': step = step_adi
    elif method == 'fds': step = step_fds
    elif method == 'exp': step = step_explicit
        
    K = int(round(T/tau)); snaps = {}
    for k in range(K):
        u = step(u, xs, ys, hx, hy, tau, k*tau)
        if k+1 in snapshots: snaps[k+1] = u.copy()
    return xs, ys, snaps

# =============================================================================
# 6. ГЛАВНЫЙ ГРАФИК СХОДИМОСТИ  + СРАВНЕНИЕ ВРЕМЕНИ
# =============================================================================
def plot_convergence_uniform_refinement():
    """График сходимости при равномерном измельчении tau и h"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Базовая сетка и шаг по времени
    N_base = 8
    tau_base = T / 20  # Достаточно крупный начальный шаг
    
    # Количество уровней измельчения (будет 5 точек: k=0..4)
    levels = [0, 1, 2, 3, 4]  
    
    err_adi, err_fds, err_exp = [], [], []
    h_vals, tau_vals = [], []
    
    print("\nИсследование сходимости (равномерное измельчение):")
    for k in levels:
        factor = 2**k
        N_curr = N_base * factor
        tau_curr = tau_base / factor
        
        h_curr = XMAX / N_curr
        h_vals.append(h_curr)
        tau_vals.append(tau_curr)
        
        res_adi = run('adi', N_curr, N_curr, tau_curr, T)
        res_fds = run('fds', N_curr, N_curr, tau_curr, T)
        res_exp = run('exp', N_curr, N_curr, tau_curr, T)
        
        err_adi.append(res_adi['end'])
        err_fds.append(res_fds['end'])
        err_exp.append(res_exp['end'] if not res_exp['unstable'] else np.nan)
        
        print(f"  Уровень {k}: N={N_curr}, τ={tau_curr:.5f} | "
              f"МПН ε={err_adi[-1]:.2e} | МДШ ε={err_fds[-1]:.2e} | "
              f"Явный ε={err_exp[-1]:.2e}")

    h_arr = np.array(h_vals)
    
    ax.loglog(h_arr, err_adi, 'o-', label='МПН', linewidth=2.5, markersize=8)
    ax.loglog(h_arr, err_fds, 's-', label='МДШ', linewidth=2.5, markersize=8)
    valid_exp = ~np.isnan(err_exp)
    ax.loglog(h_arr[valid_exp], np.array(err_exp)[valid_exp], 
              '^-', label='Явный', linewidth=2.5, markersize=8, color='green')
    
    # Эталон O(h²)
    ref = err_adi[0] * (h_arr / h_arr[0])**2
    ax.loglog(h_arr, ref, '--', color='gray', linewidth=1.5, label='O(h²) эталон')
    
    ax.set_xlabel('Шаг по пространству h', fontsize=13)
    ax.set_ylabel('Погрешность ‖u_h - U‖∞', fontsize=13)
    ax.set_title('Сходимость при равномерном измельчении τ и h\n'
                 '(τₖ = τ₀/2ᵏ, hₖ = h₀/2ᵏ)', fontsize=14)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Аннотации с численным порядком
    for errs, name, color in [(err_adi, 'МПН', 'blue'), 
                               (err_fds, 'МДШ', 'orange'),
                               (np.array(err_exp)[valid_exp], 'Явный', 'green')]:
        if len(errs) >= 2:
            p_num = np.log(errs[0]/errs[-1]) / np.log(h_arr[0]/h_arr[len(errs)-1])
            mid_idx = len(errs) // 2
            ax.annotate(f'{name}: O(h^{p_num:.2f})', 
                       xy=(h_arr[mid_idx], errs[mid_idx]),
                       xytext=(h_arr[mid_idx]*2.0, errs[mid_idx]*0.3),
                       fontsize=11, color=color, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    plt.tight_layout()
    plt.savefig('convergence_uniform.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ График сохранён: convergence_uniform.png")

def plot_convergence_adi_vs_fds():
    """Чистое сравнение сходимости МПН и МДШ при равномерном измельчении"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    N_base = 8
    tau_base = T / 20
    levels = [0, 1, 2, 3, 4]
    
    err_adi, err_fds = [], []
    h_vals = []
    
    print("\nИсследование сходимости (МПН vs МДШ):")
    for k in levels:
        factor = 2**k
        N_curr = N_base * factor
        tau_curr = tau_base / factor
        h_curr = XMAX / N_curr
        h_vals.append(h_curr)
        
        res_adi = run('adi', N_curr, N_curr, tau_curr, T)
        res_fds = run('fds', N_curr, N_curr, tau_curr, T)
        
        err_adi.append(res_adi['end'])
        err_fds.append(res_fds['end'])
        
        print(f"  Уровень {k}: N={N_curr}, τ={tau_curr:.5f} | "
              f"МПН ε={err_adi[-1]:.2e} | МДШ ε={err_fds[-1]:.2e}")

    h_arr = np.array(h_vals)
    
    # Отрисовка линий
    ax.loglog(h_arr, err_adi, 'o-', label='МПН (τ∝h)', 
              linewidth=2.5, markersize=9, color='#1f77b4')
    ax.loglog(h_arr, err_fds, 's-', label='МДШ (τ∝h²)', 
              linewidth=2.5, markersize=9, color='#ff7f0e')
    
    # Эталон O(h²) — привязан к МПН как наиболее точному
    ref = err_adi[0] * (h_arr / h_arr[0])**2
    ax.loglog(h_arr, ref, '--', color='gray', linewidth=1.5, label='O(h²) эталон')
    
    # Автоматический масштаб по Y с запасом
    all_errs = err_adi + err_fds
    ax.set_ylim(min(all_errs) * 0.3, max(all_errs) * 3.0)
    
    ax.set_xlabel('Шаг по пространству h', fontsize=14)
    ax.set_ylabel('Погрешность ‖u_h - U‖∞', fontsize=14)
    ax.set_title('Сходимость методов расщепления\n'
                 '(равномерное измельчение: τₖ=τ₀/2ᵏ, hₖ=h₀/2ᵏ)', fontsize=15)
    ax.legend(fontsize=13, loc='lower left', framealpha=0.9)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Аннотации с численным порядком сходимости
    for errs, name, color in [(err_adi, 'МПН', '#1f77b4'), 
                               (err_fds, 'МДШ', '#ff7f0e')]:
        if len(errs) >= 2:
            p_num = np.log(errs[0]/errs[-1]) / np.log(h_arr[0]/h_arr[-1])
            mid_idx = len(errs) // 2
            ax.annotate(f'{name}: O(h^{p_num:.2f})', 
                       xy=(h_arr[mid_idx], errs[mid_idx]),
                       xytext=(h_arr[mid_idx]*2.2, errs[mid_idx]*0.35),
                       fontsize=12, color=color, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    
    plt.tight_layout()
    plt.savefig('convergence_adi_fds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ График сохранён: convergence_adi_fds.png")
    
def plot_time_comparison():
    """Сравнение времени работы трех методов"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Диапазон размеров сеток (явный метод не считаем для N>64 из-за времени)
    Ns = [8, 16, 32, 48, 64, 128]
    tau_base = T / 20
    
    time_adi, time_fds, time_exp = [], [], []
    
    print("\nЗамер времени выполнения:")
    print(f"{'N':>5} | {'МПН t(с)':>10} | {'МДШ t(с)':>10} | {'Явный t(с)':>10}")
    print("-" * 45)
    
    for n in Ns:
        h_curr = XMAX / n
        tau_curr = tau_base / (n // 8)  # Масштабируем tau пропорционально h
        
        res_adi = run('adi', n, n, tau_curr, T)
        res_fds = run('fds', n, n, tau_curr, T)
        
        # Безопасный шаг для явного метода
        tau_exp_safe = 0.8 * (h_curr**2) / (6 * A)
        res_exp = run('exp', n, n, tau_exp_safe, T)
        
        time_adi.append(res_adi['time'])
        time_fds.append(res_fds['time'])
        time_exp.append(res_exp['time'] if not res_exp['unstable'] else np.nan)
        
        exp_str = f"{res_exp['time']:>10.4f}" if not res_exp['unstable'] else "     Н/У"
        print(f"{n:>5} | {res_adi['time']:>10.4f} | {res_fds['time']:>10.4f} | {exp_str}")

    # Построение графика
    ax.plot(Ns, time_adi, 'o-', label='МПН', linewidth=2.5, markersize=9, color='#1f77b4')
    ax.plot(Ns, time_fds, 's-', label='МДШ', linewidth=2.5, markersize=9, color='#ff7f0e')
    
    valid_exp = ~np.isnan(time_exp)
    ax.plot(np.array(Ns)[valid_exp], np.array(time_exp)[valid_exp], 
            '^-', label='Явный', linewidth=2.5, markersize=9, color='green')
    
    ax.set_xlabel('Размер сетки N (Nx=Ny=N)', fontsize=14)
    ax.set_ylabel('Время расчета до T (секунды)', fontsize=14)
    ax.set_title('Сравнение производительности методов\n'
                 '(τ масштабируется пропорционально h)', fontsize=15)
    ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Аннотация с оценкой сложности
    if len(time_adi) >= 2:
        ratio = time_adi[-1] / time_adi[0]
        ax.annotate(f'Рост МПН: ×{ratio:.0f} при N×{Ns[-1]/Ns[0]:.0f}\n'
                    f'(≈ O(N²))', 
                   xy=(Ns[-1], time_adi[-1]),
                   xytext=(Ns[-1]*0.6, time_adi[-1]*1.5),
                   fontsize=11, color='#1f77b4', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))
    
    if len(time_exp) >= 2 and valid_exp.sum() >= 2:
        exp_times = np.array(time_exp)[valid_exp]
        exp_ns = np.array(Ns)[valid_exp]
        ratio_exp = exp_times[-1] / exp_times[0]
        ax.annotate(f'Рост Явного: ×{ratio_exp:.0f}\n'
                    f'(≈ O(N⁴))', 
                   xy=(exp_ns[-1], exp_times[-1]),
                   xytext=(exp_ns[-1]*0.4, exp_times[-1]*0.5),
                   fontsize=11, color='green', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ График сохранён: time_comparison.png")
    
# =============================================================================
# 7. ОСНОВНОЙ БЛОК
# =============================================================================
if __name__ == '__main__':
    print("="*60)
    plot_solutions_comparison()
    plot_convergence_uniform_refinement()
    plot_convergence_adi_vs_fds()
    plot_time_comparison()
    print("\nВсе графики сохранены в текущую директорию.")
    print("1. comparison_3d.png, comparison_heatmaps.png, errors.png")
    print("2. convergence_and_time.png (Сходимость + Время)")