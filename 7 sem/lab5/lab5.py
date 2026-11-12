import numpy as np
import matplotlib.pyplot as plt
import os

TASK = {  #4 variant
    'a': 1.0,
    'L': np.pi,
    'T_final': 1.0,
    'equation_type': 'homogeneous',
    'u_init': lambda x: np.sin(x),
    'u_exact': lambda x, t, a=1.0: np.exp(-a * t) * np.sin(x),
    'bc_type': 'neumann',
    'bc_left_der':  lambda t: np.exp(-1.0 * t),
    'bc_right_der': lambda t: -np.exp(-1.0 * t),
}

""" TASK = { #1variant
    'a': 1.0,
    'L': 1.0,
    'T_final': 0.1,
    'equation_type': 'homogeneous',
    'u_init': lambda x: np.sin(2*np.pi*x),
    'u_exact': lambda x, t, a=1.0: np.exp(-4*np.pi**2*a * t) * np.sin(2*np.pi*x),
    'bc_type': 'dirichlet',
    'bc_left':  lambda t: 0.0,
    'bc_right': lambda t: 0.0,
} """


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

def solve(cfg, Nx, Nt, scheme='crank_nicolson', approx='2pt_2nd'):
    a, L, T = cfg['a'], cfg['L'], cfg['T_final']
    h, tau = L / Nx, T / Nt
    gamma = a * tau / h**2
    
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    
    u = cfg['u_init'](x).copy()
    U = np.zeros((Nt + 1, Nx + 1))
    U[0] = u.copy()
    
    N_eq = Nx + 1

    for n in range(Nt):
        tn, tn1 = t[n], t[n+1]
        
        if scheme == 'explicit':
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + gamma * (u[2:] - 2*u[1:-1] + u[:-2])
            
            if cfg['bc_type'] == 'dirichlet':
                u_new[0] = cfg['bc_left'](tn1)
                u_new[-1] = cfg['bc_right'](tn1)
            else:
                mu_L, mu_R = cfg['bc_left_der'](tn1), cfg['bc_right_der'](tn1)
                if approx == '2pt_1st':
                    u_new[0] = u_new[1] - h * mu_L
                    u_new[-1] = u_new[-2] + h * mu_R
                elif approx == '3pt_2nd':
                    u_new[0] = (4*u_new[1] - u_new[2])/3 - (2*h/3)*mu_L
                    u_new[-1] = (4*u_new[-2] - u_new[-3])/3 + (2*h/3)*mu_R
                elif approx == '2pt_2nd':
                    mu_L_n, mu_R_n = cfg['bc_left_der'](tn), cfg['bc_right_der'](tn)
                    u_ghost_L = u[1] - 2*h*mu_L_n
                    u_ghost_R = u[-2] + 2*h*mu_R_n
                    u_new[0] = u[0] + gamma*(u[1] - 2*u[0] + u_ghost_L)
                    u_new[-1] = u[-1] + gamma*(u_ghost_R - 2*u[-1] + u[-2])
        
        else:  # Implicit / Crank-Nicolson
            # Инициализация диагоналей и правой части
            a_sub = np.zeros(N_eq - 1)
            b_main = np.zeros(N_eq)
            c_sup = np.zeros(N_eq - 1)
            d = np.zeros(N_eq)
            
            # Внутренние узлы
            for i in range(1, Nx):
                if scheme == 'implicit':
                    a_sub[i-1] = -gamma
                    b_main[i] = 1 + 2*gamma
                    c_sup[i] = -gamma
                    d[i] = u[i]
                else:
                    a_sub[i-1] = -gamma/2
                    b_main[i] = 1 + gamma
                    c_sup[i] = -gamma/2
                    d[i] = (gamma/2)*u[i-1] + (1-gamma)*u[i] + (gamma/2)*u[i+1]
            
            # Граничные условия
            if cfg['bc_type'] == 'dirichlet':
                b_main[0] = 1.0;  d[0] = cfg['bc_left'](tn1)
                b_main[-1] = 1.0; d[-1] = cfg['bc_right'](tn1)
            else:
                mu_L, mu_R = cfg['bc_left_der'](tn1), cfg['bc_right_der'](tn1)
                mu_L_n, mu_R_n = cfg['bc_left_der'](tn), cfg['bc_right_der'](tn)
                
                if approx == '2pt_1st':
                    b_main[0] = 1.0; c_sup[0] = -1.0; d[0] = -h * mu_L
                    b_main[-1] = 1.0; a_sub[-1] = -1.0; d[-1] = h * mu_R
                    
                elif approx == '2pt_2nd':
                    if scheme == 'implicit':
                        b_main[0] = 1 + 2*gamma; c_sup[0] = -2*gamma
                        d[0] = u[0] - 2*gamma*h*mu_L
                        b_main[-1] = 1 + 2*gamma; a_sub[-1] = -2*gamma
                        d[-1] = u[-1] + 2*gamma*h*mu_R
                    else:
                        b_main[0] = 1 + gamma; c_sup[0] = -gamma
                        d[0] = (1-gamma)*u[0] + gamma*u[1] - gamma*h*(mu_L + mu_L_n)
                        b_main[-1] = 1 + gamma; a_sub[-1] = -gamma
                        d[-1] = (1-gamma)*u[-1] + gamma*u[-2] + gamma*h*(mu_R + mu_R_n)
                
                elif approx == '3pt_2nd':
                    # 3pt_2nd ломает трехдиагональность (появляется A[0,2]).
                    # Для него используем плотный решатель.
                    A = np.zeros((N_eq, N_eq))
                    A[np.arange(N_eq), np.arange(N_eq)] = b_main
                    A[np.arange(1, N_eq), np.arange(N_eq-1)] = a_sub
                    A[np.arange(N_eq-1), np.arange(1, N_eq)] = c_sup
                    
                    A[0, 0], A[0, 1], A[0, 2] = 3.0, -4.0, 1.0
                    d[0] = -2*h*mu_L
                    if scheme == 'implicit':
                        A[1, 1] = 3 + 2*gamma; A[1, 2] = -2*gamma; d[1] = 3*u[1] - 2*gamma*h*mu_L
                    else:
                        A[1, 1] = 3 + gamma; A[1, 2] = -gamma
                        d[1] = 3 * ((gamma/2)*u[0] + (1-gamma)*u[1] + (gamma/2)*u[2]) - gamma*h*mu_L
                    
                    A[-1, -1], A[-1, -2], A[-1, -3] = 3.0, -4.0, 1.0
                    d[-1] = 2*h*mu_R
                    if scheme == 'implicit':
                        A[-2, -2] = 3 + 2*gamma; A[-2, -3] = -2*gamma; d[-2] = 3*u[-2] + 2*gamma*h*mu_R
                    else:
                        A[-2, -2] = 3 + gamma; A[-2, -3] = -gamma
                        d[-2] = 3 * ((gamma/2)*u[-1] + (1-gamma)*u[-2] + (gamma/2)*u[-3]) + gamma*h*mu_R
                    
                    u_new = np.linalg.solve(A, d)
                    u = u_new
                    U[n+1] = u.copy()
                    continue # Переходим к следующему шагу времени
            
            # Решение трехдиагональной системы прогонкой
            u_new = progonka(a_sub, b_main, c_sup, d)

        u = u_new
        U[n+1] = u.copy()
        
    return x, t, U


def convergence_study(cfg, scheme, approx):
    Ns = [20, 40, 80, 160]
    errors = []
    for Nx in Ns:
        Nt = int(2.5 * cfg['a'] * cfg['T_final'] * (Nx / cfg['L'])**2) if scheme == 'explicit' else Nx * 2
        x, t, U = solve(cfg, Nx, Nt, scheme=scheme, approx=approx)
        err = np.max(np.abs(U[-1] - cfg['u_exact'](x, cfg['T_final'], cfg['a'])))
        errors.append(err)
        print(f"  Nx={Nx:4d}, Nt={Nt:4d}, error={err:.4e}")
    
    print("  Порядок сходимости:")
    for i in range(1, len(errors)):
        print(f"    {Ns[i-1]}->{Ns[i]}: ≈ {np.log2(errors[i-1]/errors[i]):.2f}")


def plot_comparison(cfg, filename='lab5_result.png'):
    plt.figure(figsize=(10, 6))
    Nx_plot = 100
    schemes = ['explicit', 'implicit', 'crank_nicolson']
    labels  = ['Явная', 'Неявная', 'Кранка-Николсон']
    colors  = ['blue', 'green', 'red']
    
    for scheme, label, color in zip(schemes, labels, colors):
        Nt_plot = int(2.5 * cfg['a'] * cfg['T_final'] * (Nx_plot / cfg['L'])**2) if scheme == 'explicit' else Nx_plot * 2
        x, t, U = solve(cfg, Nx_plot, Nt_plot, scheme=scheme, approx='2pt_1st')
        plt.plot(x, U[-1], color=color, linewidth=2, label=label)
    
    x_ex = np.linspace(0, cfg['L'], 200)
    plt.plot(x_ex, cfg['u_exact'](x_ex, cfg['T_final'], cfg['a']),
             'k--', linewidth=2, label='Аналитическое')
    
    plt.xlabel('x'); plt.ylabel('u')
    plt.title(f'Сравнение схем (аппроксимация 2-го порядка), t = {cfg["T_final"]}')
    plt.legend(); plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранён: {os.path.abspath(filename)}")


def plot_error_vs_time(cfg, aprx='2pt_2nd', filename='lab5_error_vs_time.png'):
    plt.figure(figsize=(10, 6))
    Nx_plot = 100
    schemes = ['explicit', 'implicit', 'crank_nicolson']
    labels  = ['Явная', 'Неявная', 'Кранка-Николсон']
    colors  = ['blue', 'green', 'red']
    
    for scheme, label, color in zip(schemes, labels, colors):
        # Безопасный выбор Nt для всех схем
        Nt_plot = int(2.5 * cfg['a'] * cfg['T_final'] * (Nx_plot / cfg['L'])**2) if scheme == 'explicit' else Nx_plot * 2
        
        x, t, U = solve(cfg, Nx_plot, Nt_plot, scheme=scheme, approx=aprx)
        
        # Вычисляем максимальную погрешность на каждом временном слое
        errors_t = []
        for n in range(len(t)):
            u_ex = cfg['u_exact'](x, t[n], cfg['a'])
            err = np.max(np.abs(U[n] - u_ex))
            errors_t.append(err)
        
        plt.plot(t, errors_t, color=color, linewidth=2, label=label)
    
    plt.xlabel('Время t')
    plt.ylabel('Максимальная погрешность (логарифмическая шкала)')
    plt.title('Зависимость максимальной погрешности от времени')
    #plt.yscale('log')  # Логарифмическая шкала для наглядности порядков ошибки
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"График ошибок от времени сохранён: {os.path.abspath(filename)}")
    

if __name__ == '__main__':
    approx_methods = ['2pt_1st', '3pt_2nd', '2pt_2nd']
    labels = ['2pt 1-й пор.', '3pt 2-й пор.', '2pt 2-й пор. (фикт. узел)']
    files = ['lab5_error_1.png', 'lab5_error_2.png', 'lab5_error_3.png',]
    
    for approx, label, fl in zip(approx_methods, labels, files):
        print(f"\n{'='*50}")
        print(f"Аппроксимация: {label}")
        print(f"{'='*50}")
        for scheme in ['explicit', 'implicit', 'crank_nicolson']:
            print(f"--- {scheme} ---")
            convergence_study(TASK, scheme, approx)
        plot_error_vs_time(TASK, approx, filename=fl)
    plot_comparison(TASK, 'lab5_result.png')
    