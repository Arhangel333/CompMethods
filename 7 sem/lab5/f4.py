import numpy as np
import matplotlib.pyplot as plt
import os

# НАСТРОЙКА ПОД ВАРИАНТ 4 
TASK = {
    'a': 1.0,
    'L': np.pi,
    'T_final': 0.1,
    'equation_type': 'homogeneous',
    'u_init': lambda x: np.sin(x),
    'u_exact': lambda x, t, a=1.0: np.exp(-a * t) * np.sin(x),
    'bc_type': 'neumann',
    'bc_left_der':  lambda t: np.exp(-1.0 * t),
    'bc_right_der': lambda t: -np.exp(-1.0 * t),
}
""" # ВАР 1
TASK = {
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
                u_new[0]  = cfg['bc_left'](tn1)
                u_new[-1] = cfg['bc_right'](tn1)
            else:  # neumann
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
            A = np.zeros((N_eq, N_eq))
            rhs = np.zeros(N_eq)
            
            # Внутренние узлы
            for i in range(1, Nx):
                if scheme == 'implicit':
                    A[i, i-1] = -gamma
                    A[i, i]   = 1 + 2*gamma
                    A[i, i+1] = -gamma
                    rhs[i]    = u[i]
                else:
                    A[i, i-1] = -gamma/2
                    A[i, i]   = 1 + gamma
                    A[i, i+1] = -gamma/2
                    rhs[i]    = (gamma/2)*u[i-1] + (1-gamma)*u[i] + (gamma/2)*u[i+1]
            
            if cfg['bc_type'] == 'dirichlet':
                A[0, 0] = 1.0;  rhs[0] = cfg['bc_left'](tn1)
                A[-1, -1] = 1.0; rhs[-1] = cfg['bc_right'](tn1)
            else:  # neumann
                mu_L, mu_R = cfg['bc_left_der'](tn1), cfg['bc_right_der'](tn1)
                mu_L_n, mu_R_n = cfg['bc_left_der'](tn), cfg['bc_right_der'](tn)
                
                if approx == '2pt_1st':
                    A[0, 0], A[0, 1] = 1.0, -1.0;    rhs[0] = -h * mu_L
                    A[-1, -1], A[-1, -2] = 1.0, -1.0; rhs[-1] = h * mu_R
                elif approx == '3pt_2nd':
                    A[0, 0], A[0, 1], A[0, 2] = -3.0, 4.0, -1.0; rhs[0] = 2*h*mu_L
                    A[-1, -1], A[-1, -2], A[-1, -3] = -3.0, 4.0, -1.0; rhs[-1] = 2*h*mu_R
                elif approx == '2pt_2nd':
                    if scheme == 'implicit':
                        A[0, 0], A[0, 1] = 1+2*gamma, -2*gamma
                        rhs[0] = u[0] - 2*gamma*h*mu_L
                        A[-1, -1], A[-1, -2] = 1+2*gamma, -2*gamma
                        rhs[-1] = u[-1] + 2*gamma*h*mu_R
                    else:
                        A[0, 0], A[0, 1] = 1+gamma, -gamma
                        rhs[0] = (1-gamma)*u[0] + gamma*u[1] - gamma*h*(mu_L + mu_L_n)
                        A[-1, -1], A[-1, -2] = 1+gamma, -gamma
                        rhs[-1] = (1-gamma)*u[-1] + gamma*u[-2] + gamma*h*(mu_R + mu_R_n)
            
            u_new = np.linalg.solve(A, rhs)

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
        x, t, U = solve(cfg, Nx_plot, Nt_plot, scheme=scheme, approx='2pt_2nd')
        plt.plot(x, U[-1], color=color, linewidth=2, label=label)
    
    x_ex = np.linspace(0, cfg['L'], 200)
    plt.plot(x_ex, cfg['u_exact'](x_ex, cfg['T_final'], cfg['a']),
             'k--', linewidth=2, label='Аналитическое')
    
    plt.xlabel('x'); plt.ylabel('u')
    plt.title(f'Сравнение схем (аппроксимация 2-го порядка), t = {cfg["T_final"]}')
    plt.legend(); plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nГрафик успешно сохранён: {os.path.abspath(filename)}")
    #plt.show()

if __name__ == '__main__':
    approx_methods = ['2pt_1st', '3pt_2nd', '2pt_2nd']
    labels = ['2pt 1-й пор.', '3pt 2-й пор.', '2pt 2-й пор. (фикт. узел)']
    
    for approx, label in zip(approx_methods, labels):
        print(f"\n{'='*50}")
        print(f"Аппроксимация: {label}")
        print(f"{'='*50}")
        for scheme in ['explicit', 'implicit', 'crank_nicolson']:
            print(f"--- {scheme} ---")
            convergence_study(TASK, scheme, approx)
            
    plot_comparison(TASK, 'lab5_fin.png')