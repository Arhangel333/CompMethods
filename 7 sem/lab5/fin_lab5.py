import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os

TASK = {
    'a': 1.0,
    'L': 1.0,
    'T_final': 0.1,
    'equation_type': 'homogeneous',
    'source': lambda x: np.sin(np.pi * x),
    'u_init': lambda x: np.sin(2 * np.pi * x),
    'u_exact': lambda x, t, a=1.0: np.exp(-4*np.pi**2*a*t) * np.sin(2*np.pi*x),
    'bc_type': 'dirichlet',
    'bc_left':  lambda t: 0.0,
    'bc_right': lambda t: 0.0,
    'bc_left_der':  lambda t: np.exp(-t),
    'bc_right_der': lambda t: -np.exp(-t),
    'derivative_approx': '2pt_2nd',
}


def solve(cfg, Nx, Nt, scheme='crank_nicolson'):
    a      = cfg['a']
    L      = cfg['L']
    T      = cfg['T_final']
    h      = L / Nx
    tau    = T / Nt
    gamma  = a * tau / h**2
    
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    
    u = cfg['u_init'](x).copy()
    U = np.zeros((Nt + 1, Nx + 1))
    U[0] = u.copy()
    
    n_int = Nx - 1
    
    A = None
    B = None
    if scheme in ('implicit', 'crank_nicolson'):
        main_A = (1 + 2*gamma) * np.ones(n_int)
        off_A  = -gamma * np.ones(n_int - 1)
        if scheme == 'crank_nicolson':
            main_A = (1 + gamma) * np.ones(n_int)
            off_A  = -gamma/2 * np.ones(n_int - 1)
        A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csc')
        
        if scheme == 'crank_nicolson':
            main_B = (1 - gamma) * np.ones(n_int)
            off_B  = gamma/2 * np.ones(n_int - 1)
            B = diags([off_B, main_B, off_B], [-1, 0, 1], format='csc')
    
    for n in range(Nt):
        tn = t[n]
        tn1 = t[n+1]
        
        if scheme == 'explicit':
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + gamma * (u[2:] - 2*u[1:-1] + u[:-2])
            
            if cfg['equation_type'] == 'inhomogeneous':
                u_new[1:-1] += tau * cfg['source'](x[1:-1])
        
        elif scheme == 'implicit':
            rhs = u[1:-1].copy()
            if cfg['equation_type'] == 'inhomogeneous':
                rhs += tau * cfg['source'](x[1:-1])
            u_int = spsolve(A, rhs)
            u_new = np.zeros(Nx + 1)
            u_new[1:-1] = u_int
        
        elif scheme == 'crank_nicolson':
            rhs = B @ u[1:-1]
            if cfg['equation_type'] == 'inhomogeneous':
                rhs += tau * cfg['source'](x[1:-1])
            u_int = spsolve(A, rhs)
            u_new = np.zeros(Nx + 1)
            u_new[1:-1] = u_int
        
        apply_boundary_conditions(cfg, u_new, u, h, gamma, tn1, scheme, A)
        
        u = u_new
        U[n+1] = u.copy()
    
    return x, t, U


def apply_boundary_conditions(cfg, u_new, u_old, h, gamma, t, scheme, A):
    bc = cfg['bc_type']
    
    if bc == 'dirichlet':
        u_new[0]  = cfg['bc_left'](t)
        u_new[-1] = cfg['bc_right'](t)
        
        if scheme in ('implicit', 'crank_nicolson'):
            pass
    
    elif bc == 'neumann':
        mu_L  = cfg['bc_left_der'](t)
        mu_R  = cfg['bc_right_der'](t)
        approx = cfg['derivative_approx']
        
        if scheme == 'explicit':
            apply_neumann_explicit(u_new, u_old, h, gamma, mu_L, mu_R, approx)
        else:
            apply_neumann_implicit(u_new, u_old, h, gamma, mu_L, mu_R, approx, A, scheme)


def apply_neumann_explicit(u_new, u_old, h, gamma, mu_L, mu_R, approx):
    if approx == '2pt_1st':
        u_new[0]  = u_new[1]  - h * mu_L
        u_new[-1] = u_new[-2] + h * mu_R
    
    elif approx == '3pt_2nd':
        u_new[0]  = (4*u_new[1]  - u_new[2])  / 3 - (2*h/3) * mu_L
        u_new[-1] = (4*u_new[-2] - u_new[-3]) / 3 + (2*h/3) * mu_R
    
    elif approx == '2pt_2nd':
        u_ghost_L = u_new[1]  - 2*h*mu_L
        u_ghost_R = u_new[-2] + 2*h*mu_R
        u_new[0]  = u_old[0]  + gamma*(u_new[1]  - 2*u_new[0]  + u_ghost_L)
        u_new[-1] = u_old[-1] + gamma*(u_ghost_R - 2*u_new[-1] + u_new[-2])


def apply_neumann_implicit(u_new, u_old, h, gamma, mu_L, mu_R, approx, A, scheme):
    n_int = len(u_new) - 2
    A_mod = A.toarray().copy()
    
    if approx == '2pt_1st':
        u_new[0]  = u_new[1]  - h * mu_L
        u_new[-1] = u_new[-2] + h * mu_R
    
    elif approx == '3pt_2nd':
        u_new[0]  = (4*u_new[1]  - u_new[2])  / 3 - (2*h/3) * mu_L
        u_new[-1] = (4*u_new[-2] - u_new[-3]) / 3 + (2*h/3) * mu_R
    
    elif approx == '2pt_2nd':
        if scheme == 'implicit':
            c_main = 1 + 2*gamma
            c_off  = -gamma
            rhs_corr = 2*gamma*h
        else:
            c_main = 1 + gamma
            c_off  = -gamma/2
            rhs_corr = gamma*h
        
        A_mod[0, 0] = c_main
        A_mod[0, 1] = 2*c_off
        
        A_mod[-1, -1] = c_main
        A_mod[-1, -2] = 2*c_off
        
        u_new[0]  = u_new[1]  - 2*h*mu_L
        u_new[-1] = u_new[-2] + 2*h*mu_R


def convergence_study(cfg, scheme='crank_nicolson'):
    Ns = [20, 40, 80, 160]
    errors = []
    for Nx in Ns:
        if scheme == 'explicit':
            Nt = int(2.5 * cfg['a'] * cfg['T_final'] * (Nx / cfg['L'])**2)
        else:
            Nt = int(Nx * 2)
        
        x, t, U = solve(cfg, Nx, Nt, scheme=scheme)
        u_num = U[-1]
        u_ex  = cfg['u_exact'](x, cfg['T_final'], cfg['a'])
        err = np.max(np.abs(u_num - u_ex))
        errors.append(err)
        print(f"  Nx={Nx:4d}, Nt={Nt:4d}, max|error|={err:.4e}")
    
    print("  Порядок сходимости:")
    for i in range(1, len(errors)):
        p = np.log2(errors[i-1] / errors[i])
        print(f"    Nx {Ns[i-1]}->{Ns[i]}: ≈ {p:.2f}")
    return errors


def plot_comparison(cfg, filename='lab5_comparison.png'):
    plt.figure(figsize=(10, 6))
    
    Nx_plot = 100
    schemes = ['explicit', 'implicit', 'crank_nicolson']
    labels  = ['Явная', 'Неявная', 'Кранка-Николсон']
    colors  = ['blue', 'green', 'red']
    
    for scheme, label, color in zip(schemes, labels, colors):
        Nt_plot = int(2.5 * cfg['a'] * cfg['T_final'] * (Nx_plot / cfg['L'])**2) \
                  if scheme == 'explicit' else 200
        x, t, U = solve(cfg, Nx_plot, Nt_plot, scheme=scheme)
        plt.plot(x, U[-1], color=color, linewidth=2, label=label)
    
    x_ex = np.linspace(0, cfg['L'], 200)
    plt.plot(x_ex, cfg['u_exact'](x_ex, cfg['T_final'], cfg['a']),
             'k--', linewidth=2, label='Аналитическое')
    
    plt.xlabel('x'); plt.ylabel('u')
    plt.title(f'Сравнение схем, t = {cfg["T_final"]}')
    plt.legend(); plt.grid(True)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"График сохранён: {os.path.abspath(filename)}")
    plt.show()

if __name__ == '__main__':
    approx_methods = ['2pt_1st', '3pt_2nd', '2pt_2nd']
    labels = ['2pt 1-й порядок', '3pt 2-й порядок', '2pt 2-й порядок (фикт. узел)']
    
    for approx, label in zip(approx_methods, labels):
        TASK['derivative_approx'] = approx
        TASK['bc_type'] = 'neumann'  # нужно для варианта 4
        
        print(f"\n{'='*50}")
        print(f"Аппроксимация: {label}")
        print(f"{'='*50}")
        
        print("=== Явная схема ===")
        convergence_study(TASK, 'explicit')
        
        print("\n=== Неявная схема ===")
        convergence_study(TASK, 'implicit')
        
        print("\n=== Кранка-Николсон ===")
        convergence_study(TASK, 'crank_nicolson')
    
    plot_comparison(TASK, 'lab5_result.png')