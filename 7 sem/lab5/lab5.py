import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ─── Параметры задачи ───
a = 1.0          # коэффициент (можно менять)
L = 1.0          # длина области
T_final = 0.1    # конечное время

u_edge = {0.0, 1.0}

# ─── Аналитическое решение ───
def u_exact(x, t, a=1.0):
    return np.exp(-4 * np.pi**2 * a * t) * np.sin(2 * np.pi * x)

# ─── Начальное условие ───
def u_init(x):
    return np.sin(2 * np.pi * x)

# ─── Функция решения ───
def solve(Nx, Nt, scheme='explicit', bc_type='dirichlet', a=1.0):
    h = L / Nx
    tau = T_final / Nt
    gamma = a * tau / h**2
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T_final, Nt + 1)

    u = u_init(x).copy()
    U = np.zeros((Nt + 1, Nx + 1))
    U[0] = u.copy()

    # Матрицы для неявных схем (внутренние узлы 1..Nx-1)
    n_int = Nx - 1
    if scheme in ('implicit', 'crank_nicolson'):
        if scheme == 'implicit':
            main = (1 + 2*gamma) * np.ones(n_int)
            off  = -gamma * np.ones(n_int - 1)
            A = diags([off, main, off], [-1, 0, 1], format='csc')
        else:  # crank_nicolson
            main_A = (1 + gamma) * np.ones(n_int)
            off_A  = -gamma/2 * np.ones(n_int - 1)
            A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csc')

            main_B = (1 - gamma) * np.ones(n_int)
            off_B  = gamma/2 * np.ones(n_int - 1)
            B = diags([off_B, main_B, off_B], [-1, 0, 1], format='csc')

    for n in range(Nt):
        if scheme == 'explicit':
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + gamma * (u[2:] - 2*u[1:-1] + u[:-2])
        elif scheme == 'implicit':
            rhs = u[1:-1].copy()
            # Граничные условия Дирихле: u[0]=u[Nx]=0 уже учтены (не входят в rhs)
            u_int = spsolve(A, rhs)
            u_new = np.zeros(Nx + 1)
            u_new[1:-1] = u_int
        elif scheme == 'crank_nicolson':
            rhs = B @ u[1:-1]
            u_int = spsolve(A, rhs)
            u_new = np.zeros(Nx + 1)
            u_new[1:-1] = u_int

        # Граничные условия
        u_new[0] = u_edge[0]
        u_new[-1] = u_edge[1]
        u = u_new
        U[n+1] = u.copy()

    return x, t, U

# ─── Исследование сходимости ───
def convergence_study(scheme='crank_nicolson'):
    Ns = [20, 40, 80, 160]
    errors = []
    for Nx in Ns:
        Nt = int(2.1 * a * T_final * (Nx / L)**2)   #Nt = int(Nx * 2)  # tau ~ h^2 для устойчивости явной для неявных можно иначе
        x, t, U = solve(Nx, Nt, scheme=scheme)
        # Погрешность в момент t = T_final
        u_num = U[-1]
        u_ex  = u_exact(x, T_final)
        err = np.max(np.abs(u_num - u_ex))
        errors.append(err)
        print(f"Nx={Nx:4d}, Nt={Nt:4d}, max|error|={err:.4e}")

    print("\nПорядок сходимости:")
    for i in range(1, len(errors)):
        p = np.log2(errors[i-1] / errors[i])
        print(f"  Nx {Ns[i-1]}->{Ns[i]}: порядок ≈ {p:.2f}")

print("=== Явная схема ===")
convergence_study('explicit')

print("\n=== Неявная схема ===")
convergence_study('implicit')

print("\n=== Схема Кранка-Николсона ===")
convergence_study('crank_nicolson')

# ─── Графики решения в разные моменты времени ───
Nx, Nt = 100, 200
x, t, U = solve(Nx, Nt, scheme='crank_nicolson')
""" 
plt.figure(figsize=(8, 5))
for t_idx in [0, Nt//4, Nt//2, 3*Nt//4, Nt]:
    plt.plot(x, U[t_idx], label=f't={t[t_idx]:.3f}')
plt.plot(x, u_exact(x, T_final), 'k--', linewidth=2, label='Аналитическое (t=T)')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.title('Кранка-Николсон')
plt.grid(True)
plt.tight_layout()
plt.savefig("fig.png")
plt.show() """



plt.figure(figsize=(10, 6))

Nx_plot = 100
schemes = ['explicit', 'implicit', 'crank_nicolson']
labels = ['Явная', 'Неявная', 'Кранка-Николсон']
colors = ['blue', 'green', 'red']

for scheme, label, color in zip(schemes, labels, colors):
    Nt_plot = int(2.5 * a * T_final * (Nx_plot / L)**2) if scheme == 'explicit' else 200
    x, t, U = solve(Nx_plot, Nt_plot, scheme=scheme)
    plt.plot(x, U[-1], color=color, linewidth=2, label=label)

x_exact = np.linspace(0, L, 100)
plt.plot(x_exact, u_exact(x_exact, T_final), 'k--', linewidth=2, label='Аналитическое')

plt.xlabel('x')
plt.ylabel('u')
plt.title(f'Сравнение схем в момент t = {T_final}')
plt.legend()
plt.grid(True)


filename = 'lab5_comparison01.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"График сохранен в файл: {os.path.abspath(filename)}")

plt.show()