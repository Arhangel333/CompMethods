import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- параметры задачи ----------
A    = 1.0
XMAX = np.pi/4
YMAX = np.log(2.0)
T    = 0.1

def u_exact(x, y, t):  return np.cos(2*x)*np.cosh(y)*np.exp(-3*A*t)
def bc_left(y, t):     return np.cosh(y)*np.exp(-3*A*t)
def bc_right(y, t):    return np.zeros_like(y)
def bc_bottom(x, t):   return np.cos(2*x)*np.exp(-3*A*t)
def flux_top(x, t):    return 0.75*np.cos(2*x)*np.exp(-3*A*t)

def set_dirichlet(u, xs, ys, t):
    u[0,:], u[-1,:], u[:,0] = bc_left(ys,t), bc_right(ys,t), bc_bottom(xs,t)

def thomas_batch(sub, diag, sup, rhs):
    n, m = diag.size, rhs.shape[1]
    c = np.zeros((n,m)); d = np.zeros((n,m))
    c[0] = sup[0]/diag[0]; d[0] = rhs[0]/diag[0]
    for i in range(1,n):
        den  = diag[i] - sub[i]*c[i-1]
        c[i] = sup[i]/den if i < n-1 else 0.0
        d[i] = (rhs[i] - sub[i]*d[i-1])/den
    x = np.zeros_like(d); x[-1] = d[-1]
    for i in range(n-2,-1,-1):
        x[i] = d[i] - c[i]*x[i+1]
    return x

def make_matrices(mx, my, sx, sy):
    subx = np.full(mx,-sx); subx[0] = 0.0
    supx = np.full(mx,-sx); supx[-1] = 0.0
    suby = np.full(my,-sy); suby[0] = 0.0; suby[-1] = -2*sy
    supy = np.full(my,-sy); supy[-1] = 0.0
    return ((subx, np.full(mx,1+2*sx), supx), (suby, np.full(my,1+2*sy), supy))

def step_adi(u, xs, ys, hx, hy, tau, t):
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    mx, my = Nx-1, Ny
    sx, sy = A*tau/(2*hx**2), A*tau/(2*hy**2)
    Mx, My = make_matrices(mx, my, sx, sy)
    th, tn = t+tau/2, t+tau
    
    uh = u.copy(); set_dirichlet(uh, xs, ys, th)
    
    # --- Полушаг 1 (неявно по X) ---
    rhs = np.empty((mx, my))
    # Явная часть по Y (со слоя k)
    rhs[:, :my-1] = u[1:Nx,1:Ny] + sy*(u[1:Nx,0:Ny-1] - 2*u[1:Nx,1:Ny] + u[1:Nx,2:Ny+1])
    ghost_k = u[1:Nx,Ny-1] + 2*hy*flux_top(xs[1:Nx], t)
    rhs[:, my-1]  = u[1:Nx,Ny] + sy*(u[1:Nx,Ny-1] - 2*u[1:Nx,Ny] + ghost_k)
    
    # ИСПРАВЛЕНИЕ: учет границ по X (со слоя k+1/2)
    rhs[0, :]  += sx * uh[0, 1:Ny+1]
    rhs[-1, :] += sx * uh[Nx, 1:Ny+1]
    
    uh[1:Nx,1:Ny+1] = thomas_batch(*Mx, rhs)
    set_dirichlet(uh, xs, ys, th)
    
    # --- Полушаг 2 (неявно по Y) ---
    r2 = np.empty((my, mx))
    # Явная часть по X (со слоя k+1/2)
    r2[:my-1, :]  = uh[1:Nx,1:Ny].T + sx*(uh[0:Nx-1,1:Ny].T - 2*uh[1:Nx,1:Ny].T + uh[2:Nx+1,1:Ny].T)
    r2[my-1, :]   = uh[1:Nx,Ny].T + sx*(uh[0:Nx-1,Ny].T - 2*uh[1:Nx,Ny].T + uh[2:Nx+1,Ny].T)
    
    # Учет границ по Y (со слоя k+1)
    r2[0, :] += sy * bc_bottom(xs[1:Nx], tn)
    r2[-1, :] += 2 * sy * hy * flux_top(xs[1:Nx], tn)
    
    un = u.copy(); un[1:Nx,1:Ny+1] = thomas_batch(*My, r2).T
    set_dirichlet(un, xs, ys, tn)
    return un

def step_fds(u, xs, ys, hx, hy, tau, t):
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    mx, my = Nx-1, Ny
    rx, ry = A*tau/hx**2, A*tau/hy**2
    Mx, My = make_matrices(mx, my, rx, ry)
    tn = t+tau
    
    uh = u.copy()
    # Шаг 1 (только X)
    rhs = u[1:Nx,1:Ny+1].copy()
    rhs[0, :]  += rx * bc_left(ys[1:Ny+1], tn)
    rhs[-1, :] += rx * bc_right(ys[1:Ny+1], tn)
    uh[1:Nx,1:Ny+1] = thomas_batch(*Mx, rhs)
    set_dirichlet(uh, xs, ys, tn)
    
    # Шаг 2 (только Y)
    r2 = uh[1:Nx,1:Ny+1].T.copy()
    r2[0, :]    += ry * bc_bottom(xs[1:Nx], tn)
    r2[-1, :]   += 2 * ry * hy * flux_top(xs[1:Nx], tn)
    un = u.copy(); un[1:Nx,1:Ny+1] = thomas_batch(*My, r2).T
    set_dirichlet(un, xs, ys, tn)
    return un

def run_with_snapshots(method, Nx, Ny, tau, T, snapshots):
    xs = np.linspace(0, XMAX, Nx+1); ys = np.linspace(0, YMAX, Ny+1)
    hx, hy = xs[1]-xs[0], ys[1]-ys[0]
    u = u_exact(xs[:,None], ys[None,:], 0.0)
    step = step_adi if method=='adi' else step_fds
    K = int(round(T/tau)); snaps = {}
    for k in range(K):
        u = step(u, xs, ys, hx, hy, tau, k*tau)
        if k+1 in snapshots: snaps[k+1] = u.copy()
    return xs, ys, snaps

def run(method, Nx, Ny, tau, T):
    xs = np.linspace(0, XMAX, Nx+1); ys = np.linspace(0, YMAX, Ny+1)
    hx, hy = xs[1]-xs[0], ys[1]-ys[0]
    u = u_exact(xs[:,None], ys[None,:], 0.0)
    step = step_adi if method=='adi' else step_fds
    K = int(round(T/tau)); out = {}
    for k in range(K):
        u = step(u, xs, ys, hx, hy, tau, k*tau)
        if k+1 == K//2: out['mid'] = np.max(np.abs(u - u_exact(xs[:,None], ys[None,:], (k+1)*tau)))
        if k+1 == K:    out['end'] = np.max(np.abs(u - u_exact(xs[:,None], ys[None,:], (k+1)*tau)))
    return out

# ---------- Визуализация ----------
def plot_solution_3d(xs, ys, u, title, ax):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ax.plot_surface(X, Y, u, cmap='viridis', alpha=0.9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_title(title)

def plot_heatmap(xs, ys, u, title, ax):
    im = ax.pcolormesh(xs, ys, u.T, cmap='hot', shading='auto')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)
    return im

def plot_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    taus = [T/10, T/20, T/40, T/80, T/160, T/320, T/2000, T/4000, T/10000]
    errors_adi, errors_fds = [], []
    print("Исследование сходимости по tau...")
    for tau in taus:
        errors_adi.append(run('adi', 64, 64, tau, T)['end'])
        errors_fds.append(run('fds', 64, 64, tau, T)['end'])
        print(f"  tau=T/{T/tau:.0f}: МПН={errors_adi[-1]:.3e}, МДШ={errors_fds[-1]:.3e}")
        
    ax1.loglog(taus, errors_adi, 'o-', label='МПН (ADI)', linewidth=2)
    ax1.loglog(taus, errors_fds, 's-', label='МДШ (FDS)', linewidth=2)
    ax1.loglog(taus, [errors_adi[0]*(t/taus[0])**2 for t in taus], '--', color='gray', label='O(τ²)')
    ax1.loglog(taus, [errors_fds[0]*(t/taus[0]) for t in taus], '--', color='gray', label='O(τ)')
    ax1.set_xlabel('τ'); ax1.set_ylabel('Погрешность')
    ax1.set_title('Сходимость по τ (Nx=Ny=64)'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    hs = [16, 32, 64, 128]
    h_values = [XMAX/n for n in hs]
    errors_adi_h, errors_fds_h = [], []
    print("\nИсследование сходимости по h...")
    for n in hs:
        errors_adi_h.append(run('adi', n, n, T/2000, T)['end'])
        errors_fds_h.append(run('fds', n, n, T/2000, T)['end'])
        print(f"  N={n}: МПН={errors_adi_h[-1]:.3e}, МДШ={errors_fds_h[-1]:.3e}")
        
    ax2.loglog(h_values, errors_adi_h, 'o-', label='МПН (ADI)', linewidth=2)
    ax2.loglog(h_values, errors_fds_h, 's-', label='МДШ (FDS)', linewidth=2)
    ax2.loglog(h_values, [errors_adi_h[0]*(h/h_values[0])**2 for h in h_values], '--', color='gray', label='O(h²)')
    ax2.set_xlabel('h'); ax2.set_ylabel('Погрешность')
    ax2.set_title('Сходимость по h (τ=T/2000)'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_solutions_comparison():
    Nx, Ny = 32, 32
    tau = T/100
    snapshots = [int(T/tau)] # Только финальный момент
    
    print("\nРасчёт снимков решения...")
    xs, ys, snaps_adi = run_with_snapshots('adi', Nx, Ny, tau, T, snapshots)
    _, _, snaps_fds = run_with_snapshots('fds', Nx, Ny, tau, T, snapshots)
    
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_analytic = u_exact(X, Y, T)
    
    # 1. 3D поверхности (Точное, МПН, МДШ)
    fig = plt.figure(figsize=(18, 5))
    for idx, (data, title) in enumerate([(u_analytic, 'Точное'), (snaps_adi[snapshots[-1]], 'МПН'), (snaps_fds[snapshots[-1]], 'МДШ')]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        plot_solution_3d(xs, ys, data, title, ax)
    plt.suptitle(f'Сравнение 3D решений при t={T}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('comparison_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Тепловые карты (Точное, МПН, МДШ)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (data, title) in enumerate([(u_analytic, 'Точное'), (snaps_adi[snapshots[-1]], 'МПН'), (snaps_fds[snapshots[-1]], 'МДШ')]):
        im = plot_heatmap(xs, ys, data, title, axes[idx])
        plt.colorbar(im, ax=axes[idx])
    plt.suptitle(f'Тепловые карты при t={T}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('comparison_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Карты ошибок
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    err_adi = np.abs(snaps_adi[snapshots[-1]] - u_analytic)
    err_fds = np.abs(snaps_fds[snapshots[-1]] - u_analytic)
    
    vmax = max(np.max(err_adi), np.max(err_fds)) # Единая шкала для честного сравнения
    
    im1 = ax1.pcolormesh(xs, ys, err_adi.T, cmap='Reds', shading='auto', vmin=0, vmax=vmax)
    ax1.set_title(f'Ошибка МПН (max={np.max(err_adi):.3e})'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.pcolormesh(xs, ys, err_fds.T, cmap='Reds', shading='auto', vmin=0, vmax=vmax)
    ax2.set_title(f'Ошибка МДШ (max={np.max(err_fds):.3e})'); ax2.set_xlabel('x'); ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig('errors.png', dpi=150, bbox_inches='tight')
    plt.close()

# ---------- Запуск ----------
if __name__ == '__main__':
    print("="*60)
    plot_solutions_comparison()
    plot_convergence()
    
    def table(title, e):
        print(title)
        print(f"  e1={e[0]:.3e} e2={e[1]:.3e} e3={e[2]:.3e} | e1/e2={e[0]/e[1]:5.2f} e2/e3={e[1]/e[2]:5.2f}")
    
    print("\n" + "="*60)
    print("Таблицы сходимости")
    print("="*60)
    for meth, label in (('adi','МПН'), ('fds','МДШ')):
        table(f"\n{label}: измельчение tau (Nx=Ny=64)",
              [run(meth, 64, 64, T/d, T)['end'] for d in (20, 40, 80)])
        table(f"{label}: измельчение h (tau=T/2000)",
              [run(meth, n, n, T/2000, T)['end'] for n in (16, 32, 64)])