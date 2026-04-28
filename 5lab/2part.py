"""
Решение системы нелинейных уравнений
Использует класс NonlinearSolver из first_part.py
Методы: Ньютон, простая итерация, Зейдель
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sympify
from first_part import NonlinearSolver  
import input as inp

# ==================== ВВОД СИСТЕМЫ ====================
print("="*60)
print("РЕШЕНИЕ СИСТЕМЫ НЕЛИНЕЙНЫХ УРАВНЕНИЙ")
print("="*60)

# Берём данные из input.py
f1_str = inp.f1_system
f2_str = inp.f2_system
eps = float(inp.eps)
x0 = inp.x0_system
y0 = inp.y0_system

print(f"\nСистема:")
print(f"  f1(x,y) = {f1_str} = 0")
print(f"  f2(x,y) = {f2_str} = 0")
print(f"  Точность ε = {eps}")
print(f"  Начальное приближение: ({x0}, {y0})")

# ==================== СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ ====================
x, y = symbols('x y')

# Преобразуем строки в sympy-выражения
f1_sym = sympify(f1_str)
f2_sym = sympify(f2_str)

# Числовые функции
f1 = lambdify([x, y], f1_sym, 'numpy')
f2 = lambdify([x, y], f2_sym, 'numpy')

# Производные для метода Ньютона
f1_x_sym = f1_sym.diff(x)
f1_y_sym = f1_sym.diff(y)
f2_x_sym = f2_sym.diff(x)
f2_y_sym = f2_sym.diff(y)

f1_x = lambdify([x, y], f1_x_sym, 'numpy')
f1_y = lambdify([x, y], f1_y_sym, 'numpy')
f2_x = lambdify([x, y], f2_x_sym, 'numpy')
f2_y = lambdify([x, y], f2_y_sym, 'numpy')

# ==================== ГРАФИК ДЛЯ НАЧАЛЬНОГО ПРИБЛИЖЕНИЯ ====================
print("\nСтроим графики...")

x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, f1(X, Y), levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, f2(X, Y), levels=[0], colors='red', linewidths=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графики f1=0 (синий) и f2=0 (красный)')
plt.grid(True)
plt.legend(['f1=0', 'f2=0'])
plt.savefig(inp.second_graphic)
print(f" График сохранён в файл: {inp.second_graphic}")

# ==================== МЕТОД НЬЮТОНА ДЛЯ СИСТЕМЫ ====================
def newton_system(x0, y0, eps, max_iter=50):
    """Метод Ньютона для системы двух уравнений"""
    x, y = x0, y0
    history = [[x, y]]
    
    for it in range(max_iter):
        # Значения функций
        F1 = f1(x, y)
        F2 = f2(x, y)
        
        # Матрица Якоби
        J11 = f1_x(x, y)
        J12 = f1_y(x, y)
        J21 = f2_x(x, y)
        J22 = f2_y(x, y)
        
        det = J11 * J22 - J12 * J21
        
        if abs(det) < 1e-12:
            print(f"  Предупреждение: определитель близок к нулю на итерации {it}")
            return np.array([x, y]), it, history
        
        # Решаем систему J * Δ = -F
        dx = (-F1 * J22 + F2 * J12) / det
        dy = (-F2 * J11 + F1 * J21) / det
        
        x_new = x + dx
        y_new = y + dy
        history.append([x_new, y_new])
        if(inp.output):
            print(f"dx = {dx} dy = {dy} x = {x} y = {y} x_new = {x_new} y_new = {y_new} max(abs(dx), abs(dy)) = {max(abs(dx), abs(dy))}")
        if max(abs(dx), abs(dy)) < eps:
            return np.array([x_new, y_new]), it + 1, history
        
        x, y = x_new, y_new
    
    return np.array([x, y]), max_iter, history

# ==================== МЕТОД ПРОСТОЙ ИТЕРАЦИИ ====================
def simple_iteration_system(x0, y0, eps, max_iter=200):
    """Метод простой итерации с безопасным подбором alpha"""
    
    # Вычисляем производные в начальной точке
    J11 = float(f1_x(x0, y0))
    J22 = float(f2_y(x0, y0))
    
    # Подбираем alpha для устойчивости
    alpha = 0.05  # маленький шаг по умолчанию
    
    if abs(J11) > 1e-10:
        alpha1 = 1.8 / abs(J11)
        if alpha1 < alpha:
            alpha = alpha1
    if abs(J22) > 1e-10:
        alpha2 = 1.8 / abs(J22)
        if alpha2 < alpha:
            alpha = alpha2
    
    alpha = min(alpha, 0.1)  # ограничиваем сверху
    print(f"  Используется alpha = {alpha:.4f}")
    
    x, y = x0, y0
    history = [[x, y]]
    
    for it in range(max_iter):
        try:
            x_new = x - alpha * f1(x, y)
            y_new = y - alpha * f2(x, y)
            
            if(inp.output):
                print(f"abs(x_new) = {abs(x_new)} abs(y_new) = {abs(y_new)}")
            if abs(x_new) > 1e10 or abs(y_new) > 1e10:
                print(f"   Расходится на итерации {it+1}")
                return np.array([x, y]), it, history
            
            history.append([x_new, y_new])
            
            if max(abs(x_new - x), abs(y_new - y)) < eps:
                return np.array([x_new, y_new]), it + 1, history
            
            x, y = x_new, y_new
        except:
            print(f"   Ошибка на итерации {it+1}")
            return np.array([x, y]), it, history
    
    return np.array([x, y]), max_iter, history


def seidel_system(x0, y0, eps, max_iter=200):
    """Метод Зейделя с безопасным подбором alpha"""
    
    J11 = float(f1_x(x0, y0))
    J22 = float(f2_y(x0, y0))
    
    alpha = 0.05
    
    if abs(J11) > 1e-10:
        alpha1 = 1.8 / abs(J11)
        if alpha1 < alpha:
            alpha = alpha1
    if abs(J22) > 1e-10:
        alpha2 = 1.8 / abs(J22)
        if alpha2 < alpha:
            alpha = alpha2
    
    alpha = min(alpha, 0.1)
    print(f"  Используется alpha = {alpha:.4f}")
    
    x, y = x0, y0
    history = [[x, y]]
    
    for it in range(max_iter):
        try:
            # Сначала обновляем x
            x_new = x - alpha * f1(x, y)
            # Потом y с использованием нового x
            y_new = y - alpha * f2(x_new, y)
            
            if abs(x_new) > 1e10 or abs(y_new) > 1e10:
                print(f"   Расходится на итерации {it+1}")
                return np.array([x, y]), it, history
            
            history.append([x_new, y_new])
            
            if max(abs(x_new - x), abs(y_new - y)) < eps:
                return np.array([x_new, y_new]), it + 1, history
            
            x, y = x_new, y_new
        except:
            print(f"   Ошибка на итерации {it+1}")
            return np.array([x, y]), it, history
    
    return np.array([x, y]), max_iter, history

# ==================== ВЫПОЛНЕНИЕ ====================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ")
print("="*60)

# 1. Метод Ньютона
print("\n--- МЕТОД НЬЮТОНА (для системы) ---")
root_n, it_n, hist_n = newton_system(x0, y0, eps)
print(f"  Корень: x = {root_n[0]:.8f}, y = {root_n[1]:.8f}")
print(f"  f1 = {f1(*root_n):.2e}, f2 = {f2(*root_n):.2e}")
print(f"  Итераций: {it_n}")

# 2. Метод простой итерации
print("\n--- МЕТОД ПРОСТОЙ ИТЕРАЦИИ ---")
root_si, it_si, hist_si = simple_iteration_system(x0, y0, eps)
print(f"  Корень: x = {root_si[0]:.8f}, y = {root_si[1]:.8f}")
print(f"  f1 = {f1(*root_si):.2e}, f2 = {f2(*root_si):.2e}")
print(f"  Итераций: {it_si}")

# 3. Метод Зейделя
print("\n--- МЕТОД ЗЕЙДЕЛЯ ---")
root_z, it_z, hist_z = seidel_system(x0, y0, eps)
print(f"  Корень: x = {root_z[0]:.8f}, y = {root_z[1]:.8f}")
print(f"  f1 = {f1(*root_z):.2e}, f2 = {f2(*root_z):.2e}")
print(f"  Итераций: {it_z}")

# ==================== ПРОВЕРКА УСЛОВИЙ СХОДИМОСТИ ====================
print("\n" + "="*60)
print("ПРОВЕРКА УСЛОВИЙ СХОДИМОСТИ")
print("="*60)

# Для метода Ньютона: определитель Якоби
det_J = f1_x(*root_n) * f2_y(*root_n) - f1_y(*root_n) * f2_x(*root_n)
print(f"\n  Для метода Ньютона:")
print(f"    det(J) = {det_J:.6f}")
if abs(det_J) > 1e-6:
    print("     det(J) ≠ 0 → метод сходится (квадратично)")
else:
    print("     det(J) ≈ 0 → метод может расходиться")

# Для методов итераций: норма матрицы итераций
alpha_est = 0.1
J = np.array([[f1_x(*root_n), f1_y(*root_n)], [f2_x(*root_n), f2_y(*root_n)]])
iter_matrix = np.eye(2) - alpha_est * J
norm = np.linalg.norm(iter_matrix, ord=np.inf)

print(f"\n  Для методов простой итерации и Зейделя:")
print(f"    Норма матрицы итераций: {norm:.4f}")
if norm < 1:
    print("     Норма < 1 → методы сходятся")
else:
    print("     Норма ≥ 1 → методы могут расходиться")

# ==================== ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ ====================
plt.figure(figsize=(10, 8))

# Контуры
X, Y = np.meshgrid(x_vals, y_vals)
plt.contour(X, Y, f1(X, Y), levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, f2(X, Y), levels=[0], colors='red', linewidths=2)

# Траектории
hist_n = np.array(hist_n)
plt.plot(hist_n[:, 0], hist_n[:, 1], 'go-', markersize=6, label='Ньютон')

hist_si = np.array(hist_si)
plt.plot(hist_si[:, 0], hist_si[:, 1], 'ms-', markersize=4, label='Простая итерация')

hist_z = np.array(hist_z)
plt.plot(hist_z[:, 0], hist_z[:, 1], 'c^-', markersize=4, label='Зейдель')

plt.plot(root_n[0], root_n[1], 'ro', markersize=12, label='Решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение сходимости методов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("convergence.png")

# ==================== ИТОГОВАЯ ТАБЛИЦА ====================
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*80)
print(f"{'Метод':<20} {'x':<15} {'y':<15} {'max|f|':<15} {'Итераций':<10}")
print("-"*80)
print(f"{'Ньютона':<20} {root_n[0]:<15.8f} {root_n[1]:<15.8f} "
      f"{max(abs(f1(*root_n)), abs(f2(*root_n))):<15.2e} {it_n:<10}")
print(f"{'Простая итерация':<20} {root_si[0]:<15.8f} {root_si[1]:<15.8f} "
      f"{max(abs(f1(*root_si)), abs(f2(*root_si))):<15.2e} {it_si:<10}")
print(f"{'Зейделя':<20} {root_z[0]:<15.8f} {root_z[1]:<15.8f} "
      f"{max(abs(f1(*root_z)), abs(f2(*root_z))):<15.2e} {it_z:<10}")

print("\n" + "="*60)
print(" Работа завершена")
print("="*60)