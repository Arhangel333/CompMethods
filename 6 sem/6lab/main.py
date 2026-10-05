import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import input
from prettytable import PrettyTable

def parse_equation(eq_str, x_var, y_var):
    """Парсит строку уравнения y' = f(x,y)"""
    
    rhs = sp.sympify(eq_str)
    return sp.lambdify((x_var, y_var), rhs, 'numpy')

def parse_analytic(analytic_str, x_var):
    """Парсит аналитическое решение"""
    expr = sp.sympify(analytic_str)
    return sp.lambdify(x_var, expr, 'numpy')

def euler_cauchy(f, x0, y0, x_end, h):
    """Метод Эйлера-Коши"""
    n = int((x_end - x0) / h) + 1
    x_vals = np.linspace(x0, x_end, n)
    y_vals = np.zeros(n)
    y_vals[0] = y0
    
    for i in range(n-1):
        x = x_vals[i]
        y = y_vals[i]
        
        
        y_p = y + h * f(x, y)
        
        
        y_vals[i+1] = y + (h/2) * (f(x, y) + f(x_vals[i+1], y_p))
    
    return x_vals, y_vals

def runge_kutta_4(f, x0, y0, x_end, h):
    """Метод Рунге-Кутты 4-го порядка"""
    n = int((x_end - x0) / h) + 1
    x_vals = np.linspace(x0, x_end, n)
    y_vals = np.zeros(n)
    y_vals[0] = y0
    
    for i in range(n-1):
        x = x_vals[i]
        y = y_vals[i]
        
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2)*k1)
        k3 = f(x + h/2, y + (h/2)*k2)
        k4 = f(x + h, y + h*k3)
        
        y_vals[i+1] = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_vals, y_vals

def analyze_step_effect(f, analytic_func, x0, y0, x_end, h_values):
    """Исследование влияния шага на погрешность"""
    results = []
    
    for h in h_values:
        
        x_e, y_e = euler_cauchy(f, x0, y0, x_end, h)
        y_analytic = analytic_func(x_e)
        
        
        error_e = np.max(np.abs(y_e - y_analytic))
        
        
        x_r, y_r = runge_kutta_4(f, x0, y0, x_end, h)
        error_r = np.max(np.abs(y_r - y_analytic))
        
        results.append({
            'h': h,
            'n_steps': len(x_e) - 1,
            'error_euler_cauchy': error_e,
            'error_runge_kutta': error_r
        })
    
    return results


def plot_comparison(x_vals, y_analytic, y_numeric, method_name, h, x0, x_end):
    """Построение графика сравнения численного и аналитического решений"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_vals, y_analytic, 'b-', linewidth=2, label='Аналитическое решение')
    plt.plot(x_vals, y_numeric, 'ro--', markersize=4, linewidth=1, label=f'{method_name}, h={h}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{method_name}: сравнение с аналитическим решением\n(h={h})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    error = np.abs(y_numeric - y_analytic)
    plt.semilogy(x_vals, error, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('|y_numeric - y_analytic|')
    plt.title(f'Погрешность {method_name} (логарифмическая шкала)')
    plt.grid(True)
    
    plt.tight_layout()
    return plt

def plot_step_effect(x0, y0, x_end, f, analytic_func, h_values, method_name):
    """Исследование влияния шага на погрешность с графиками"""
    
    plt.figure(figsize=(14, 10))
    
    # График 1: решения с разными шагами
    plt.subplot(2, 2, 1)
    x_fine = np.linspace(x0, x_end, 1000)
    y_fine = analytic_func(x_fine)
    plt.plot(x_fine, y_fine, 'k-', linewidth=2, label='Аналитическое решение', alpha=0.7)
    
    colors = ['r', 'g', 'b', 'm']
    for i, h in enumerate(h_values):
        if method_name == "Euler-Cauchy":
            x_vals, y_vals = euler_cauchy(f, x0, y0, x_end, h)
        else:
            x_vals, y_vals = runge_kutta_4(f, x0, y0, x_end, h)
        
        plt.plot(x_vals, y_vals, 'o--', color=colors[i], markersize=3, 
                linewidth=1, label=f'h={h}', alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{method_name}: решения с разными шагами')
    plt.legend()
    plt.grid(True)
    
    # График 2: зависимость погрешности от шага
    plt.subplot(2, 2, 2)
    h_list = []
    errors = []
    
    for h in h_values:
        if method_name == "Euler-Cauchy":
            x_vals, y_vals = euler_cauchy(f, x0, y0, x_end, h)
        else:
            x_vals, y_vals = runge_kutta_4(f, x0, y0, x_end, h)
        
        y_analytic = np.array([analytic_func(xi) for xi in x_vals])
        max_error = np.max(np.abs(y_vals - y_analytic))
        
        h_list.append(h)
        errors.append(max_error)
    
    plt.loglog(h_list, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Шаг h (логарифмическая шкала)')
    plt.ylabel('Максимальная погрешность (логарифмическая шкала)')
    plt.title(f'{method_name}: зависимость погрешности от шага')
    plt.grid(True)
    
    # Добавляем теоретические линии для сравнения
    h_theory = np.array([h_list[0], h_list[-1]])
    if method_name == "Euler-Cauchy":
        error_theory_2nd = errors[0] * (h_theory / h_list[0])**2
        plt.loglog(h_theory, error_theory_2nd, 'r--', linewidth=1.5, 
                  label='Теория (2-й порядок)')
    else:
        error_theory_4th = errors[0] * (h_theory / h_list[0])**4
        plt.loglog(h_theory, error_theory_4th, 'r--', linewidth=1.5, 
                  label='Теория (4-й порядок)')
    
    plt.legend()
    
    # График 3: погрешность вдоль интервала для разных шагов
    plt.subplot(2, 2, 3)
    for i, h in enumerate(h_values[:3]):  # показываем первые 3 шага для наглядности
        if method_name == "Euler-Cauchy":
            x_vals, y_vals = euler_cauchy(f, x0, y0, x_end, h)
        else:
            x_vals, y_vals = runge_kutta_4(f, x0, y0, x_end, h)
        
        y_analytic = np.array([analytic_func(xi) for xi in x_vals])
        error = np.abs(y_vals - y_analytic)
        
        plt.semilogy(x_vals, error, 'o-', color=colors[i], markersize=3, 
                    linewidth=1, label=f'h={h}')
    
    plt.xlabel('x')
    plt.ylabel('Погрешность (логарифмическая шкала)')
    plt.title(f'{method_name}: распределение погрешности по x')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt

def plot_both_methods_comparison(x0, y0, x_end, f, analytic_func, h):
    """Сравнение двух методов при одинаковом шаге"""
    
    # Решаем обоими методами
    x_ec, y_ec = euler_cauchy(f, x0, y0, x_end, h)
    x_rk, y_rk = runge_kutta_4(f, x0, y0, x_end, h)
    
    # Точное решение
    x_fine = np.linspace(x0, x_end, 500)
    y_fine = analytic_func(x_fine)
    
    plt.figure(figsize=(14, 6))
    
    # График 1: сравнение решений
    plt.subplot(1, 2, 1)
    plt.plot(x_fine, y_fine, 'k-', linewidth=2, label='Аналитическое', alpha=0.8)
    plt.plot(x_ec, y_ec, 'ro-', markersize=4, linewidth=1, label='Эйлер-Коши', alpha=0.7)
    plt.plot(x_rk, y_rk, 'bs-', markersize=4, linewidth=1, label='Рунге-Кутта 4', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Сравнение методов при h={h}')
    plt.legend()
    plt.grid(True)
    
    # График 2: погрешности
    plt.subplot(1, 2, 2)
    y_analytic_ec = np.array([analytic_func(xi) for xi in x_ec])
    y_analytic_rk = np.array([analytic_func(xi) for xi in x_rk])
    
    error_ec = np.abs(y_ec - y_analytic_ec)
    error_rk = np.abs(y_rk - y_analytic_rk)
    
    plt.semilogy(x_ec, error_ec, 'ro-', markersize=4, linewidth=1, label='Эйлер-Коши')
    plt.semilogy(x_rk, error_rk, 'bs-', markersize=4, linewidth=1, label='Рунге-Кутта 4')
    plt.xlabel('x')
    plt.ylabel('Погрешность (логарифмическая шкала)')
    plt.title(f'Сравнение погрешностей при h={h}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt


def main():
    print("="*80)
    print("РЕШЕНИЕ ЗАДАЧИ КОШИ ДЛЯ ОДУ ПЕРВОГО ПОРЯДКА")
    print("="*80)
    
    

    
    eq_str = input.eq_str.strip()
    
    
    x = sp.Symbol('x')
    y = sp.Symbol('y')   
    yp = sp.Derivative(y, x)  

    
    equation = x**2 * yp - y**2 - x*y

    
    solution = sp.solve(equation, yp)

    
    eq_str = str(solution[0])  
    
    
    
    print(f"\nВаше дифференциальное уравнение в виде y' = f(x,y):")
    print(f"y' = {eq_str}")

    
    
    analytic_str = input.analytic_str.strip()
    print(f"\nВаше аналитическое решение y(x): {analytic_str}")
    
    x0 = input.x0
    
    y0 = input.y0
    print(f"\n x0 = {x0}, y0 = {y0}")
    
    x_end = input.x_end
    print(f"\n x = [{x0}, {x_end}]")

    
    h_main = input.h_main
    print(f"\n h = {h_main}")

    
    x_var = sp.Symbol('x')
    y_var = sp.Symbol('y')
    
    
    f = parse_equation(eq_str, x_var, y_var)
    analytic_func = parse_analytic(analytic_str, x_var)
    
    print("\n" + "="*80)
    print("ОСНОВНОЙ РАСЧЁТ (h = {})".format(h_main))
    print("="*80)
    
    
    x_euler, y_euler = euler_cauchy(f, x0, y0, x_end, h_main)
    x_rk4, y_rk4 = runge_kutta_4(f, x0, y0, x_end, h_main)
    
    
    y_analytic = analytic_func(x_euler)
    
    
    table = PrettyTable()
    table.field_names = ["x", "Аналит. решение", "Эйлер-Коши", "Ошибка E-K", 
                        "Рунге-Кутта 4", "Ошибка RK4"]
    table.align = "r"
    table.float_format = ".6"
    
    for i in range(len(x_euler)):
        error_ec = abs(y_euler[i] - y_analytic[i])
        error_rk4 = abs(y_rk4[i] - y_analytic[i])
        
        table.add_row([
            x_euler[i],
            y_analytic[i],
            y_euler[i],
            error_ec,
            y_rk4[i],
            error_rk4
        ])
    
    print(table)
    
    
    max_error_ec = np.max(np.abs(y_euler - y_analytic))
    max_error_rk4 = np.max(np.abs(y_rk4 - y_analytic))
    mean_error_ec = np.mean(np.abs(y_euler - y_analytic))
    mean_error_rk4 = np.mean(np.abs(y_rk4 - y_analytic))
    
    print("\n" + "="*80)
    print("СТАТИСТИКА ПОГРЕШНОСТЕЙ")
    print("="*80)
    print(f"Метод Эйлера-Коши:")
    print(f"  Максимальная погрешность: {max_error_ec:.6e}")
    print(f"  Средняя погрешность:      {mean_error_ec:.6e}")
    print(f"\nМетод Рунге-Кутты 4-го порядка:")
    print(f"  Максимальная погрешность: {max_error_rk4:.6e}")
    print(f"  Средняя погрешность:      {mean_error_rk4:.6e}")
    
    
    print("\n" + "="*80)
    print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ ШАГА ИНТЕГРИРОВАНИЯ")
    print("="*80)
    
    h_values = [h_main, h_main/2, h_main/4, h_main/8]
    step_results = analyze_step_effect(f, analytic_func, x0, y0, x_end, h_values)
    
    step_table = PrettyTable()
    step_table.field_names = ["Шаг h", "Число шагов", "Макс. ошибка E-K", "Макс. ошибка RK4"]
    step_table.align = "r"
    
    for res in step_results:
        step_table.add_row([
            res['h'],
            res['n_steps'],
            f"{res['error_euler_cauchy']:.6e}",
            f"{res['error_runge_kutta']:.6e}"
        ])
    
    print(step_table)
    
    
    print("\n" + "="*80)
    print("АНАЛИЗ СХОДИМОСТИ")
    print("="*80)
    print("При уменьшении шага в 2 раза:")
    
    for i in range(len(step_results)-1):
        h1 = step_results[i]['h']
        h2 = step_results[i+1]['h']
        ratio_ec = step_results[i]['error_euler_cauchy'] / step_results[i+1]['error_euler_cauchy']
        ratio_rk4 = step_results[i]['error_runge_kutta'] / step_results[i+1]['error_runge_kutta']
        
        print(f"\nh = {h1} -> {h2}:")
        print(f"  Эйлер-Коши:   ошибка уменьшилась в {ratio_ec:.2f} раз (теоретически ~4)")
        print(f"  Рунге-Кутта4: ошибка уменьшилась в {ratio_rk4:.2f} раз (теоретически ~16)")
    





 # График 1: сравнение методов при основном шаге
    plot_both_methods_comparison(x0, y0, x_end, f, analytic_func, h_main)
    plt.savefig('methods_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ График 'methods_comparison.png' сохранён")
    
    # График 2: влияние шага для метода Эйлера-Коши
    h_values = [h_main, h_main/2, h_main/4, h_main/8]
    plot_step_effect(x0, y0, x_end, f, analytic_func, h_values, "Euler-Cauchy")
    plt.savefig('euler_cauchy_step_effect.png', dpi=150, bbox_inches='tight')
    print("✓ График 'euler_cauchy_step_effect.png' сохранён")
    
    # График 3: влияние шага для метода Рунге-Кутты 4
    plot_step_effect(x0, y0, x_end, f, analytic_func, h_values, "Runge-Kutta 4")
    plt.savefig('runge_kutta_step_effect.png', dpi=150, bbox_inches='tight')
    print("✓ График 'runge_kutta_step_effect.png' сохранён")
    
    # График 4: сравнение при основном шаге (детальный)
    plot_comparison(x_euler, y_analytic, y_euler, "Эйлер-Коши", h_main, x0, x_end)
    plt.savefig('euler_cauchy_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ График 'euler_cauchy_comparison.png' сохранён")
    
    plot_comparison(x_rk4, y_analytic, y_rk4, "Рунге-Кутта 4", h_main, x0, x_end)
    plt.savefig('runge_kutta_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ График 'runge_kutta_comparison.png' сохранён")
        

if __name__ == "__main__":
    main()
