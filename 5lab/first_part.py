import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import sympy as sp
from sympy import lambdify

import input as inp

class NonlinearSolver:
    """
    Универсальный решатель нелинейных уравнений.
    Требует только функцию f(x). Всё остальное вычисляется автоматически.
    """
    
    def __init__(self, f_expr: str):
        """
        f_expr - строковое выражение функции, например "x**3 - 3*x + 1"
        """
        self.f_expr = f_expr
        
        # Символьное вычисление производной и phi(x)
        self.x_symbol = sp.Symbol('x')
        self.f_sym = sp.sympify(f_expr)
        self.f_prime_sym = sp.diff(self.f_sym, self.x_symbol)
        
        # Преобразуем в числовые функции
        self.f = lambdify(self.x_symbol, self.f_sym, 'numpy')
        self.f_prime = lambdify(self.x_symbol, self.f_prime_sym, 'numpy')
        
        # Для метода простой итерации (будет подобран автоматически)
        self.alpha = None
        self.phi = None
        self.phi_prime = None
        
        print(f"\n Исходная функция: f(x) = {self.f_expr}")
        print(f" Производная (вычислена автоматически): f'(x) = {self.f_prime_sym}")
    
    def build_phi_auto(self, x0: float) -> bool:
        """
        Автоматическое построение phi(x) = x - alpha * f(x)
        Подбирает alpha так, чтобы |phi'(x0)| < 1
        """
        f_prime0 = float(self.f_prime(x0))
        
        if abs(f_prime0) < 1e-10:
            print(f"   f'(x0) = {f_prime0:.6f} близка к нулю")
            print(f"  Использую alpha = 0.1")
            self.alpha = 0.1
        else:
            # Условие: |1 - alpha * f'(x0)| < 1
            # Решаем: -1 < 1 - alpha*f'(x0) < 1
            # 0 < alpha*f'(x0) < 2
            
            if f_prime0 > 0:
                # 0 < alpha < 2/f'(x0)
                alpha_max = 2.0 / f_prime0
                self.alpha = alpha_max * 0.9  # чуть меньше максимума для запаса
            else:  # f_prime0 < 0
                # 2/f'(x0) < alpha < 0 (alpha отрицательное)
                alpha_min = 2.0 / f_prime0
                self.alpha = alpha_min * 0.9
        
        # Строим phi(x)
        self.phi = lambda x: x - self.alpha * self.f(x)
        self.phi_prime = lambda x: 1 - self.alpha * self.f_prime(x)
        
        phi_prime0 = abs(self.phi_prime(x0))
        print(f"   Подобран alpha = {self.alpha:.6f}")
        print(f"   |φ'(x0)| = {phi_prime0:.4f} {'✓ < 1' if phi_prime0 < 1 else '✗ ≥ 1'}")
        
        return phi_prime0 < 1
    
    def find_intervals(self, x_range: Tuple[float, float], n_points: int = 1000) -> list:
        """
        Автоматический поиск отрезков, содержащих корни
        """
        x_min, x_max = x_range
        x_vals = np.linspace(x_min, x_max, n_points)
        intervals = []
        
        for i in range(len(x_vals) - 1):
            f1 = self.f(x_vals[i])
            f2 = self.f(x_vals[i + 1])
            
            if f1 * f2 < 0:
                intervals.append([x_vals[i], x_vals[i + 1]])
            elif abs(f1) < 1e-10:
                intervals.append([x_vals[i], x_vals[i]])
        
        return intervals
    
    # ==================== МЕТОД ДИХОТОМИИ ====================
    def bisection(self, a: float, b: float, eps: float = 1e-3, 
                  max_iter: int = 100) -> Tuple[float, int, list]:
        """
        Метод половинного деления
        """
        if self.f(a) * self.f(b) >= 0:
            raise ValueError(f"На отрезке [{a}, {b}] нет корня")
        
        history = []
        
        for iteration in range(max_iter):
            c = (a + b) / 2
            history.append(c)
            
            if abs(b - a) / 2 < eps:
                return c, iteration + 1, history
            
            if self.f(a) * self.f(c) < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2, max_iter, history
    
    # ==================== МЕТОД ПРОСТОЙ ИТЕРАЦИИ ====================
    def simple_iteration(self, x0: float, eps: float = 1e-3, 
                         max_iter: int = 100) -> Tuple[float, int, list]:
        """
        Метод простой итерации с автоматическим подбором phi(x)
        """
        print(f"\n   Построение φ(x) для точки x0 = {x0:.6f}")
        self.build_phi_auto(x0)
        
        x_prev = x0
        history = [x_prev]
        
        for iteration in range(max_iter):
            x_next = self.phi(x_prev)
            history.append(x_next)
            
            if abs(x_next - x_prev) < eps:
                return x_next, iteration + 1, history
            
            x_prev = x_next
        
        return x_prev, max_iter, history
    
    # ==================== МЕТОД НЬЮТОНА ====================
    def newton(self, x0: float, eps: float = 1e-3, 
               max_iter: int = 100) -> Tuple[float, int, list]:
        """
        Метод Ньютона (использует автоматически вычисленную производную)
        """
        x_prev = x0
        history = [x_prev]
        
        for iteration in range(max_iter):
            f_val = self.f(x_prev)
            f_prime_val = self.f_prime(x_prev)
            
            if abs(f_prime_val) < 1e-12:
                raise ValueError(f"Производная близка к нулю в x = {x_prev}")
            
            x_next = x_prev - f_val / f_prime_val
            history.append(x_next)
            
            if abs(x_next - x_prev) < eps:
                return x_next, iteration + 1, history
            
            x_prev = x_next
        
        return x_prev, max_iter, history


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_function(f: Callable, x_range: Tuple[float, float], 
                  roots: list = None, title: str = None):
    """Построение графика функции"""
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, 500)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='r', linestyle='--', label='y = 0')
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title or 'График функции')
    plt.legend()
    
    if roots:
        for root in roots:
            plt.plot(root, 0, 'ro', markersize=8)
            plt.annotate(f'{root:.6f}', (root, 0), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.savefig(inp.graphic)
    print(f" График сохранён в файл: {inp.graphic}")


def print_results_table(results: dict):
    """Вывод результатов в виде таблицы"""
    print("\n" + "="*100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*100)
    print(f"{'№':<4} {'Метод':<22} {'Корень':<15} {'f(корень)':<15} {'Итераций':<10} {'Примечание':<30}")
    print("-"*100)
    
    for i, (method_name, data) in enumerate(results.items(), 1):
        if data['success']:
            note = data.get('note', '✓')
            print(f"{i:<4} {method_name:<22} {data['root']:<15.6f} "
                  f"{data['f_root']:<15.2e} {data['iterations']:<10} {note:<30}")
        else:
            print(f"{i:<4} {method_name:<22} {'—':<15} {'—':<15} {'—':<10} {data['error']:<30}")


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("="*60)
    print("РЕШЕНИЕ НЕЛИНЕЙНЫХ УРАВНЕНИЙ")
    print("="*60)
    
    # ----- ВВОД ДАННЫХ ПОЛЬЗОВАТЕЛЕМ -----
    print("\n Введите уравнение в виде f(x) = 0")
    print("   Примеры:")
    print("     x**3 - 3*x + 1")
    print("     sin(x) - x/2")
    print("     exp(x) - 2*x - 1")
    print("     log(x) + x**2 - 2")
    print("     log(x**2+1)/log(10) = log10(x**2+1)")
    
    
    f_input = inp.f_input
    
    if not f_input:
        f_input = "x**3 - 3*x + 1"  # пример по умолчанию
    print(f"\n   Используем: {f_input}")
    
    eps_input = inp.eps
    eps = float(eps_input) if eps_input else 0.001
    
    x_min_input = inp.left
    x_max_input = inp.right
    
    x_min = float(x_min_input) if x_min_input else -3
    x_max = float(x_max_input) if x_max_input else 3
    x_range = (x_min, x_max)
    
    # ----- СОЗДАНИЕ РЕШАТЕЛЯ -----
    print("\n" + "="*60)
    print("АНАЛИЗ ФУНКЦИИ")
    print("="*60)
    
    try:
        solver = NonlinearSolver(f_input)
    except Exception as e:
        print(f" Ошибка: {e}")
        print("   Проверьте правильность ввода функции")
        return
    
    # ----- ПОИСК ОТРЕЗКОВ С КОРНЯМИ -----
    print(f"\n Поиск корней на отрезке [{x_range[0]}, {x_range[1]}]...")
    intervals = solver.find_intervals(x_range)
    
    if not intervals:
        print(" Корней не найдено. Попробуйте расширить диапазон поиска.")
        return
    
    print(f"\n Найдено отрезков с корнями: {len(intervals)}")
    for i, (a, b) in enumerate(intervals, 1):
        if a == b:
            print(f"   {i}. x = {a:.6f} (точный корень)")
        else:
            print(f"   {i}. [{a:.6f}, {b:.6f}] (длина = {b-a:.6f})")
    
    # ----- ГРАФИК ФУНКЦИИ -----
    plot_function(solver.f, x_range, title=f"f(x) = {f_input}")
    
    # ----- РЕШЕНИЕ ДЛЯ КАЖДОГО КОРНЯ -----
    all_results = {}
    
    for idx, (a, b) in enumerate(intervals, 1):
        print(f"\n{'='*60}")
        print(f"КОРЕНЬ №{idx}")
        
        if a == b:
            print(f" Точный корень: x = {a:.10f}")
            x0 = a
            results = {}
            
            # Проверка значения функции
            results['Проверка'] = {
                'success': True,
                'root': a,
                'f_root': solver.f(a),
                'iterations': 0,
                'note': 'точный корень'
            }
            
        else:
            print(f" Отрезок: [{a:.10f}, {b:.10f}]")
            x0 = (a + b) / 2
            print(f" Начальное приближение: x0 = {x0:.10f}")
            
            results = {}
            
            # 1. Метод дихотомии
            print(f"\n{'─'*40}")
            print("МЕТОД ДИХОТОМИИ")
            try:
                root, iterations, history = solver.bisection(a, b, eps)
                results['Дихотомия'] = {
                    'success': True,
                    'root': root,
                    'f_root': solver.f(root),
                    'iterations': iterations,
                    'note': 'всегда сходится'
                }
                print(f"  ✓ Корень: {root:.10f}")
                print(f"  ✓ f(корень) = {solver.f(root):.2e}")
                print(f"  ✓ Итераций: {iterations}")
            except Exception as e:
                results['Дихотомия'] = {'success': False, 'error': str(e)}
                print(f"  ✗ Ошибка: {e}")
            
            # 2. Метод простой итерации
            print(f"\n{'─'*40}")
            print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
            try:
                root, iterations, history = solver.simple_iteration(x0, eps)
                # Проверка сходимости
                if solver.phi_prime:
                    phi_prime_root = abs(solver.phi_prime(root))
                    note = f"|φ'(x*)| = {phi_prime_root:.4f} {'< 1 ✓' if phi_prime_root < 1 else '≥ 1 ✗'}"
                else:
                    note = "сходимость не проверена"
                
                results['Простая итерация'] = {
                    'success': True,
                    'root': root,
                    'f_root': solver.f(root),
                    'iterations': iterations,
                    'note': note
                }
                print(f"  ✓ Корень: {root:.10f}")
                print(f"  ✓ f(корень) = {solver.f(root):.2e}")
                print(f"  ✓ Итераций: {iterations}")
            except Exception as e:
                results['Простая итерация'] = {'success': False, 'error': str(e)}
                print(f"  ✗ Ошибка: {e}")
            
            # 3. Метод Ньютона
            print(f"\n{'─'*40}")
            print("МЕТОД НЬЮТОНА")
            try:
                root, iterations, history = solver.newton(x0, eps)
                results['Ньютона'] = {
                    'success': True,
                    'root': root,
                    'f_root': solver.f(root),
                    'iterations': iterations,
                    'note': 'квадратичная сходимость'
                }
                print(f"  ✓ Корень: {root:.10f}")
                print(f"  ✓ f(корень) = {solver.f(root):.2e}")
                print(f"  ✓ Итераций: {iterations}")
            except Exception as e:
                results['Ньютона'] = {'success': False, 'error': str(e)}
                print(f"  ✗ Ошибка: {e}")
        
        all_results[f"Корень_{idx}"] = results
    
    # ----- ИТОГОВАЯ ТАБЛИЦА -----
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    
    for root_name, results in all_results.items():
        print(f"\n {root_name.replace('_', ' ')}:")
        print_results_table(results)
    
    print("\n" + "="*60)
    print(" Работа завершена")
    print("="*60)


if __name__ == "__main__":
    main()