import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import input

class Interpolation:
    def __init__(self, x_values, y_values):
        """Инициализация с табличными данными"""
        self.x = np.array(x_values, dtype=float)
        self.y = np.array(y_values, dtype=float)
        self.n = len(x_values)

    def choose_nodes(self, x_star, degree):
        """Выбирает ближайшие к x* узлы для интерполяции заданной степени"""
        n_needed = degree + 1  # нужно degree+1 узлов

        # Сортируем узлы по расстоянию до x*
        distances = [(abs(x - x_star), i) for i, x in enumerate(self.x)]
        distances.sort()

        # Берем индексы ближайших узлов
        indices = [distances[i][1] for i in range(n_needed)]
        indices.sort()  # сортируем по возрастанию x

        return indices
        
    """ def lagrange(self, x_points, y_points, x_star):
        
        n = len(x_points)
        result = 0

        for i in range(n):
            # Базисный полином Лагранжа
            term = y_points[i]
            for j in range(n):
                if j != i:
                    term *= (x_star - x_points[j]) / (x_points[i] - x_points[j])
            result += term

        return result """
        

    def lagrange(self, x_points, y_points, x_star):
        """
        Интерполяционный многочлен Лагранжа
        Возвращает: (значение в точке x_star, коэффициенты многочлена)
        
        Коэффициенты многочлена Лагранжа в форме:
        L(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n
        """
        n = len(x_points)
        result = 0
        
        # Собираем все слагаемые для последующего вычисления коэффициентов
        # Каждый базисный полином ℓ_i(x) — это многочлен степени n-1
        # Мы будем хранить коэффициенты в массиве coeffs_total
        
        # Инициализируем массив коэффициентов нулями
        coeffs_total = np.zeros(n)
        
        for i in range(n):
            # Базисный полином Лагранжа ℓ_i(x)
            # Сначала создаем полином, который начинается как 1
            poly_coeffs = np.array([1.0])  # коэффициенты от младшей степени к старшей
            
            for j in range(n):
                if j != i:
                    # Умножаем на (x - x_j) / (x_i - x_j)
                    # (x - x_j) = -x_j + 1*x
                    factor = np.array([-x_points[j], 1.0])  # коэффициенты: [-x_j, 1]
                    # Делим на знаменатель
                    factor = factor / (x_points[i] - x_points[j])
                    # Умножаем полиномы
                    poly_coeffs = np.polymul(poly_coeffs, factor)
            
            # Умножаем на y_i
            poly_coeffs = poly_coeffs * y_points[i]
            
            # Добавляем к общим коэффициентам
            # Приводим к одинаковой длине
            if len(poly_coeffs) > len(coeffs_total):
                coeffs_total = np.pad(coeffs_total, (0, len(poly_coeffs) - len(coeffs_total)), 'constant')
            coeffs_total[:len(poly_coeffs)] += poly_coeffs
            
            # Вычисляем значение в точке x_star для текущего слагаемого
            term = y_points[i]
            for j in range(n):
                if j != i:
                    term *= (x_star - x_points[j]) / (x_points[i] - x_points[j])
            result += term
        
        # Обрезаем лишние нули в конце
        while len(coeffs_total) > 1 and abs(coeffs_total[-1]) < 1e-12:
            coeffs_total = coeffs_total[:-1]
        
        return result, coeffs_total

    def newton(self, x_points, y_points, x_star):
        """Интерполяционный многочлен Ньютона"""
        n = len(x_points)

        # Создаем таблицу разделенных разностей
        diff_table = np.zeros((n, n))
        diff_table[:, 0] = y_points  # нулевой столбец — значения y

        # Заполняем таблицу
        for j in range(1, n):
            for i in range(n - j):
                diff_table[i][j] = (diff_table[i+1][j-1] - diff_table[i][j-1]) / (x_points[i+j] - x_points[i])

        # Коэффициенты многочлена — первая строка таблицы
        coefficients = diff_table[0, :]

        # Вычисляем значение многочлена
        result = coefficients[0]
        product = 1

        for i in range(1, n):
            product *= (x_star - x_points[i-1])
            result += coefficients[i] * product

        return result, coefficients

    def estimate_error(self, x_points, y_points, x_star, coefficients, degree):
        """
        Оценка погрешности интерполяции
        Используем остаточный член: |R_n(x)| ≈ |coeff_{n+1}| * |(x-x_0)...(x-x_n)|
        """
        n = len(x_points)

        # Вычисляем произведение (x - x_i)
        product = 1
        for i in range(n):
            product *= (x_star - x_points[i])

        # Оценка погрешности (используем следующий коэффициент как приближение производной)
        if len(coefficients) > n:
            error = abs(coefficients[n]) * abs(product)
        else:
            # Если нет следующего коэффициента, используем разность между разными степенями
            error = None

        return error

    def interpolate_at_point(self, x_star, degree):
        """
        Выполняет интерполяцию в точке x* многочленом заданной степени
        """
        # Выбираем узлы для интерполяции
        indices = self.choose_nodes(x_star, degree)
        x_nodes = [self.x[i] for i in indices]
        y_nodes = [self.y[i] for i in indices]

        print(f"\n{'='*60}")
        print(f"ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНОМ {degree}-Й СТЕПЕНИ")
        print(f"{'='*60}")
        print(f"Выбранные узлы:")
        for i in range(len(x_nodes)):
            print(f"  x[{i}] = {x_nodes[i]}, y[{i}] = {y_nodes[i]}")
        print(f"Точка интерполяции x* = {x_star}")

        # Метод Лагранжа
        lagrange_result, coefficients = self.lagrange(x_nodes, y_nodes, x_star)
        print(f"\nМногочлен Лагранжа {degree}-й степени:")
        print(f"  L_{degree}({x_star}) = {lagrange_result:.6f}")
        for i, coeff in enumerate(coefficients):
            print(f"    a_{i} = {coeff:.6f}")

        # Метод Ньютона
        newton_result, coefficients = self.newton(x_nodes, y_nodes, x_star)
        print(f"\nМногочлен Ньютона {degree}-й степени:")
        print(f"  P_{degree}({x_star}) = {newton_result:.6f}")
        print(f"  Коэффициенты многочлена:")
        for i, coeff in enumerate(coefficients):
            print(f"    a_{i} = {coeff:.6f}")

        print(f"\n  Уравнение Ньютона (интерполяционная форма):")
        newton_eq = f"    P_{degree}(x) = {coefficients[0]:.8f}"
        for i in range(1, degree + 1):
            # Формируем произведение (x - x_0)(x - x_1)...(x - x_{i-1})
            product_str = ""
            for j in range(i):
                if j == 0:
                    product_str = f"(x - {x_nodes[j]})"
                else:
                    product_str += f"(x - {x_nodes[j]})"
            
            if coefficients[i] >= 0:
                newton_eq += f" + {coefficients[i]:.8f}·{product_str}"
            else:
                newton_eq += f" - {abs(coefficients[i]):.8f}·{product_str}"
        print(newton_eq)
        
        # Вывод уравнения Лагранжа
        print(f"\n  Уравнение Лагранжа (сумма базисных полиномов):")
        print(f"    L_{degree}(x) =", end="")
        
        for i in range(len(x_nodes)):
            # Формируем базисный полином ℓ_i(x)
            numerator = ""
            denominator = ""
            for j in range(len(x_nodes)):
                if j != i:
                    if numerator:
                        numerator += f"(x - {x_nodes[j]})"
                    else:
                        numerator = f"(x - {x_nodes[j]})"
                    
                    if denominator:
                        denominator += f"({x_nodes[i]} - {x_nodes[j]})"
                    else:
                        denominator = f"({x_nodes[i]} - {x_nodes[j]})"
            
            if i == 0:
                print(f" {y_nodes[i]}·[{numerator}/{denominator}]", end="")
            else:
                if y_nodes[i] >= 0:
                    print(f" + {y_nodes[i]}·[{numerator}/{denominator}]", end="")
                else:
                    print(f" - {abs(y_nodes[i])}·[{numerator}/{denominator}]", end="")
        print()
        
        # Оценка погрешности
        if degree < len(self.x) - 1:
            # Строим многочлен следующей степени для оценки погрешности
            indices_next = self.choose_nodes(x_star, degree + 1)
            x_next = [self.x[i] for i in indices_next]
            y_next = [self.y[i] for i in indices_next]
            _, coeff_next = self.newton(x_next, y_next, x_star)

            # Оцениваем погрешность
            product = 1
            for i in range(len(x_nodes)):
                product *= (x_star - x_nodes[i])

            error_estimate = abs(coeff_next[degree + 1]) * abs(product) if len(coeff_next) > degree + 1 else None

            if error_estimate:
                print(f"\nОценка погрешности:")
                print(f"  |R_{degree}({x_star})| ≈ {error_estimate:.6f}")

        return lagrange_result, newton_result

    def plot_interpolation(self, x_star=None, degrees=[2, 3]):
        """
        Строит графики интерполяционных многочленов Лагранжа и исходных точек
        degrees: степени многочленов (2 и 3)
        """
        plt.figure(figsize=(12, 8))
        
        # Исходные точки
        plt.scatter(self.x, self.y, color='black', s=100, 
                label='Исходные данные', zorder=5)
        
        # Гладкие точки для построения графиков
        x_smooth = np.linspace(min(self.x) - 0.3, max(self.x) + 0.3, 500)
        
        # Цвета для разных степеней
        colors = ['red', 'green', 'blue']
        
        for deg, color in zip(degrees, colors[:len(degrees)]):
            # Для каждой точки x_smooth строим интерполяционный многочлен
            y_smooth = []
            for xi in x_smooth:
                # Выбираем узлы для данной точки
                indices = self.choose_nodes(xi, deg)
                x_nodes = [self.x[i] for i in indices]
                y_nodes = [self.y[i] for i in indices]
                # Сортируем узлы по возрастанию x
                sorted_pairs = sorted(zip(x_nodes, y_nodes))
                x_sorted = [p[0] for p in sorted_pairs]
                y_sorted = [p[1] for p in sorted_pairs]
                # Вычисляем значение многочлена Лагранжа
                val, coeffs_lagrange = self.lagrange(x_sorted, y_sorted, xi)
                y_smooth.append(val)
            
            plt.plot(x_smooth, y_smooth, color=color, linewidth=2,
                    label=f'Интерполяция {deg}-й степени (Лагранж)')
        
        # Отмечаем точку x*, если задана
        if x_star is not None:
            plt.axvline(x=x_star, color='purple', linestyle='--', alpha=0.7,
                    label=f'x* = {x_star}')
            
            for deg, color in zip(degrees, colors[:len(degrees)]):
                indices = self.choose_nodes(x_star, deg)
                x_nodes = [self.x[i] for i in indices]
                y_nodes = [self.y[i] for i in indices]
                sorted_pairs = sorted(zip(x_nodes, y_nodes))
                x_sorted = [p[0] for p in sorted_pairs]
                y_sorted = [p[1] for p in sorted_pairs]
                value, coeffs_lagrange = self.lagrange(x_sorted, y_sorted, x_star)
                plt.plot(x_star, value, 'o', color=color, markersize=10, zorder=10)
                plt.annotate(f'P_{deg}({x_star})={value:.3f}',
                            xy=(x_star, value), xytext=(5, 5),
                            textcoords='offset points', fontsize=9, color=color)
        
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('Интерполяция табличных данных (многочлены Лагранжа)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig('interpolation_graph.png', dpi=150, bbox_inches='tight')


def main():
    """  # Задаем табличные данные (пример)
    x_values = [0, 1, 2, 3, 4]
    y_values = [1, 4, 3, 5, 2]
    x_star = 2.5 """

    x_values = input.x_values
    y_values = input.y_values
    x_star = input.x_star

    print("="*60)
    print("ИНТЕРПОЛЯЦИЯ ТАБЛИЧНОЙ ФУНКЦИИ")
    print("="*60)
    print("\nТаблица данных:")
    print("   x    |    y")
    print("--------|--------")
    for i in range(len(x_values)):
        print(f"  {x_values[i]:3.6f}   |   {y_values[i]:3.6f}")
    print(f"\nТочка для интерполяции: x* = {x_star}")

    # Создаем объект интерполяции
    interp = Interpolation(x_values, y_values)

    # Интерполяция 2-й степени
    lagrange_2, newton_2 = interp.interpolate_at_point(x_star, degree=2)

    # Интерполяция 3-й степени
    lagrange_3, newton_3 = interp.interpolate_at_point(x_star, degree=3)

    # Сравнение результатов
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    print(f"Метод Лагранжа 2-й степени:  {lagrange_2:.6f}")
    print(f"Метод Лагранжа 3-й степени:  {lagrange_3:.6f}")
    print(f"Метод Ньютона 2-й степени:   {newton_2:.6f}")
    print(f"Метод Ньютона 3-й степени:   {newton_3:.6f}")

    # Оценка погрешности через сравнение разных степеней
    print(f"\nОценка погрешности (по разности многочленов 2-й и 3-й степени):")
    error_2_3 = abs(lagrange_3 - lagrange_2)
    print(f"  Δ = |L₃ - L₂| = {error_2_3:.6f}")

    # Проверка согласованности методов
    print(f"\nСогласованность методов Лагранжа и Ньютона:")
    print(f"  Разница для 2-й степени:  {abs(lagrange_2 - newton_2):.10f}")
    print(f"  Разница для 3-й степени:  {abs(lagrange_3 - newton_3):.10f}")

    interp.plot_interpolation(x_star, degrees=[2, 3])

if __name__ == "__main__":
    main()
