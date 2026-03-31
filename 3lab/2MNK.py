import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import input

# Настройка русских шрифтов
rcParams['font.family'] = 'DejaVu Sans'

class LeastSquares:
    def __init__(self, x_values, y_values):
        """
        Инициализация с табличными данными
        """
        self.x = np.array(x_values, dtype=float)
        self.y = np.array(y_values, dtype=float)
        self.m = len(x_values)  # количество точек
        
    def fit_polynomial(self, degree):
        """
        Построение аппроксимирующего многочлена степени degree методом МНК
        Возвращает: коэффициенты многочлена (от a0 до an)
        """
        n = degree + 1  # количество коэффициентов
        
        # Строим матрицу нормальной системы
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                A[i, j] = np.sum(self.x ** (i + j))
            b[i] = np.sum(self.x ** i * self.y)
        
        # Решаем систему
        coeffs = np.linalg.solve(A, b)
        
        return coeffs
    
    def evaluate(self, coeffs, x_star):
        """
        Вычисляет значение многочлена в точке x_star
        coeffs: коэффициенты от a0 до an
        """
        result = 0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x_star ** i)
        return result
    
    def compute_sse(self, coeffs):
        """
        Вычисляет сумму квадратов ошибок (SSE)
        """
        y_pred = np.polyval(coeffs[::-1], self.x)  # polyval ожидает старшие степени сначала
        sse = np.sum((self.y - y_pred) ** 2)
        return sse
    
    def compute_r2(self, coeffs):
        """
        Вычисляет коэффициент детерминации R²
        """
        y_pred = np.polyval(coeffs[::-1], self.x)
        sse = np.sum((self.y - y_pred) ** 2)
        sst = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - sse / sst
        return r2
    
    def print_results(self, degree):
        """
        Выводит результаты аппроксимации для заданной степени
        """
        coeffs = self.fit_polynomial(degree)
        sse = self.compute_sse(coeffs)
        r2 = self.compute_r2(coeffs)
        
        print(f"\n{'='*60}")
        print(f"АППРОКСИМАЦИЯ МНОГОЧЛЕНОМ {degree}-Й СТЕПЕНИ")
        print(f"{'='*60}")
        
        # Вывод коэффициентов
        print(f"\nКоэффициенты многочлена P_{degree}(x) = a0 + a1·x + a2·x² + ... + a{degree}·x^{degree}:")
        for i, coeff in enumerate(coeffs):
            print(f"  a{i} = {coeff:.8f}")
        
        # Вывод в стандартной форме
        poly_str = f"P_{degree}(x) = {coeffs[0]:.6f}"
        for i in range(1, degree + 1):
            sign = "+" if coeffs[i] >= 0 else "-"
            poly_str += f" {sign} {abs(coeffs[i]):.6f}·x^{i}"
        print(f"\n{poly_str}")
        
        print(f"\nСтатистические показатели:")
        print(f"  Сумма квадратов ошибок (SSE) = {sse:.8f}")
        print(f"  Коэффициент детерминации R² = {r2:.8f}")
        
        # Оценка качества
        if r2 > 0.95:
            print(f"  Качество: Отличное (R² > 0.95)")
        elif r2 > 0.8:
            print(f"  Качество: Хорошее (R² > 0.8)")
        elif r2 > 0.6:
            print(f"  Качество: Удовлетворительное (R² > 0.6)")
        else:
            print(f"  Качество: Низкое (R² < 0.6)")
        
        return coeffs, sse, r2
    
    def predict_at_point(self, x_star, degrees=[1, 2, 3]):
        """
        Вычисляет значения всех приближающих многочленов в точке x*
        """
        print(f"\n{'='*60}")
        print(f"ВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ В ТОЧКЕ x* = {x_star}")
        print(f"{'='*60}")
        
        results = {}
        for deg in degrees:
            coeffs = self.fit_polynomial(deg)
            value = self.evaluate(coeffs, x_star)
            results[deg] = value
            print(f"P_{deg}({x_star}) = {value:.8f}")
        
        return results
    
    def plot_all(self, x_star=None, degrees=[1, 2, 3]):
        """
        Строит графики всех аппроксимирующих многочленов и исходных точек
        """
        plt.figure(figsize=(12, 8))
        
        # Исходные точки
        plt.scatter(self.x, self.y, color='black', s=100, label='Исходные данные', zorder=5)
        
        # Гладкие точки для построения графиков
        x_smooth = np.linspace(min(self.x) - 0.5, max(self.x) + 0.5, 500)
        
        # Цвета для разных степеней
        colors = ['red', 'green', 'blue']
        
        for deg, color in zip(degrees, colors[:len(degrees)]):
            coeffs = self.fit_polynomial(deg)
            y_smooth = np.polyval(coeffs[::-1], x_smooth)
            plt.plot(x_smooth, y_smooth, color=color, linewidth=2, 
                    label=f'Многочлен {deg}-й степени (R²={self.compute_r2(coeffs):.4f})')
        
        # Отмечаем точку x*, если задана
        if x_star is not None:
            plt.axvline(x=x_star, color='purple', linestyle='--', alpha=0.7, 
                       label=f'x* = {x_star}')
            
            # Отмечаем значения многочленов в x*
            for deg, color in zip(degrees, colors[:len(degrees)]):
                coeffs = self.fit_polynomial(deg)
                value = self.evaluate(coeffs, x_star)
                plt.plot(x_star, value, 'o', color=color, markersize=10, zorder=10)
                plt.annotate(f'P_{deg}({x_star})={value:.3f}', 
                           xy=(x_star, value), xytext=(5, 5),
                           textcoords='offset points', fontsize=9, color=color)
        
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('Аппроксимация табличных данных методом наименьших квадратов', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig('least_squares_approximation.png', dpi=150, bbox_inches='tight')


def main():
    
    x_values = input.x_values_mnk
    y_values = input.y_values_mnk
    x_star = input.x_star_mnk
    
    
    print("="*70)
    print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ (МНК)")
    print("="*70)
    print("\nТаблица данных:")
    print("   x    |    y")
    print("--------|--------")
    for i in range(len(x_values)):
        print(f"  {x_values[i]:3.6f}   |   {y_values[i]:6.6f}")
    
    # Создаем объект МНК
    ls = LeastSquares(x_values, y_values)
    
    # Аппроксимация многочленами 1, 2, 3 степени
    results = {}
    for deg in [1, 2, 3]:
        coeffs, sse, r2 = ls.print_results(deg)
        results[deg] = {'coeffs': coeffs, 'sse': sse, 'r2': r2}
    
    # Вычисление значений в точке x*
    predictions = ls.predict_at_point(x_star, degrees=[1, 2, 3])
    
    # Сравнение SSE разных степеней
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ КАЧЕСТВА АППРОКСИМАЦИИ")
    print(f"{'='*60}")
    print(f"{'Степень':<10} {'SSE':<15} {'R²':<15}")
    print("-"*40)
    for deg in [1, 2, 3]:
        print(f"{deg:<10} {results[deg]['sse']:<15.8f} {results[deg]['r2']:<15.8f}")
    
    # Построение графиков
    ls.plot_all(x_star=x_star, degrees=[1, 2, 3])
    
    print("\nГрафик сохранен как 'least_squares_approximation.png'")


if __name__ == "__main__":
    main()