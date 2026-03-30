import numpy as np
import input

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
        """
        result = 0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x_star ** i)
        return result
    
    def compute_sse(self, coeffs):
        """
        Вычисляет сумму квадратов ошибок (SSE)
        """
        # Вычисляем предсказанные значения
        y_pred = np.zeros_like(self.y)
        for i, x_val in enumerate(self.x):
            y_pred[i] = self.evaluate(coeffs, x_val)
        
        sse = np.sum((self.y - y_pred) ** 2)
        return sse
    
    def compute_r2(self, coeffs):
        """
        Вычисляет коэффициент детерминации R²
        """
        y_pred = np.zeros_like(self.y)
        for i, x_val in enumerate(self.x):
            y_pred[i] = self.evaluate(coeffs, x_val)
            
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
            if coeffs[i] >= 0:
                poly_str += f" + {coeffs[i]:.6f}·x^{i}"
            else:
                poly_str += f" - {abs(coeffs[i]):.6f}·x^{i}"
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
    print(f"\nТочка для вычисления: x* = {x_star}")
    
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
    
    # Вывод лучшей модели
    best_deg = min(results.keys(), key=lambda d: results[d]['sse'])
    print(f"\nЛучшая модель (минимальная SSE): многочлен {best_deg}-й степени")
    print(f"  SSE = {results[best_deg]['sse']:.8f}")
    print(f"  R² = {results[best_deg]['r2']:.8f}")


if __name__ == "__main__":
    main()