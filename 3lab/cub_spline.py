import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import input

class CubicSpline:
    def __init__(self, x_values, y_values):
        """
        Инициализация и построение естественного кубического сплайна
        x_values: массив узлов (неравномерная сетка)
        y_values: массив значений функции в узлах
        """
        self.x = np.array(x_values, dtype=float)
        self.y = np.array(y_values, dtype=float)
        self.n = len(x_values)
        
        # Коэффициенты сплайнов на каждом отрезке
        self.a = np.zeros(self.n - 1)  # свободный член
        self.b = np.zeros(self.n - 1)  # коэффициент при (x-x_i)
        self.c = np.zeros(self.n - 1)  # коэффициент при (x-x_i)^2
        self.d = np.zeros(self.n - 1)  # коэффициент при (x-x_i)^3
        
        # Построение сплайна
        self._build_spline()
    
    def _build_spline(self):
        """
        Построение естественного кубического сплайна
        """
        n = self.n
        h = np.zeros(n - 1)
        
        # Шаг 1: вычисляем шаги сетки
        for i in range(n - 1):
            h[i] = self.x[i + 1] - self.x[i]
        
        # Шаг 2: формируем систему для вторых производных M_i
        # M_i = S''(x_i)
        # Для естественного сплайна: M_0 = 0, M_{n-1} = 0
        
        # Создаем матрицу системы (трехдиагональная)
        # Размер n-2 (внутренние узлы)
        A = np.zeros((n - 2, n - 2))
        B = np.zeros(n - 2)
        
        for i in range(1, n - 1):
            # Индекс в матрице для внутренних узлов
            idx = i - 1
            
            # Диагональные элементы
            if i > 1:
                A[idx, idx - 1] = h[i - 1] / 6
            
            A[idx, idx] = (h[i - 1] + h[i]) / 3
            
            if i < n - 2:
                A[idx, idx + 1] = h[i] / 6
            
            # Правая часть
            B[idx] = (self.y[i + 1] - self.y[i]) / h[i] - (self.y[i] - self.y[i - 1]) / h[i - 1]
        
        # Шаг 3: решаем систему для M_1 ... M_{n-2}
        M = np.zeros(n)  # M[0] = 0, M[n-1] = 0
        if n > 2:
            M[1:n-1] = np.linalg.solve(A, B)
        
        # Шаг 4: вычисляем коэффициенты сплайнов на каждом отрезке
        for i in range(n - 1):
            self.a[i] = self.y[i]
            
            self.b[i] = (self.y[i + 1] - self.y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
            
            self.c[i] = M[i] / 2
            
            self.d[i] = (M[i + 1] - M[i]) / (6 * h[i])
        
        # Сохраняем вторые производные и шаги
        self.M = M
        self.h = h
    
    def evaluate(self, x_star):
        """
        Вычисляет значение сплайна в точке x_star
        """
        # Находим отрезок, содержащий x_star
        if x_star < self.x[0] or x_star > self.x[-1]:
            raise ValueError(f"Точка {x_star} вне интервала интерполяции [{self.x[0]}, {self.x[-1]}]")
        
        # Находим индекс i, такой что x_i <= x_star <= x_{i+1}
        i = 0
        while i < len(self.x) - 2 and x_star > self.x[i + 1]:
            i += 1
        
        # Вычисляем значение сплайна на отрезке i
        dx = x_star - self.x[i]
        result = (self.a[i] + 
                  self.b[i] * dx + 
                  self.c[i] * dx**2 + 
                  self.d[i] * dx**3)
        
        return result, i
    
    def print_coefficients(self):
        """
        Выводит коэффициенты сплайна на всех отрезках
        """
        print("\n" + "="*80)
        print("КОЭФФИЦИЕНТЫ КУБИЧЕСКОГО СПЛАЙНА")
        print("="*80)
        print("Форма сплайна на отрезке [x_i, x_{i+1}]:")
        print("S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3")
        print("-"*80)
        
        for i in range(len(self.a)):
            print(f"\nОтрезок {i}: [{self.x[i]:.4f}, {self.x[i+1]:.4f}]")
            print(f"  a_{i} = {self.a[i]:.8f}")
            print(f"  b_{i} = {self.b[i]:.8f}")
            print(f"  c_{i} = {self.c[i]:.8f}")
            print(f"  d_{i} = {self.d[i]:.8f}")
            
            # Проверка на концах отрезка
            S_left = self.evaluate(self.x[i])[0]
            S_right = self.evaluate(self.x[i+1])[0]
            print(f"  Проверка: S({self.x[i]}) = {S_left:.8f} (должно быть {self.y[i]:.8f})")
            print(f"           S({self.x[i+1]}) = {S_right:.8f} (должно быть {self.y[i+1]:.8f})")
    
    def print_second_derivatives(self):
        """
        Выводит вторые производные в узлах
        """
        print("\n" + "="*80)
        print("ВТОРЫЕ ПРОИЗВОДНЫЕ В УЗЛАХ (M_i = S''(x_i))")
        print("="*80)
        for i in range(self.n):
            print(f"  M_{i} = S''({self.x[i]}) = {self.M[i]:.8f}")
    
    def estimate_error(self, x_star):
        """
        Оценка погрешности сплайна
        """
        # Находим отрезок, содержащий x_star
        i = 0
        while i < len(self.x) - 2 and x_star > self.x[i + 1]:
            i += 1
        
        # Оценка погрешности через остаточный член
        # |R(x)| ≤ (h_i^4 / 384) * max|f^(4)(ξ)|
        h_i = self.h[i]
        
        # Приближенная оценка максимальной четвертой производной
        # через разности третьих производных на соседних отрезках
        if i > 0 and i < len(self.h) - 1:
            # Третьи производные на соседних отрезках: 6*d_i
            d3_left = 6 * self.d[i-1]
            d3_right = 6 * self.d[i]
            d3_next = 6 * self.d[i+1] if i + 1 < len(self.d) else d3_right
            
            max_d4 = max(abs(d3_right - d3_left), abs(d3_next - d3_right)) / h_i
        else:
            max_d4 = abs(6 * self.d[i]) / h_i if i < len(self.d) else 0
        
        # Оценка погрешности
        error_estimate = (h_i**4 / 384) * max_d4 if max_d4 > 0 else 0.001
        
        return error_estimate


    def plot_spline(self, x_star=None, save_path="Spline.png"):
        """
        Строит график кубического сплайна и исходных точек
        
        x_star: точка для отображения (опционально)
        save_path: путь для сохранения графика (опционально)
        """
        plt.figure(figsize=(12, 8))
        
        # Исходные точки
        plt.scatter(self.x, self.y, color='black', s=100, 
                label='Исходные данные', zorder=5)
        
        # Гладкие точки для построения графика сплайна
        x_smooth = np.linspace(min(self.x), max(self.x), 500)
        y_smooth = []
        
        for xi in x_smooth:
            val, _ = self.evaluate(xi)
            y_smooth.append(val)
        
        plt.plot(x_smooth, y_smooth, color='red', linewidth=2, 
                label='Кубический сплайн (естественный)')
        
        # Отмечаем точку x*, если задана
        if x_star is not None:
            plt.axvline(x=x_star, color='purple', linestyle='--', alpha=0.7,
                    label=f'x* = {x_star}')
            value, segment = self.evaluate(x_star)
            plt.plot(x_star, value, 'o', color='green', markersize=10, zorder=10)
            plt.annotate(f'S({x_star})={value:.4f}',
                        xy=(x_star, value), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, color='green')
        
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('Естественный кубический сплайн дефекта 1', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сохранен как {save_path}")
        


def main():
    # Табличные данные (пример)
    """ x_values = [0, 1, 2, 3, 4]
    y_values = [1, 4, 3, 5, 2]
    x_star = 2.5 """

    x_values = input.x_values_spline
    y_values = input.y_values_spline
    x_star = input.x_star_spline
    
    print("="*80)
    print("ЕСТЕСТВЕННЫЙ КУБИЧЕСКИЙ СПЛАЙН ДЕФЕКТА 1")
    print("="*80)
    print("\nТаблица данных:")
    print("   x    |    y")
    print("--------|--------")
    for i in range(len(x_values)):
        print(f"  {x_values[i]:3.6f}   |   {y_values[i]:3.6f}")
    print(f"\nТочка для интерполяции: x* = {x_star}")
    
    # Создаем сплайн
    spline = CubicSpline(x_values, y_values)
    
    # Вычисляем значение в точке x*
    value, segment = spline.evaluate(x_star)
    
    print(f"\n" + "="*80)
    print("РЕЗУЛЬТАТ ИНТЕРПОЛЯЦИИ")
    print("="*80)
    print(f"Точка x* = {x_star}")
    print(f"Находится на отрезке [{spline.x[segment]:.4f}, {spline.x[segment+1]:.4f}]")
    print(f"Значение сплайна: S({x_star}) = {value:.8f}")
    
    # Выводим вторые производные
    spline.print_second_derivatives()
    
    # Выводим коэффициенты сплайна
    spline.print_coefficients()
    
    # Оценка погрешности
    error = spline.estimate_error(x_star)
    print(f"\n" + "="*80)
    print("ОЦЕНКА ПОГРЕШНОСТИ")
    print("="*80)
    print(f"Приближенная оценка погрешности: |R(x*)| ≤ {error:.8f}")

    spline.plot_spline(x_star)


if __name__ == "__main__":
    main()
