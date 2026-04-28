eps = "0.001"
output = 0


# first part

f_input = "log(x**2+1)/log(10)-x**2+2*x+10"
left = "-3"
right = "6"

graphic = '1graph.png'

# second part

""" f1_system = "x**2 + y**2 - 4"  # первое уравнение системы
f2_system = "x - y**2"         # второе уравнение системы """

f1_system = "2*x - 3*cos(2*y - 1)"  # первое уравнение системы
f2_system = "2*y - exp(x + 1) + 3"  # второе уравнение системы
x0_system = 1.5                 # начальное x
y0_system = 1.2                 # начальное y

second_graphic = '2graph.png'