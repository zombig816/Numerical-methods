import numpy as np
import matplotlib.pyplot as plt
import math

# ##################################################### 1 ##################################################### #

# функция нахождения F(x) на отрезке с точностью eps
def f(x, eps):
    # вычисляем значения функции у при различных x
    y = []
    n = 1
    a = 0
    for xi in x:
        # в переменной а будем хранить всю сумму
        a = 0
        # в переменной b - отдельный элемент ряда
        b = (-1 * (xi ** 4)) / (2 * 4)
        n = 1
        # вычисляем пока b больше эпсилон по модулю
        while abs(b) >= eps:
            a += b
            b *= ((-1 * xi ** 6 * (3 * n - 1)) / (12 * n ** 3 + 26 * n ** 2 + 18 * n + 4))
            n += 1
        y.append(a)
    return y

# функция нахождения F`(x) на отрезке с точностью eps
def f_differencial(x, eps):
    i = 0
    y1 = []
    # считаем значения производной функции при различных х
    for xi in x:
        a1 = 0
        b1 = (-1 * xi ** 3) / 2
        while abs(b1) >= eps:
            a1 += b1
            b1 = b1 * (-1 * xi ** 6 * (1 / ((2 * n + 1) * (2 * n + 2))))
            n = n + 1
        n = 1
        y1.append(a1)
        i = i + 1
    return y1

# ##################################################### 2 ##################################################### #

# функция интерполяции по Лагранжу
def lagranzh(x, y, z):
    lagranzh = []
    mult = 1
    s = 0
    for zi in z:
        s = 0
        for i in range(0, len(x)):
            mult = 1
            for j in range(0, len(y)):
                if i != j:
                    mult *= (zi - x[j]) / (x[i] - x[j])
            s += y[i] * mult
        lagranzh.append(s)
    return lagranzh

# функция интерполяции по Ньютону
def newton(x, y1, z):
    newton = []
    y = []
    for yi in y1:
        y.append(yi)

    for k in range(1, len(x)+1):
        for i in range(len(x)-1, k-1, -1):
            y[i] = (y[i] - y[i-1])/(x[i] - x[i-k])

    for xi in z:
        p = y[len(x)-1]
        for i in range(len(x) - 1, -1, -1):
            p = y[i] + (xi - x[i]) * p
        newton.append(p)
    return newton

# функция вычисления чебышевских узлов интерполяции
def chebysh(x0, xn, n):
    x = [x0]
    for i in range(1, n - 1):
        x.append(((x0 + xn)/2) - ((xn - x0)/2)*math.cos((2*i-1)*math.pi/(2*n)))
    x.append(xn)
    return x

# функция нахождения погрешностей между функцией и интерполированной функцией
def error(L, z, eps):
    f1 = f(z, eps)
    error_array = []
    for i in range(len(z)):
        error_array.append(abs(f1[i] - L[i]))
    return error_array

# ##################################################### 3 ##################################################### #

# функция вычисления фи(t) в конкретной точке
def fi_from_t(t):
    if(t != 0):
        return (math.cos(t*t*t) - 1) / (t*t*t)
    else:
        return -(t*t*t)/2

# вычисление интеграла по составной квадратурной формуле Гаусса
def gauss_si(x1, i, hn):
    zi_1 = x1 + (i - 1) * hn
    t1 = zi_1 + (hn / 2) * (1 - (1 / (math.sqrt(3))))
    t2 = zi_1 + (hn / 2) * (1 + (1 / (math.sqrt(3))))

    si = (hn / 2) * (fi_from_t(t1) + fi_from_t(t2))
    return si

# вычисление интеграла по составной квадратурной формуле правых прямоугольников
def right_rect_si(x1, i, hn):
    b = x1 + i * hn
    si = hn * fi_from_t(b)
    return si

# вычисление интеграла по составной квадратурной формуле центральных прямоугольников
def center_rect_si(x1, i, hn):
    b = x1 + i * hn
    a = b - hn
    si = (b - a) * fi_from_t((b + a) / 2)
    return si

# вычисление интеграла по составной квадратурной формуле трапеции
def trapezoid_si(x1, i, hn):
    b = x1 + i * hn
    a = b - hn
    si = (b - a) / 2 * (fi_from_t(a) + fi_from_t(b))
    return si

# вычисление интеграла по составной квадратурной формуле Симпсона
def simpson_si(x1, i, hn):
    b = x1 + i * hn
    a = b - hn
    c = (a + b) / 2
    si = (b - a) / 6 * (fi_from_t(a) + 4 * fi_from_t(c) + fi_from_t(b))
    return si

# функция для вычисления интеграла (f_si - формула, по которой хотим вычислить)
def square(f_si, x, y, eps):
    sn = 0
    s2n = 0
    n = 2

    for j in range(1, len(x)):
        n = 2
        while s2n == 0 or abs(s2n - sn) > eps:
            hn = x[j] / n
            h2n = x[j] / (2*n)
            s = 0
            for i in range(1, n + 1):
                s += f_si(0, i, hn)
            sn = s

            s = 0
            for i in range(1, 2*n + 1):
                s += f_si(0, i, h2n)
            s2n = s

            n *= 2
        print(x[j], "\t", np.round(y[j], 8), "\t", np.round(sn, 8), "\t", n-1, "\t", np.round(y[j]-sn, 8))
        s2n = 0
    return sn

# ##################################################### 1 ##################################################### #
# шаг
step = 0.08
# точность
eps = 0.0000001
# левая граница
a = 0
# правая граница
b = 2
# количество точек
n = int((b - a) / step) + 1
# массив точек x
x = np.linspace(a, b, n)
# массив точек y
y = f(x, eps)
# выводим точки x и y
for i in range(len(x)):
    print("точка x", i, " =    ", x[i], "       точка у", i, " =    ", y[i])
# массив точек y`
y1 = f_differencial(x, eps)

# вывод графика y = F(x)
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 1)
plt.title('y = F(x)')
plt.xlim(-1, 2.1)
plt.ylim(-2, 1)
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax.scatter(x, y, marker='.')

# вывод графика y = F`(x)
fig1, ax1 = plt.subplots(1, 1)
plt.title('y = F`(x)')
plt.xlim(-1, 2.1)
plt.ylim(-2, 1)
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax1.scatter(x, y1, marker='.')

# ##################################################### 2 ##################################################### #

# строим график функции после интерполяции по Лагранжу
fig2, ax2 = plt.subplots(1, 1)
plt.title("Интерполирование по Лагранжу")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
plt.xlim(-1, 2.1)
plt.ylim(-2, 1)
n = 26
z = np.linspace(0, 2, n)
L = lagranzh(x, y, z)
ax2.scatter(z, L, marker='.')

# вычисляем чебышевские узлы
x_cheb = chebysh(a, b, 30)
y_cheb = f(x_cheb, eps)

# график ошибок по Лагранжу
n = 50

n_array = []
for i in range(n):
    n_array.append(i+1)
max_errors = []

# вычисляем массив погрешностей при разном количестве узлов( от 1 до n )
for ni in n_array:
    z = np.linspace(a, b, ni)
    L = lagranzh(x, y, z)
    max_errors.append(max(error(L, z, eps)))

# строим график ошибок
fig3, ax3 = plt.subplots(1, 1)
plt.title("График ошибок при интерполяции по Лагранжу на равномерной сетке узлов")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax3.scatter(n_array, max_errors, marker='.')

print("\nЛагранж")
print("x\tF(x)\tL(x)\terror")
for i in range(len(z)):
    print(np.round(z[i],8), "\t", np.round(f(z, eps)[i],8), "\t", np.round(L[i],8), "\t", np.round(max_errors[i],8))

# то же самое по Лагранжу для чебышевских узлов
max_ch_l_errors = []

for ni in n_array:
    z = np.linspace(a, b, ni)
    L = lagranzh(x_cheb, y_cheb, z)
    max_ch_l_errors.append(max(error(L, z, eps)))

fig_c_l, ax_c_l = plt.subplots(1, 1)
plt.title("График ошибок при интерполяции по Лагранжу на чебышевской сетке узлов")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax_c_l.scatter(n_array, max_ch_l_errors, marker='.')

print("\nЛагранж чебышевские узлы")
print("x\tF(x)\tL(x)\terror")
for i in range(len(z)):
    print(np.round(z[i],8), "\t", np.round(f(z, eps)[i],8), "\t", np.round(L[i],8), "\t", np.round(max_ch_l_errors[i],8))

# график ошибок по Ньютону
n1 = 50

# вычисляем точки при интерполяции по Ньютону
z1 = np.linspace(0, 2, n1)
newton_Y = newton(x, y, z1)

# строим график функции после интерполяции по Ньютону
fig_newton, ax_newton = plt.subplots(1, 1)
plt.title("Интерполирование по Ньютону")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
plt.xlim(-1, 2.1)
plt.ylim(-2, 1)
ax_newton.scatter(z1, newton_Y, marker='.')

n1_array = []
for i in range(n1):
    n1_array.append(i+1)
max1_errors = []

for ni in n1_array:
    z1 = np.linspace(a, b, ni)
    N = newton(x, y, z1)
    max1_errors.append(max(error(N, z1, eps)))

print("\nНьютон")
print("x\tF(x)\tN(x)\terror")
for i in range(len(z)):
    print(np.round(z1[i],8), "\t", np.round(f(z1, eps)[i],8), "\t", np.round(newton_Y[i],8), "\t", np.round(max1_errors[i],8))


fig4, ax4 = plt.subplots(1, 1)
plt.title("График ошибок при интерполировании по Ньютону на равномерной сетке узлов")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax4.scatter(n1_array, max1_errors, marker='.')

# Ньютон чебышевские узлы
max_ch_n_errors = []

for ni in n1_array:
    z1 = np.linspace(a, b, ni)
    N = newton(x_cheb, y_cheb, z1)
    max_ch_n_errors.append(max(error(N, z1, eps)))


print("\nНьютон чебышевские узлы")
print("x\tF(x)\tN(x)\terror")
for i in range(len(z)):
    print(np.round(z1[i],8), "\t", np.round(f(z1, eps)[i],8), "\t", np.round(N[i],8), "\t", np.round(max_ch_n_errors[i],8))

fig_ch_n, ax_ch_n = plt.subplots(1, 1)
plt.title("График ошибок при интерполировании по Ньютону на чебышевской сетке узлов")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax_ch_n.scatter(n1_array, max_ch_n_errors, marker='.')

# ##################################################### 3 ##################################################### #

print("\nright rectangles")
print("x\ty\ts\tn\terror")
square(right_rect_si, x, y, 0.00001)

print("\ncentral rectangles")
print("x\ty\ts\tn\terror")
square(center_rect_si, x, y, 0.00001)

print("\ntrapezoid")
print("x\ty\ts\tn\terror")
square(trapezoid_si, x, y, 0.00001)

print("\nsimpson")
print("x\ty\ts\tn\terror")
square(simpson_si, x, y, 0.00001)

print("\ngauss")
print("x\ty\ts\tn\terror")
square(gauss_si, x, y, 0.00001)

plt.show()