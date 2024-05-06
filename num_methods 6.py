import numpy as np
import matplotlib.pyplot as plt


def f(x, eps = 0.05):
  return ((1/eps) + np.pi**2) * np.cos(np.pi * x)

def q(x, eps = 0.05):
  return 1/eps

def u_x(x, eps = 0.05):
  return (np.cos(np.pi * x)
   + np.exp((x - 1.0)/(eps**0.5))
    + np.exp(-(x + 1.0)/(eps**0.5)))


def getMatrix(h, arr):
    n = len(arr)
    q_values = [q(i) for i in arr]

    a_values = []
    b_values = []
    c_values = []
    a_values.append(0.0)
    b_values.append(0.0)
    c_values.append(1.0)
    for i in range(1, n - 1):
        a_values.append(-1 / h ** 2)
        b_values.append(-1 / h ** 2)
        c_values.append((2 / h ** 2) + q_values[i])
    a_values.append(0.0)
    c_values.append(1.0)
    b_values.append(0.0)
    matrix = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            matrix[i][i + 1] = b_values[i]
            matrix[i][i] = c_values[i]
        elif i == n - 1:
            matrix[i][i - 1] = a_values[i]
            matrix[i][i] = c_values[i]
        else:
            matrix[i][i - 1] = a_values[i]
            matrix[i][i + 1] = b_values[i]
            matrix[i][i] = c_values[i]
    return matrix.copy()

def check_res(matr, x, h, f_values,eps, counter):
    s = 0.0
    A = np.matmul(matr, x)
    for i in range(len(f_values)):
        s += (A[i] - f_values[i]) ** 2
    print(((h ** 2) * s) ** 0.5, counter)
    return ((h ** 2) * s) ** 0.5 < eps


def jacobi(n, b, A, eps):
    h = (2.0)/n
    x = np.zeros(len(b))
    x_pred = np.zeros(len(b))
    x[:] = 1
    x_pred[:] = 1
    x[0] = u_x(-1)
    x[-1] = u_x(1)

    counter = 0
    while not check_res(A, x, h, b, eps, counter):
        counter += 1
        x[0] = (-1 * x_pred[1] * A[0][1] + b[0])/A[0][0]

        for i in range(1, n-1):
            x[i] = (-1* x_pred[i-1] * A[i][i-1] - x_pred[i+1]* A[i][i-1] + b[i])/A[i][i]

        x[n-1] = (-1* x_pred[n-2] * A[n-1][n-2] + b[n-1])/A[n-1][n-1]

        x_pred = x.copy()

    return x

def relax(n, matr, f_values, mu, eps):
    h = 2.0/n
    x = np.zeros(n)
    counter = 0
    while not check_res(matr, x, h, f_values, eps, counter):
        counter+=1
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(matr[i][j] * x_new[j] for j in range(i))
            s2 = sum(matr[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (1-mu)*x[i]+mu*(f_values[i] - s1 - s2) / matr[i][i]
        for i in range(n):
            x[i] = x_new[i]
    return x

def naiskSpusk(A, b, eps):
    x = np.zeros(len(A))
    x_old = x.copy()
    it = 0
    r = 5
    while np.linalg.norm(r) >= eps:
        it += 1
        r = np.dot(A, x_old) - b
        tau = np.dot(r, r) / np.dot(r, np.matmul(A, r))
        x = x_old - tau * r
        x_old = np.copy(x)
    print(it)
    return x

def soprGrad(A, b, eps):
    x = np.zeros(len(A))
    x_old = x.copy()
    it = 0
    r_old = b - np.dot(A, x_old)
    po = np.dot(r_old, r_old)
    p = r_old
    q = np.dot(A, p)
    alpha = po / np.dot(p, q)
    x = x_old + alpha * p
    r = r_old - alpha * q
    while np.linalg.norm(np.dot(A, x) - b) >= eps:
        print(np.linalg.norm(np.dot(A, x) - b))
        it += 1
        po_old = po
        po = np.dot(r, r)
        beta = po / po_old
        p = r + beta * p
        q = np.dot(A, p)
        alpha = po / np.dot(p, q)
        x = x + alpha * p
        r = r - alpha * q
    print(it)
    return x

n = 50
h = 2.0 / n
arr = np.linspace(-1.0, 1.0, n)
A = getMatrix(h, arr.copy())
b = [f(x) for x in arr]
b[0] = u_x(-1.0)
b[n - 1] = u_x(1.0)
b = np.array(b)

ux = u_x(arr)

# y = jacobi(n, b, A, 0.001)
# y = relax(n, A, b, 0.7, 0.001)
# y = naiskSpusk(A, b, 0.001)
y = soprGrad(A, b, 0.001)
errors = abs(ux-y)
print("Максимальная погрешность\n", max(errors))
print("h^2\n", h**2)

fig, ax = plt.subplots()
ax.scatter(arr, y, label='y(x)')
plt.legend()
plt.show()

fig2, ax2 = plt.subplots()
ax2.scatter(arr, ux, label='u(x)')
plt.legend()
plt.show()

fig3, ax3 = plt.subplots()
ax3.scatter(arr, errors, label='errors')
plt.legend()

plt.show()
