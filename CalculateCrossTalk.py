import numpy as np
import sympy
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize

h = lambda n, J, z: sum([J[i] * (z[i] - 1) * (z[i + 1] - 1) for i in range(n - 1)])


def crosstalk_operator(size, t):
    J = [sp.Symbol(f'J{i + 1}') for i in range(size - 1)]
    hem = np.zeros((2 ** size, 2 ** size), dtype=object)
    for i in range(2 ** size):
        binary = '{0:b}'.format(i).zfill(size)
        Z = [(-1) ** int(i) for i in binary]
        hem[i, i] = sp.exp(-h(size, J, Z) * 1j * t)
    return hem


def state(size, i):
    a = np.zeros((1, 2 ** size))
    a[0][i] = a[0][i] + 1
    return a


# A function that takes in time and qiskit counts data and returns the cross talk matrix
def one_dim(t, results, n):
    shots = sum(results.values())
    h1 = (1 / (np.sqrt(2))) * np.matrix([[1, 1], [1, -1]])
    H = np.ones((1, 1))
    for i in range(n):
        H = np.kron(H, h1)
    equations = []
    for i in range(2 ** n):
        op = np.matmul(H, crosstalk_operator(n, t))
        op = np.matmul(op, H)

        eq = (np.abs(np.dot(np.dot(state(n, i), op), state(n, 0).T))**2 * shots - results.get(f"{i:0{n}b}", 0))**2
        #eq = sp.sympify(eq)
        equations.append(eq)
    return solve(equations,n)



def solve(equations, size):

    J = [sp.Symbol(f'J{i + 1}') for i in range(size - 1)]
    expr = sp.sympify(sum(equations))
    #expr = sp.sympify(equations[0]+equations[1])
    f = sp.lambdify(J, expr, 'numpy')

    def print_progress(xk, state):
        print(f"Iteration: {state['nit']}, Function Value: {state['fun']}, Variables: {xk}")

    def wrapper_func(x):
        try:
            return f(*x)
        except ValueError:
            return np.inf

    #result = minimize(wrapper_func, x0=[1] * (size-1), bounds=(0,2), method='Powell')
    result = minimize(wrapper_func, x0=[1] * (size-1))
    print(result.fun)
    return result.x

