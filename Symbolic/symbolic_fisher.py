import symbolic_evolution
from sympy import symbols, exp, I, sqrt, Matrix, expand_complex, simplify, init_printing, Sum
from sympy import summation, zeros
from symbolic_evolution import zero_state, evolve_state, expectation_value


def gaussian_fisher_matrix(exp, times):
    # Calculate the Fisher information matrix for a Gaussian distribution
    # exp: Expectation value functions
    # times: Time points
    # Returns: Fisher information matrix

    symbs = list(exp.free_symbols)
    # symbs = sorted(symbs, key=symbolic_evolution.sort_key)
    print(symbs)
    t = symbols('t', real=True)
    n = len(symbs) - 1
    matrix = zeros(n, n)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = Sum(exp.diff(symbs[i]) * exp.diff(symbs[j]), (t, 0, times))

    return matrix




