import random

import numpy as np
from scipy.optimize import curve_fit
from sympy import symbols, lambdify

from Symbolic import symbolic_evolution


def complex_fit(batch_x, batch_y):
    def model_func(t, a, w):
        x_model = (np.cos(w * t)) * np.exp(-a * t)
        y_model = -(np.sin(w * t)) * np.exp(-a * t)
        return np.concatenate([x_model, y_model])

    data_x = []
    data_y = []

    for i in range(batch_x.n):
        data_x.append(batch_x.get_zi(i))
        data_y.append(batch_y.get_zi(i))

    parameters = []
    for i in range(len(data_x)):
        initial_guess = [random.random(), random.random()]
        # Perform the curve fitting
        t_points = batch_x.delay
        z_points = np.concatenate([np.array(data_x[i]), np.array(data_y[i])])
        try:
            params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess)
        except:
            params = [100, 100]
        # self.decay_fit.append(params[0])
        # self.W_fit.append(params[1])
        parameters.append(params)
    return parameters


def fit_X(batch_x):
    def model_func(t, a, w):
        x_model = (np.cos(w * t)) * np.exp(-a * t)
        return x_model

    data_x = []

    for i in range(batch_x.n):
        data_x.append(batch_x.get_zi(i))


    parameters = []
    for i in range(len(data_x)):
        initial_guess = [random.random(), random.random()]
        # Perform the curve fitting
        t_points = batch_x.delay
        z_points = np.concatenate([np.array(data_x[i])])
        try:
            bounds_lower = [0] * 2
            bounds_upper = [2 * np.pi] * 2
            bounds = (bounds_lower, bounds_upper)
            params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess, bounds=bounds)
        except:
            params = [100, 100]
        parameters.append(np.abs(params))
    return parameters


def one_by_one_fit(batch_x_detuning, batch_y_detuning, batch_x_crosstalk, batch_y_crosstalk):
    n = batch_x_detuning.n
    params = complex_fit(batch_x_detuning, batch_y_detuning)
    W = [params[i][1] for i in range(n)]
    decay = [params[i][0] for i in range(n)]

    crosstalk_qubits_measured = batch_x_crosstalk.qubits_measured
    params = complex_fit(batch_x_crosstalk, batch_y_crosstalk)
    J = [params[i][1] - W[crosstalk_qubits_measured[i]] for i in range(n - 1)]
    return decay, W, J


def one_by_one_X(batch_x_detuning, batch_x_crosstalk):
    n = batch_x_detuning.n
    params = fit_X(batch_x_detuning)
    W = [params[i][1] for i in range(n)]
    decay = [params[i][0] for i in range(n)]

    crosstalk_qubits_measured = batch_x_crosstalk.qubits_measured
    params = fit_X(batch_x_crosstalk)
    J = [params[i][1] - W[crosstalk_qubits_measured[i]] for i in range(n - 1)]
    return decay, W, J


def full_complex_fit(batch_x, batch_y, neighbors=0):
    n = batch_x.n

    data = []
    for i in range(len(batch_x.RamseyExperiments)):
        data.append(batch_x.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))
        data.append(batch_y.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))

    # symbolic_exp = symbolic_evolution.minimize_functions(n, times, neighbors=neighbors)

    symbolic_exp = symbolic_evolution.get_expectation_values_exp(n, neighbors=neighbors)
    t = symbols('t', real=True)
    w = symbols(f'ω0:{n}', real=True)
    j = symbols(f'j0:{n - 1}', real=True)
    a = symbols(f'a0:{n}', real=True)

    symbolic_exp = [lambdify([t, *w, *a, *j], expr, 'numpy') for expr in symbolic_exp]

    def model_func(t, *params):
        n = batch_x.n
        A = params[:n]
        W = params[n:2 * n]
        J = params[2 * n:2 * n + n - 1]
        functions = np.array([expr(t, *W, *A, *J) for expr in symbolic_exp])
        functions = functions.T
        return np.concatenate(functions)
        # functions = np.array([symbolic_evolution.set_parameters(expr, W, J, A) for expr in symbolic_exp])
        # return functions

    # initial_guess = [1] * (2 * batch_x.n + (batch_x.n - 1))  # Adjusted to include J
    initial_guess = [random.random() for i in range(2 * batch_x.n + (batch_x.n - 1))]
    bounds_lower = [-2 * np.pi] * (2 * batch_x.n + (batch_x.n - 1))
    bounds_upper = [2 * np.pi] * (2 * batch_x.n + (batch_x.n - 1))
    bounds = (bounds_lower, bounds_upper)

    # Perform the curve fitting
    t_points = batch_x.delay
    z_points = np.concatenate(data)
    params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess, bounds=bounds)
    guessed_decay = params[:batch_x.n][::-1]
    guessed_W = params[batch_x.n:2 * batch_x.n][::-1]
    guessed_J = params[2 * batch_x.n:3 * batch_x.n - 1][::-1]
    return guessed_decay, guessed_W, guessed_J


def percent_error(correct, fitted):
    mse = np.mean((correct - fitted) ** 2)
    return np.sqrt(mse) / np.mean(np.abs(correct)) * 100


def calc_dist(fitted_values, correct_values):
    fitted_values = np.array(fitted_values)
    correct_values = np.array(correct_values)
    mse = (fitted_values - correct_values) ** 2 / len(fitted_values)
    precent_error = (np.sqrt(np.abs(mse)) / np.abs(correct_values)) * 100
    return precent_error
