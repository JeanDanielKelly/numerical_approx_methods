import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class rungeKutta(object):
    def __init__(self):

        self.exact_solution = {
            "solution_string": None,
            "function": None,
            "y_range": np.array([])
        }

        self.numerical_approximation = {
            "x0": None,
            "y0": None,
            "x_range": None,
            "y_range": None,
            "increment": None,
            "test_function_string": None,
            "test_function": None,
            "approx_function": None,
            "is_diff": None,
            "numerical_error": None
        }

        self.parameters = {
            "R1": np.array([]),
            "R2": np.array([]),
            "R3": np.array([]),
            "R4": np.array([]),
            "h": np.array([])
        }

        self.results = np.array([])

    def load_function(self, test_function_string, exact_solution_string = None):
        self.numerical_approximation["test_function_string"] = test_function_string

        if exact_solution_string:
            self.exact_solution["solution_string"] = exact_solution_string
            x = sp.Symbol('x')
            f_eval = eval(self.exact_solution["solution_string"])
            self.exact_solution["function"] = sp.lambdify(x, f_eval) # TBD if exact function can be also f(x,y)