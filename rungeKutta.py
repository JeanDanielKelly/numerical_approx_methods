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
            "x_range": np.array([]),
            "y_range": np.array([]),
            "increment": None,
            "test_function_string": None,
            "test_function": None,
            "approx_function": None,
            "step_type": None,
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
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        f_eval = eval(self.numerical_approximation["test_function_string"])
        self.numerical_approximation["test_function"] = sp.lambdify((x,y), f_eval)

        if exact_solution_string:
            self.exact_solution["solution_string"] = exact_solution_string
            f_eval = eval(self.exact_solution["solution_string"])
            self.exact_solution["function"] = sp.lambdify(x, f_eval) # TBD if exact function can be also f(x,y)

    def set_parameters(self, x0, y0, step_type):
        self.numerical_approximation["step_type"] = step_type # "single" if Single step - "multiple" if Multiple steps
        self.numerical_approximation["x0"] = x0
        self.numerical_approximation["y0"] = y0

    def eval_function(self, min_range, max_range, increment):
        self.numerical_approximation["min_range"] = min_range
        self.numerical_approximation["max_range"] = max_range
        self.numerical_approximation["increment"] = increment
        self.numerical_approximation["x_range"] = np.linspace(min_range, max_range, num=increment) # assuming that x0 is at min range
        x_range = self.numerical_approximation["x_range"]
        x0 = self.numerical_approximation["x0"]
        y0 = self.numerical_approximation["y0"]
        test_function = self.numerical_approximation["test_function"]

        if self.numerical_approximation["step_type"] == "single":
            for xn in x_range:
                h = xn - x0
                R1 = test_function(x0, y0) * h
                R2_x = x0 + 0.5 * h
                R2_y = y0 + 0.5 * R1
                R2 = test_function(R2_x, R2_y) * h
                R3_x = x0 + 0.5 * h
                R3_y = y0 + 0.5 * R2
                R3 = test_function(R3_x, R3_y) * h
                R4_x = x0 + h
                R4_y = y0 + R3
                R4 = test_function(R4_x, R4_y) * h
                result = y0 + 1/6 * (R1 + 2 * R2 + 2 * R3 + R4)
                self.parameters["R1"] = np.append(self.parameters["R1"], R1)
                self.parameters["R2"] = np.append(self.parameters["R2"], R2)
                self.parameters["R3"] = np.append(self.parameters["R3"], R3)
                self.parameters["R4"] = np.append(self.parameters["R4"], R4)
                self.parameters["h"] = np.append(self.parameters["h"], h)
                self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], result)
            
                if self.exact_solution["solution_string"] is not None:
                    self.exact_solution["y_range"] = np.append(self.exact_solution["y_range"], self.exact_solution["function"](xn))

        elif self.numerical_approximation["step_type"] == "multiple":

            self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], self.numerical_approximation["y0"])
            self.exact_solution["y_range"] = np.append(self.exact_solution["y_range"], self.numerical_approximation["y0"])

            for i in range(len(x_range) - 1):
                x0 = x_range[i]
                y0 = self.numerical_approximation["y_range"][i]
                xn = x_range[i + 1]

                # copy pasta of single step, this could be integrated in a defined function
                h = xn - x0
                R1 = test_function(x0, y0) * h
                R2_x = x0 + 0.5 * h
                R2_y = y0 + 0.5 * R1
                R2 = test_function(R2_x, R2_y) * h
                R3_x = x0 + 0.5 * h
                R3_y = y0 + 0.5 * R2
                R3 = test_function(R3_x, R3_y) * h
                R4_x = x0 + h
                R4_y = y0 + R3
                R4 = test_function(R4_x, R4_y) * h
                result = y0 + 1/6 * (R1 + 2 * R2 + 2 * R3 + R4)
                self.parameters["R1"] = np.append(self.parameters["R1"], R1)
                self.parameters["R2"] = np.append(self.parameters["R2"], R2)
                self.parameters["R3"] = np.append(self.parameters["R3"], R3)
                self.parameters["R4"] = np.append(self.parameters["R4"], R4)
                self.parameters["h"] = np.append(self.parameters["h"], h)
                self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], result)

                if self.exact_solution["solution_string"] is not None:
                    self.exact_solution["y_range"] = np.append(self.exact_solution["y_range"], self.exact_solution["function"](xn))

        plt.plot(x_range, self.numerical_approximation["y_range"], color='orange', label='Runge-Kutta')
        if self.exact_solution["solution_string"] is not None:
            plt.plot(x_range, self.exact_solution["y_range"], color='blue', label='Exact', linestyle='dashed')
        plt.legend()
        plt.title('Runge-Kutta approx vs exact solution')
        plt.show()

    def plot_error(self):
        self.numerical_approximation["numerical_error"] = self.exact_solution["y_range"] - self.numerical_approximation["y_range"]

        plt.plot(self.numerical_approximation["x_range"], self.numerical_approximation["numerical_error"], color='red', label='error')
        plt.legend()
        plt.title('Numerical error of Runge Kutta approximation')
        plt.show()
    
        
    def eval_value(self, value):
        return np.interp(value, self.numerical_approximation["x_range"], self.numerical_approximation["y_range"])