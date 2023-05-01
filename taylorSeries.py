import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class taylorSeries(object):
    def __init__(self):

        self.exact_solution = {
            "solution_string": None,
            "function": None,
            "y_range": np.array([])
        }

        self.numerical_approximation = {
            "x0": None, # boundary condition
            "x_range": None,
            "y0": None, # boundary condition
            "y_range": np.array([]),
            "min_range": None,
            "max_range": None,
            "increment": None,
            "test_function_string":None,
            "test_function": None,
            "is_diff": None, # True if differential equation dy/dx = f(x,y) | False if equation y(x) = ax + b
            "step_type": None, # "multiple" if multiple steps | "single" if evaluated in 1 point
            "approx_function": None,
            "numerical_error": None
        }


    def load_function(self, test_function_string, exact_solution_string = None):
        self.numerical_approximation["test_function_string"] = test_function_string

        if exact_solution_string:
            self.exact_solution["solution_string"] = exact_solution_string
            x = sp.Symbol('x')
            f_eval = eval(self.exact_solution["solution_string"])
            self.exact_solution["function"] = sp.lambdify(x, f_eval) # TBD if exact function can be also f(x,y)


    def set_parameters(self, x0, y0, step_type, is_diff):
        self.numerical_approximation["is_diff"] = is_diff # Input: True if differential eq - False if not
        self.numerical_approximation["step_type"] = step_type # "single" if Single step - "multiple" if Multiple steps
        self.numerical_approximation["x0"] = x0
        self.numerical_approximation["y0"] = y0
        #self.numerical_approximation["x0"] = np.append(self.numerical_approximation["x0"], x0) # boundary condition (bc)
        #self.numerical_approximation["y0"] = np.append(self.numerical_approximation["y0"], y0) # bc


    def series_approximation(self, test_function_string, x0, y0):

        # IMPORTANT - INPUT MAY BE DIFFERENTIAL EQUATION
        # dy/dx = f(x,y)
        # y(x) = y(x0) + y'(x0)(x-x0) + y"(x0)/2! * (x-x0)**2 + ... + y'N(x0)/N! * (x-x0)**N

        # y'(x0) = f(x0,y0) *** IF INPUT IS DIFFERENTIAL EQUATION
        # y''(x0) = [df/dx + df/dy f(x,y)] | x=x0 & y=y(x0)
        # y'''(x0) = {d/dx [df/dx + df/dy f(x,y)]} | x=x0 & y=y0 + {f(x,y) d/dy[df/dx + df/dy y'(x)]} | x=x0 & y=y0

        x = sp.Symbol('x')
        y = sp.Symbol('y')

        if self.numerical_approximation["is_diff"]:
            #print('Approximation of dy/dx = ' + test_function_string)
            #print('BC x(0) = ' + str(x0))
            #print('BC y(0) = ' + str(y0))

            f_prime = eval(test_function_string)
            y_prime = sp.lambdify((x,y), f_prime)
            #print('f_prime expanded = ' + str(f_prime))
            #print('f_prime(0) = ' + str(y_prime(x0,y0)))
            f_2prime = sp.diff(f_prime, x) + sp.diff(f_prime, y) * f_prime
            y_2prime = sp.lambdify((x,y), f_2prime)
            #print('f_2prime expanded = ' + str(f_2prime))
            #print('f_2prime(0) = ' + str(y_2prime(x0, y0)))
            f_3inter = sp.diff(f_prime, x) + sp.diff(f_prime, y) * f_prime
            f_3prime = sp.diff(f_2prime, x) + f_prime * sp.diff(f_3inter, y)
            y_3prime = sp.lambdify((x,y), f_3prime)
            #print('f_3prime expanded = ' + str(f_3prime))
            #print('y_3prime(0) = ' + str(y_3prime(x0, y0)))
            t_series = y0 + y_prime(x0, y0) * (x-x0) + (y_2prime(x0, y0) / sp.factorial(2)) * (x-x0)**2 + (y_3prime(x0, y0) / sp.factorial(3)) * (x-x0)**3

        elif self.numerical_approximation["is_diff"] is False:
            #print('Approximation of f(x) = ' + test_function_string)
            #print('BC x(0) = ' + str(x0))
            #print('BC y(0) = ' + str(y0))

            f_in = eval(test_function_string)
            f_prime = sp.diff(f_in, x)
            y_prime = sp.lambdify(x, f_prime)
            f_2prime = sp.diff(f_prime, x)
            y_2prime = sp.lambdify(x, f_2prime)
            f_3prime = sp.diff(f_2prime, x)
            y_3prime = sp.lambdify(x, f_3prime)
            t_series = y0 + y_prime(x0) * (x-x0) + (y_2prime(x0) / sp.factorial(2)) * (x-x0)**2 + (y_3prime(x0) / sp.factorial(3)) * (x-x0)**3
        
        self.numerical_approximation["approx_function"] = sp.lambdify(x, t_series)
        

    def eval_function(self, min_range, max_range, increment):
        # min_range is integer or float
        # max_range is integer or float
        # increment is integer

        self.numerical_approximation["min_range"] = min_range
        self.numerical_approximation["max_range"] = max_range
        self.numerical_approximation["increment"] = increment
        self.numerical_approximation["x_range"] = np.linspace(min_range, max_range, num=increment) # assuming that x0 is at min range
        x_range = self.numerical_approximation["x_range"]
        test_function_string = self.numerical_approximation["test_function_string"]

        if self.numerical_approximation["step_type"] == "multiple":
            
            self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], self.numerical_approximation["y0"])
            for i in range(len(x_range) - 1):
                x0 = x_range[i]
                y0 = self.numerical_approximation["y_range"][i]
                x_eval = x_range[i + 1]
                self.series_approximation(test_function_string, x0, y0)
                self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], self.numerical_approximation["approx_function"](x_eval))

            if self.exact_solution["solution_string"] is not None:
                for i in x_range:
                    self.exact_solution["y_range"] = np.append(self.exact_solution["y_range"], self.exact_solution["function"](i))

            plt.plot(x_range, self.numerical_approximation["y_range"], color='orange', label='Taylor')
            if self.exact_solution["solution_string"] is not None:
                plt.plot(x_range, self.exact_solution["y_range"], color='blue', label='Exact', linestyle='dashed')
            plt.title('Taylor approx multiple steps vs exact solution')
            plt.legend()
            plt.show()

        elif self.numerical_approximation["step_type"] == "single":
            self.series_approximation(test_function_string, self.numerical_approximation["x0"], self.numerical_approximation["y0"])

            for i in x_range:
                self.numerical_approximation["y_range"] = np.append(self.numerical_approximation["y_range"], self.numerical_approximation["approx_function"](i))
                if self.exact_solution["solution_string"] is not None:
                    self.exact_solution["y_range"] = np.append(self.exact_solution["y_range"], self.exact_solution["function"](i))

            plt.plot(x_range, self.numerical_approximation["y_range"], color='orange', label='Taylor')
            if self.exact_solution["solution_string"] is not None:
                plt.plot(x_range, self.exact_solution["y_range"], color='blue', label='Exact')
            plt.legend()
            plt.title('Taylor approx vs exact solution')
            plt.show()


    def plot_error(self):
        if self.numerical_approximation["step_type"] == "single":
            self.numerical_approximation["numerical_error"] = np.array([])
            for i in self.numerical_approximation["x_range"]:
                self.numerical_approximation["numerical_error"] = np.append(self.numerical_approximation["numerical_error"], self.exact_solution["function"](i) - self.numerical_approximation["approx_function"](i))

        elif self.numerical_approximation["step_type"] == "multiple":    
            self.numerical_approximation["numerical_error"] = self.exact_solution["y_range"] - self.numerical_approximation["y_range"]

        plt.plot(self.numerical_approximation["x_range"], self.numerical_approximation["numerical_error"], color='red', label='error')
        plt.legend()
        plt.title('Numerical error of taylor approximation')
        plt.show()

    
    def eval_value(self, value):
        return np.interp(value, self.numerical_approximation["x_range"], self.numerical_approximation["y_range"])