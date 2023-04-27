import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class taylorSeries(object):
    def __init__(self):

        self.exact_solution = {
            "solution_string": None,
            "function": None,
        }

        self.numerical_approximation = {
            "x0": np.array([]),
            "y0": np.array([]),
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


    def load_function(self, test_function_string, exact_solution_string=None):
        self.numerical_approximation["test_function_string"] = test_function_string
        x = sp.Symbol('x')
        f_eval = eval(self.numerical_approximation["test_function_string"])
        self.numerical_approximation["test_function"] = sp.lambdify(x, f_eval)

        if exact_solution_string:
            self.exact_solution["solution_string"] = exact_solution_string
            x = sp.Symbol('x')
            f_eval = eval(self.exact_solution["solution_string"])
            self.exact_solution["function"] = sp.lambdify(x, f_eval)


    def set_parameters(self, x0, y0, step_type, is_diff):
        self.numerical_approximation['is_diff'] = is_diff # Input: True if differential eq - False if not
        self.numerical_approximation['step_type'] = step_type # "single" if Single step - "multiple" if Multiple steps
        self.numerical_approximation['x0'] = np.append(self.numerical_approximation['x0'], x0) # boundary condition (bc)
        self.numerical_approximation['y0'] = np.append(self.numerical_approximation['y0'], y0) # bc


    def series_approximation(self, test_function, x0, y0):

        # IMPORTANT - INPUT MAY BE DIFFERENTIAL EQUATION
        # dy/dx = f(x,y)
        # y(x) = y(x0) + y'(x0)(x-x0) + y"(x0)/2! * (x-x0)**2 + ... + y'N(x0)/N! * (x-x0)**N

        # y'(x0) = f(x0,y0) *** IF INPUT IS DIFFERENTIAL EQUATION
        # y''(x0) = [df/dx + df/dy f(x,y)] | x=x0 & y=y(x0)
        # y'''(x0) = {d/dx [df/dx + df/dy f(x,y)]} | x=x0 & y=y0 + {f(x,y) d/dy[df/dx + df/dy y'(x)]} | x=x0 & y=y0

        print('Approximation of dy/dx = ' + test_function)
        print('y(0) = ' + str(y0))
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        if self.numerical_approximation['is_diff']:
            f_prime = eval(test_function)
            y_prime = sp.lambdify((x,y), f_prime)
            print('f_prime expanded = ' + str(f_prime))
            print('f_prime(0) = ' + str(y_prime(x0,y0)))
            f_2prime = sp.diff(f_prime, x) + sp.diff(f_prime, y) * f_prime
            y_2prime = sp.lambdify((x,y), f_2prime)
            print('f_2prime expanded = ' + str(f_2prime))
            print('f_2prime(0) = ' + str(y_2prime(x0, y0)))
            f_3inter = sp.diff(f_prime, x) + sp.diff(f_prime, y) * f_prime
            f_3prime = sp.diff(f_2prime, x) + f_prime * sp.diff(f_3inter, y)
            y_3prime = sp.lambdify((x,y), f_3prime)
            print('f_3prime expanded = ' + str(f_3prime))
            print('y_3prime(0) = ' + str(y_3prime(x0, y0)))
            t_series = y0 + y_prime(x0, y0) * (x-x0) + (y_2prime(x0, y0) / sp.factorial(2)) * (x-x0)**2 + (y_3prime(x0, y0) / sp.factorial(3)) * (x-x0)**3

        elif self.numerical_approximation['is_diff'] is False:
            f_in = eval(test_function)
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

        self.numerical_approximation['min_range'] = min_range
        self.numerical_approximation['max_range'] = max_range
        self.numerical_approximation['increment'] = increment
        self.numerical_approximation["x0"] = np.linspace(min_range, max_range, num=increment) # assuming that x0 is at min range
        x_eval = self.numerical_approximation["x0"]
        test_function = self.numerical_approximation['test_function']

        if self.numerical_approximation['step_type'] is 'multiple':

            for i in range(len(x_eval) - 1):
                x0 = x_eval[i]
                y0 = self.numerical_approximation['y0'][i]
                iter_x_eval = x_eval[i + 1]
                self.series_approximation(test_function, x0, y0)
                self.numerical_approximation['y0'] = np.append(self.numerical_approximation['y0'], self.numerical_approximation["approx_function"](iter_x_eval))

            exact_solution_list = np.array([])
            for i in x_eval:
                if self.exact_solution["function"] is not None:
                    exact_solution_list = np.append(exact_solution_list, self.exact_solution['function'](i))

            plt.plot(x_eval, self.numerical_approximation['y0'], color='orange', label='Taylor')
            plt.plot(x_eval, exact_solution_list, color='blue', label='Exact', linestyle='dashed')
            plt.title('Taylor approx multiple steps vs exact solution')
            plt.legend()
            plt.show()

            self.numerical_approximation["numerical_error"] = exact_solution_list - self.numerical_approximation['y0']

            plt.plot(x_eval, self.numerical_approximation["numerical_error"], color='red', label='error')
            plt.legend()
            plt.title('Numerical error of multiple steps Taylor approximation')
            plt.show()

        elif self.numerical_approximation['step_type'] is 'single':

            taylor_eval_list = np.array([])
            exact_solution_list = np.array([])
            for i in x_eval:
                taylor_eval_list = np.append(taylor_eval_list, self.eval_taylor_x(i))
                if self.exact["function_string"] is not None:
                    exact_solution_list = np.append(exact_solution_list, self.eval_exact_x(i))

            plt.plot(x_eval, taylor_eval_list, color='orange', label='Taylor')
            if self.exact["function_string"] is not None:
                plt.plot(x_eval, exact_solution_list, color='blue', label='Exact')
            plt.legend()
            plt.title('Taylor approx vs exact solution')
            plt.show()
            

    def plot_error(self):
        if type(self.solution["x0"]) is np.ndarray:
            self.solution["numerical_error"] = np.array([])
            for i in self.solution["x0"]:
                self.solution["numerical_error"] = np.append(self.solution["numerical_error"], self.eval_exact_x(i) - self.eval_taylor_x(i))

        plt.plot(self.solution["x0"], self.solution["numerical_error"], color='red', label='error')
        plt.legend()
        plt.title('Numerical error of taylor approximation')
        plt.show()