import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class taylorSeries(object):
    def __init__(self, is_diff, step_type):

        self.is_diff = is_diff # Input: True if differential eq - False if not
        self.step_type = step_type # Single if Single step - Multiple if Multiple steps

        self.exact = {
            "function_string": None
        }

        self.solution = {
            "min_range": None,
            "max_range": None,
            "increment": None,
            "numerical_x_values": None,
            "numerical_approx_function": None,
            "numerical_error": None
        }

    def exact_solution(self,function_string):
        self.exact["function_string"] = function_string
        
    def eval_exact_x(self, x_val):
        x = sp.Symbol('x')
        f_eval = eval(self.exact["function_string"])
        self.solution["exact_function"] = sp.lambdify(x, f_eval)
        return self.solution["exact_function"](x_val)

    def eval_taylor_x(self, x_val):
        return self.solution["numerical_approx_function"](x_val)

    def plot_solution(self, min_range, max_range, increment):
        # min_range is integer or float
        # max_range is integer or float
        # increment is integer

        self.solution["numerical_x_values"] = np.linspace(min_range, max_range, num=increment)
        x_eval = self.solution["numerical_x_values"]
        # x_eval is numpy.array format

        if type(x_eval) is np.ndarray:
            taylor_eval_list = np.array([])
            exact_solution_list = np.array([])
            for i in x_eval:
                taylor_eval_list = np.append(taylor_eval_list, self.eval_taylor_x(i))
                if self.exact["function_string"] is not None:
                    exact_solution_list = np.append(exact_solution_list, self.eval_exact_x(i))
        else:
            raise Exception("INVALID RANGE TYPE IN PLOT SOLUTION")

        plt.plot(x_eval, taylor_eval_list, color='orange', label='Taylor')
        if self.exact["function_string"] is not None:
            plt.plot(x_eval, exact_solution_list, color='blue', label='Exact')
        plt.legend()
        plt.title('Taylor approx vs exact solution')
        plt.show()

    def plot_error(self):
        if type(self.solution["numerical_x_values"]) is np.ndarray:
            self.solution["numerical_error"] = np.array([])
            for i in self.solution["numerical_x_values"]:
                self.solution["numerical_error"] = np.append(self.solution["numerical_error"], self.eval_exact_x(i) - self.eval_taylor_x(i))

        plt.plot(self.solution["numerical_x_values"], self.solution["numerical_error"], color='red', label='error')
        plt.legend()
        plt.title('Numerical error of taylor approximation')
        plt.show()

    def series_approximation(self, test_function, x0, y0):

        # IMPORTANT - INPUT MAY BE DIFFERENTIAL EQUATION

        # y(x) = y(x0) + y'(x0)(x-x0) + y"(x0)/2! * (x-x0)**2 + ... + y'N(x0)/N! * (x-x0)**N

        # y'(x0) = f(x0,y0) *** IF INPUT IS DIFFERENTIAL EQUATION
        # y''(x0) = [df/dx + df/dy f(x,y)] | x=x0 & y=y(x0)
        # y'''(x0) = {d/dx [df/dx + df/dy f(x,y)]} | x=x0 & y=y0 + {f(x,y) d/dy[df/dx + df/dy y'(x)]} | x=x0 & y=y0

        print('Approximation of dy/dx = ' + test_function)
        print('y(0) = ' + str(y0))
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        if self.is_diff:
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

        elif self.is_diff is False:
            f_in = eval(test_function)
            f_prime = sp.diff(f_in, x)
            y_prime = sp.lambdify(x, f_prime)
            f_2prime = sp.diff(f_prime, x)
            y_2prime = sp.lambdify(x, f_2prime)
            f_3prime = sp.diff(f_2prime, x)
            y_3prime = sp.lambdify(x, f_3prime)

            t_series = y0 + y_prime(x0) * (x-x0) + (y_2prime(x0) / sp.factorial(2)) * (x-x0)**2 + (y_3prime(x0) / sp.factorial(3)) * (x-x0)**3

        self.solution["numerical_approx_function"] = sp.lambdify(x, t_series)

    def mutiple_steps(self, test_function, x0, y0, min_range, max_range, increment):
        self.solution["min_range"] = min_range
        self.solution["max_range"] = max_range
        self.solution["increment"] = increment
        self.solution["numerical_x_values"] = np.linspace(min_range, max_range, num=increment) # assuming that x0 is at min range

        iter_y0 = y0
        iter_y0_array = np.array([y0]) # assuming that x0 and y0 are the starting point
        for i in range(len(self.solution["numerical_x_values"]) - 1):
            iter_x0 = self.solution["numerical_x_values"][i]
            iter_y0 = iter_y0_array[i]
            iter_x_eval = self.solution["numerical_x_values"][i + 1]
            self.series_approximation(test_function, iter_x0, iter_y0)
            iter_y0_array = np.append(iter_y0_array, self.solution["numerical_approx_function"](iter_x_eval))

        exact_solution_list = np.array([])
        for i in self.solution["numerical_x_values"]:
            if self.exact["function_string"] is not None:
                exact_solution_list = np.append(exact_solution_list, self.eval_exact_x(i))

        plt.plot(self.solution["numerical_x_values"], iter_y0_array, color='orange', label='Taylor')
        plt.plot(self.solution["numerical_x_values"], exact_solution_list, color='blue', label='Exact', linestyle='dashed')
        plt.title('Taylor approx multiple steps vs exact solution')
        plt.legend()
        plt.show()

        self.solution["numerical_error"] = exact_solution_list - iter_y0_array

        plt.plot(self.solution["numerical_x_values"], self.solution["numerical_error"], color='red', label='error')
        plt.legend()
        plt.title('Numerical error of multiple steps Taylor approximation')
        plt.show()