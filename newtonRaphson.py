import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

def newtonRaphson(test_function, x0, conv_crit=0.001):

    # Function to find root
    x_data = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])
    y_data = np.array([4, 2.660543354, 1.177232934, -0.337311046, -1.6389167, -2.249906742, -1.286286569, 2.833233227, 12.87475734, 33.60298045,])


    x = sp.Symbol('x')
    f = eval(test_function)
    f_prime = f.diff(x)
    print('f = ' + str(f))
    print('f prime = ' + str(f_prime))
    y = sp.lambdify(x, f)
    y_prime = sp.lambdify(x, f_prime)
    
    conv = 1.0
    
    # Find the root by setting f(x) = 0

    # f(x) = f(a) + f'(a)(x-a)
    # f(x) = 0 => x0 = a - f(a)/f'(a)
    # x+1 = x0 - f(0)/f'(0)
    # xp+1 = xp - f(xp)/f'(xp) ## RAPID
    # xp+1 = xp - f(xp)/f'(a) ## SAFE
    
    xp = float(x0)
    print('x0 = ' + str(xp))

    x_ns = [xp]
    y_ns = [y(xp)]
    conv_array = [conv]
    
    while conv >= conv_crit:

        x_p1 = float(xp - y(xp)/y_prime(xp))
        x_ns.append(x_p1)
        y_ns.append(y(x_p1))
        conv = abs(xp - x_p1)/x_p1
        conv_array.append(conv)
        print('conv = ' + str(conv))
        print('x_p1 = ' + str(x_p1))
        xp = x_p1

    x_ns = np.array(x_ns)
    y_ns = np.array(y_ns)

    plt.subplot(2,1,1)
    plt.title('Exact function')
    plt.plot(x_data, y_data, color='blue')
    plt.scatter(x_ns, y_ns, marker='x', color='green')

    plt.subplot(2,1,2)
    plt.title('Convergence')
    plt.plot(range(len(conv_array)), conv_array, color='red', marker='s')

    plt.show()
