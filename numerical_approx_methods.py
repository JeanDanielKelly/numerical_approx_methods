# numerical approximation main file
# test dev branch
from taylorSeries import taylorSeries
from rungeKutta import rungeKutta

def main():
    # x_data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2 ])
    # y_data = np.array([-2,-1.894829082, -1.778597242, -1.650141192, -1.508175302, -1.351278729, -1.1778812, -0.986247293, -0.774459072, -0.540396889, -0.281718172, 0.004166024, 0.320116923, 0.669296668, 1.055199967, 1.48168907, 1.953032424, 2.473947392, 3.049647464, 3.685894442, 4.389056099])

    # x_data = np.array([])
    # y_data = np.array([])

    # newtonRaphson('sp.exp(2*x)-7*sp.exp(x)+10', 2.0)

    # approximate dy/dx = (x^2)(y^2) for |x| < 1
    # subject to y(0) = 3
    # can be obtained straightforwardly: y(x) = 3/(1-x^3)
    # Therefore the exact solution at x = 0.3 => y(0.3) = 3 / 0.97300 ~ 3.08325
    
    rk_approx = rungeKutta()
    rk_approx.load_function('x**2 * y**2','-3/(-1+x**3)')
    rk_approx.set_parameters(0, 3, 'single')
    rk_approx.eval_function(0, 0.99, 50)
    rk_approx.plot_error()
    print(rk_approx.eval_value(0.3))
    
    if 1 == 0:
        num_approx = taylorSeries()
        num_approx.load_function('x**2 * y**2','3/(1-x**3)')
        num_approx.set_parameters(0, 3,'single', True)
        num_approx.eval_function(0, 0.5, 100)
        print(num_approx.eval_value(0.1))
        num_approx.plot_error()

        num_approx_cos = taylorSeries()
        num_approx_cos.load_function('sp.cos(x)', 'sp.cos(x)')
        num_approx_cos.set_parameters(0, 1,'single', False)
        num_approx_cos.eval_function(-3.1416, 3.1416, 50)
        num_approx_cos.plot_error()

        num_approx_multiple = taylorSeries()
        num_approx_multiple.load_function('x**2 * y**2','3/(1-x**3)')
        num_approx_multiple.set_parameters(0, 3,'multiple', True)
        num_approx_multiple.eval_function(0, 0.3, 50)
        num_approx_multiple.plot_error()

        num_approx_multiple_cos = taylorSeries()
        num_approx_multiple_cos.load_function('sp.cos(x)', 'sp.cos(x)')
        num_approx_multiple_cos.set_parameters(0, 1,'multiple', False)
        num_approx_multiple_cos.eval_function(0, 2 * 3.1416, 50)
        num_approx_multiple_cos.plot_error()

        approx_func = taylorSeries()
        approx_func.load_function('sp.exp(-x**2)')
        approx_func.set_parameters(0, 0, 'multiple', True)
        approx_func.eval_function(0, 10, 50)
        print(approx_func.eval_value(10))

        FO_L_DE = taylorSeries()
        FO_L_DE.load_function('-2 * x**3 - 2 * y * x','1 - x**2 + 1 * sp.exp(-x * x)')
        FO_L_DE.set_parameters(0, 2, 'single', True)
        FO_L_DE.eval_function(-2, 2, 50)
        FO_L_DE.plot_error()


if __name__ == "__main__":
    main()