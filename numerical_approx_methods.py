# numerical approximation main file
# test dev branch
from taylorSeries import taylorSeries

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

    # taylorSeries('x**2 * y**2',0,3,0.3)
    num_approx = taylorSeries(True, 'single')
    num_approx.exact_solution('3/(1-x**3)')
    num_approx.series_approximation('x**2 * y**2', 0, 3)
    num_approx.plot_solution(0, 0.5, 100)
    num_approx.plot_error()

    num_approx_cos = taylorSeries(False, 'single')
    num_approx_cos.exact_solution('sp.cos(x)')
    num_approx_cos.series_approximation('sp.cos(x)', 0, 1)
    num_approx_cos.plot_solution(-3.1416, 3.1416, 100)
    num_approx_cos.plot_error()

    num_approx_multiple = taylorSeries(True, 'multiple')
    num_approx_multiple.exact_solution('3/(1-x**3)')
    num_approx_multiple.mutiple_steps('x**2 * y**2', 0, 3, 0, 0.5, 50)

    num_approx_multiple_sin = taylorSeries(False, 'multiple')
    num_approx_multiple_sin.exact_solution('sp.cos(x)')
    num_approx_multiple_sin.mutiple_steps('sp.cos(x)', 0, 1, 0, 3.1416*2, 50)

if __name__ == "__main__":
    main()