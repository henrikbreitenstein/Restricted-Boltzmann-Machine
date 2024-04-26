import matplotlib
import numpy as np
from scipy.optimize import curve_fit

def fit(x, y, param, name, verbose=True, plot=True):

    def inverse(x, a, b, c):
        return a/(x+b) + c

    def logarithmic(x, a, b, c):
        return a*np.log(b*x) + c

    def exponential(x, a, b, c):
        return a*np.exp(b*x) + c

    def linear(x, a, b):
        return a*x + b
    
    type_array = np.array([
        "Inverse",
        "Logarithmic",
        "Exponential",
        "Linear"
    ])

    func_array = np.array([
        inverse,
        logarithmic,
        exponential,
        linear
    ])

    ab = np.array([1, 1])
    abc = np.array([0.1, 0.1, 0.1])
    
    try:
        fitted_inv, pcov_inv    = curve_fit(inverse, x, y, abc)
    except:
        fitted_inv = abc
    try:   
        fitted_log, pcov_log    = curve_fit(logarithmic, x, y, abc)
    except:
        fitted_log = abc
    try:
        fitted_exp, pcov_exp    = curve_fit(exponential, x, y, abc)
    except:
        fitted_exp = abc
    try:
        fitted_lin, pcov_linear = curve_fit(linear, x, y, ab)
    except:
        fitted_lin = ab

    fitted_array = np.array([
        fitted_inv,
        fitted_log,
        fitted_exp,
        fitted_lin
    ], dtype=object)

    inverse_err     = np.mean((y - inverse(x, *fitted_inv))**2)
    logarithmic_err = np.mean((y - logarithmic(x, *fitted_log))**2)
    exponential_err = np.mean((y - exponential(x, *fitted_exp))**2)
    linear_err      = np.mean((y - linear(x, *fitted_lin))**2)
    
    err_array = np.array([
        inverse_err,
        logarithmic_err,
        exponential_err,
        linear_err
    ])
    
    best_arg = np.argmin(err_array)
    if verbose:
        print(f"""

        Inverse fit: var = {inverse_err}
        Logarithmic fit: var = {logarithmic_err}
        Expnonential fit: var ={exponential_err}
        Linear fit: var = {linear_err}

        With the best fit:
        
        type : {type_array[best_arg]}
        var = {err_array[best_arg]}
        parameters:

        {fitted_array[best_arg]}

        """)
    
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'size'   : 20}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(9,7))
    plt.xlabel("n")
    plt.ylabel(param)
    plt.scatter(x, y, label="Data")
    
    x_ext = np.arange(1, 20)
    y_fit = func_array[best_arg](x_ext, *fitted_array[best_arg])
    plt.plot(x_ext, y_fit, label="Fitted")
    plt.legend()

    if plot:
        plt.show()
    plt.savefig(f"Figures/fitting/{name}.pdf")


if __name__ == '__main__':
    n = np.array([2, 4, 6, 8, 10], dtype=np.float64),
    lr = np.array([4.18, 4.1, 4.9, 5.5, 5.75])
    fit(n, lr, r"$\eta$", "ising_lr_fit.pdf")








