from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np
import matplotlib.pyplot as plt

RePr = 2
Da = 0.6
Beta = 6
L_inv = 1

def der_fun_TandY(U, x):
    """
    returns [T', T'', Y', Y'']
    """
    dUdx = np.array([U[1], RePr*U[1]  - RePr*Da*np.exp(-Beta/U[0])*U[2], U[3], L_inv*RePr*U[1]  - L_inv*RePr*10*Da*np.exp(-Beta/U[0])*U[1]])
    return dUdx

def def_fn_y(y, x, T):
    dYdx = x - np.exp(1/T)*y
    return dYdx

if __name__=='__main__':
    x = np.arange(0, 2, 0.1)
    y_s = ideint(der_fun_TandY, )
    y_s = solve_ivp(fun, x, 0.2, method="RK45")
    plt.plot(x, y)
    plt.show()