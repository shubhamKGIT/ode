from scipy.integrate import odeint
from scipy.integrate import OdeSolver
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.pyplot as plt

import numpy as np
from IntegrationMethods import RK4

DaNum: float = 0.2
Re: int = 1.5
Beta: float = 20

def der_fn(y: list, x: float):
    """gives back derivative
    [f', f'']
    f'' = e^f
    """
    dydx = np.array([Re*y[1], Re*y[1] - Re*DaNum*np.exp(-Beta/y[0])])
    """if x< -2:
        dydx = np.array([y[1], 0.])
    elif x>0 and x< 2:
        dydx = np.array([Re*y[1], Re*y[1] - Re*DaNum*np.exp(-Beta/y[0])])
    else:
        dydx = np.array([y[1], 0.])
        """
    
    return dydx

def jacobian(y, x):
    """[[df1/dx1, df1/dx2],
        [df2/dx1, df2/df2]]
    """
    tol = 1e-6
    J = [[ (Re*y[1] - Re*DaNum*np.exp(-Beta/y[0])) / (y[1] + tol), Re*y[1] - Re*DaNum*np.exp(-Beta/y[0]) ], 
        [ (Re*y[1]*(1 + (Beta/(y[0]**2))*DaNum*np.exp(-Beta/y[0])) - Re*DaNum*np.exp(-Beta/y[0]))/ (y[1]+tol), Re*y[1]*(1 + (Beta/(y[0]**2))*DaNum*np.exp(-Beta/y[0])) - Re*DaNum*np.exp(-Beta/y[0])]
        ]
    return J

if __name__=='__main__':
    x = np.arange(-10, 2, 0.01)   #chi ( 0 - 5) where equation valid
    y_s = odeint(func = der_fn, y0= [0.2, 0.01], t= x, Dfun= jacobian)
    print(x)
    #y_n = solve_ivp(fun = der_fn, t_span= x, y0= 0., method="RK45")
    print(y_s)
    plt.plot(x, y_s)
    plt.show()