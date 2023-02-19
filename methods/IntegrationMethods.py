# Runge kutta arbitrary class which will then be used to build classes for specific order of integration
import numpy as np
from scipy.integrate import (
    odeint,
    solve_ivp,
    solve_bvp
)
from typing import (
    Union,
    TypeVar,
    Optional,
    Any,
    Callable
)

ind_var = TypeVar("ind_var", Optional[Union[list, np.array]], Any)   # independent variable
dep_var_elem = TypeVar("dep_var_elem", Optional[Union[list, np.array]], Any)
dep_var = TypeVar("dep_var", Optional[Union[list, np.array, np.ndarray]], Any)    # dependent variable

class RungeKutta:
    "Abstract class for implementing Runge Kutta integrattion numerical method"
    def __init__(self) -> Any:
        pass
    def derivative(self, x_n: float, y_n: dep_var_elem) -> dep_var_elem:
        """reaceives a custom derivative fn with variables to be made derivative of
        returns f', f'', f'', etc. in a list"""
        return NotImplementedError
    def integrate(self, U0: dep_var_elem, x: ind_var, h: float, coeffs: ind_var) -> dep_var:
        return NotImplementedError

class RK4(RungeKutta):
    """ Fourth order runge kutta method implementation
    """
    def __init__(self, der_fn: Callable[[float, dep_var_elem], dep_var_elem], eqn_order: int) -> None:
        self.der_fn = der_fn 
        self.eqn_order = eqn_order
    def derivative(self, x_n: float, y_n: dep_var_elem) -> dep_var_elem:
        "y_n contains elements [f, f', f''] adn returns the derivatives [f', f'', f'''] as per the ODE equation passed"
        y_der = self.der_fn(x_n, y_n)
        return y_der
    def _step(self, x_n: float, y_n: dep_var_elem, h: float, coeffs: np.array) -> dep_var_elem:
        " extrapolation for a single point neighbourhood"
        k1 = self.derivative(x_n, y_n)
        k2 = self.derivative(x_n + h/2., y_n + k1*(h/2.))
        k3 = self.derivative(x_n + h/2., y_n + k2*(h/2.))
        k4 = self.derivative(x_n + h, y_n + k3*h)
        y_s = np.inner(coeffs, h*np.array([k1, k2, k3, k4]).transpose())
        return y_s
    def integrate(self, Y0: dep_var_elem, x: ind_var, h: float, coeffs : ind_var) -> dep_var:
        """takes: 
                step size h from outside, dep vairiable list/array, and inital condition on Y or Y0
            returns:
                y_final element
        """
        y = np.zeros(shape = (len(x), self.eqn_order+1))
        y[0, :] = Y0   # specify first row/element i.e. [f(x0), f'(x0), f''(x0)] from boundary condition
        if coeffs is None:
            coeffs = np.array([1/8., 3./8., 3./8., 1/8.])   
        else:
            coeffs = np.array(coeffs)    # can be passed as [1/6., 2/6., 2/6., 1/6.] from outside if needed
        # need to get length of x so we can pass it in the loop
        try:
            len_x = len(x)
        except:
            len_x = x.shape[0]
        for n in range(len_x -1):
            # do it for all Xs using values at the first x: X0
            x_n = x[n]
            y_n = y[n, :]
            # do it for f, f', f'', etc together by calling step
            y[n+1, :] = self._step(x_n, y_n, h, coeffs)
        return y

if __name__=='__main__':
    print(f"Running Main")
    def second_order_der(x: float, y: dep_var_elem) -> dep_var_elem:
        y_der: dep_var_elem = np.array([1., 2., 0.])
        return y_der

    rk = RK4(second_order_der, 2)
    dydt = rk.derivative(1., [1., 0., 0.])
    print(dydt)
    print(f"Finished running main fn!")