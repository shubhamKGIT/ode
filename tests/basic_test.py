import unittest

from ODE.methods.IntegrationMethods import RK4
from ODE.methods.IntegrationMethods import ind_var, dep_var, dep_var_elem
import numpy as np


class BasicClassDef(unittest.TestCase):

    def test_derivative_method(self):
        def constant_der_fn(x: float, y: dep_var_elem) -> dep_var_elem:
            y_der: dep_var_elem = np.array([1., 1., 0.])
            return y_der
        def first_order(x: float, y: dep_var_elem) -> dep_var_elem:
            y_der: dep_var_elem = np.array([y[1], 0., 0.])
            return y_der
        def same_as_y(x: float, y: dep_var_elem) -> dep_var_elem:
            y_der: dep_var_elem = y
            return y_der
        rk = RK4(der_fn= same_as_y, eqn_order=1)
        self.assertListEqual(rk.derivative(x_n= 0., y_n= [1., 0.]), [1., 0.])    
        rk1 = RK4(constant_der_fn, 1)
        x = np.arange(1, 2, 0.1)


if __name__=="__main__":
    unittest.main()