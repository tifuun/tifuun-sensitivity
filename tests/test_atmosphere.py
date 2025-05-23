import unittest
import numpy as np

from nose2.tools import params

from tifuun_sensitivity import atmosphere

EL_TEST = 60
PWV_TEST = 1

class Test_Atmosphere(unittest.TestCase):
    @params(np.linspace(100, 1000), 
            np.array([100]), 
            [x for x in range(100, 1000)],
            100,
            100.12345)
    def test_atmosphere(self, F):
        eta = atmosphere.eta_atm_func(F, PWV_TEST, EL_TEST)
        if isinstance(F, np.ndarray):
            self.assertEqual(eta.size, F.size)
        elif isinstance(F, list):
            self.assertEqual(eta.size, len(F))
        else:
            self.assertTrue(eta.dtype == float)


if __name__ == "__main__":
    import nose2
    nose2.main()
