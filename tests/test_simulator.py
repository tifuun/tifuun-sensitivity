import unittest
import numpy as np

from nose2.tools import params

from tifuun_sensitivity import simulator
from tifuun_sensitivity.data import DESHIMA2

N_TEST = 100

class Test_Simulator(unittest.TestCase):
    def test_spectrometer_sensitivity(self):
        R = 500
        F = np.linspace(220, 440, 350) * 1e9
        out = simulator.calculator(DESHIMA2.cascade_list, 
                                   DESHIMA2.telescope, 
                                   DESHIMA2.instrument)
        #### TODO: test outputs for sizes
        #self.assertEqual(lorentzian.size, F_sky.size)



if __name__ == "__main__":
    import nose2
    nose2.main()
