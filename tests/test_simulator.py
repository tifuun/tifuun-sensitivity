import unittest
import numpy as np

from nose2.tools import params

from tifuun_sensitivity import simulator
from tifuun_sensitivity.data import DESHIMA2

N_TEST = 100

class Test_Simulator(unittest.TestCase):
    def test_spectrometer_sensitivity(self):
        out = simulator.calculator(DESHIMA2.cascade_list, 
                                   DESHIMA2.telescope, 
                                   DESHIMA2.instrument)

        for key, item in out.items():
            if isinstance(item, np.ndarray):
                for elem in item.ravel():
                    self.assertFalse(np.isnan(elem))

if __name__ == "__main__":
    import nose2
    nose2.main()
