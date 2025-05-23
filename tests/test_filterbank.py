import unittest
import numpy as np

from nose2.tools import params

from tifuun_sensitivity import filterbank

N_TEST = 100

class Test_Filterbank(unittest.TestCase):
    @params(1, 2, 3)
    def test_Lorentzian(self, order):
        gamma = np.ones(N_TEST)
        F = N_TEST
        F_sky = np.ones(N_TEST)

        lorentzian = filterbank.Lorentzian(gamma, F_sky, F, order)

        self.assertEqual(lorentzian.size, F_sky.size)

    def test_filterbank(self):
        F = np.linspace(100,500, N_TEST)
        R = N_TEST

        fbank, F_sky, dF_sky = filterbank.generateFilterbankFromR(R, F)

        self.assertEqual(fbank.shape, (F_sky.size, F.size))


if __name__ == "__main__":
    import nose2
    nose2.main()
