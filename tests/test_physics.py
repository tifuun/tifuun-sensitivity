import unittest
import numpy as np

from nose2.tools import params

from tifuun_sensitivity import physics

N_TEST = 100

class Test_Physics(unittest.TestCase):
    @params(np.ones(N_TEST), np.ones((N_TEST))) 
    def test_rad_trans(self, eta):
        rad_in = np.ones(N_TEST)
        medium = np.ones(N_TEST)

        rad_out = physics.rad_trans(rad_in, medium, eta)

        self.assertEqual(rad_out.shape, eta.shape)

    @params("Planck", "Rayleigh-Jeans")
    def test_T_from_psd(self, method):
        F = np.ones(N_TEST)
        psd = np.ones(N_TEST)

        T = physics.T_from_psd(F, psd, method)

        self.assertEqual(F.shape, T.shape)
        self.assertEqual(psd.shape, T.shape)
    
    @params(N_TEST, np.ones(N_TEST))
    def test_johnson_nyquist_psd(self, T):
        F = np.ones(N_TEST)

        psd = physics.johnson_nyquist_psd(F, T)

        self.assertEqual(F.shape, psd.shape)

    @params(N_TEST, np.ones(N_TEST))
    def test_nph(self, T):
        F = np.ones(N_TEST)

        n_phot = physics.nph(F, T)

        self.assertEqual(F.shape, n_phot.shape)

if __name__ == "__main__":
    import nose2
    nose2.main()
