import healpy as hp
import numpy as np
import unittest
from pynilc.sht import HealpixDUCC

class TestHealpixDUCC(unittest.TestCase):
    def setUp(self):
            self.nside = 64
            self.lmax = 3 * self.nside - 1
            self.mmax = self.lmax

            L = np.arange(self.lmax + 1)
            spectra = (L * (L + 1)) +1e-2
            tt = 1e-4 / spectra
            ee = 1e-5 / spectra
            bb = 1e-6 / spectra
            te = np.zeros(self.lmax + 1)
            
            np.random.seed(56781)
            self.alm = hp.synalm([tt, ee, bb, te], lmax=self.lmax, new=True)
            self.m = hp.alm2map(self.alm, self.nside)

            self.hducc = HealpixDUCC(nside=self.nside)

    
    def test_alm2map(self):
        m = self.hducc.alm2map(self.alm, self.lmax, nthreads=1)
        self.assertTrue(np.allclose(m, self.m))

    def test_map2alm(self):
        m = self.hducc.alm2map(self.alm, self.lmax, nthreads=1)
        alm = self.hducc.map2alm(m, self.lmax, nthreads=1)
        self.assertTrue(np.allclose(alm[:,2:], self.alm[:,2:],atol=1e-1))
    
if __name__ == "__main__":
    unittest.main()
