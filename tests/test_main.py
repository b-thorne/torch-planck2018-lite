import unittest
import numpy as np
import torch
from src.main import PlanckLitePy


class TestPlanckLitePy(unittest.TestCase):
    def setUp(self):
        self.ls, self.Dltt, self.Dlte, self.Dlee = np.genfromtxt(
            "data/Dl_planck2015fit.dat", unpack=True
        )
        self.ellmin = int(self.ls[0])
        self.Dltt = torch.tensor(self.Dltt)
        self.Dlte = torch.tensor(self.Dlte)
        self.Dlee = torch.tensor(self.Dlee)

    def test_2018_TTTEEE_highl(self):
        lnL = PlanckLitePy(year=2018, spectra="TTTEEE", use_low_ell_bins=False)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -291.33481235418026, delta=1e-4)

    def test_2015_TTTEEE_highl(self):
        lnL = PlanckLitePy(year=2015, spectra="TTTEEE", use_low_ell_bins=False)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -280.9388125627618, delta=1e-4)

    def test_2018_TTTEEE_lowl(self):
        lnL = PlanckLitePy(year=2018, spectra="TTTEEE", use_low_ell_bins=True)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -293.95586501795134, delta=1e-4)

    def test_2015_TTTEEE_lowl(self):
        lnL = PlanckLitePy(year=2015, spectra="TTTEEE", use_low_ell_bins=True)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -283.1905700256343, delta=1e-4)

    def test_2015_TT_lowl(self):
        lnL = PlanckLitePy(year=2015, spectra="TT", use_low_ell_bins=True)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -104.59579619576277, delta=1e-4)

    def test_2018_TT_lowl(self):
        lnL = PlanckLitePy(year=2018, spectra="TT", use_low_ell_bins=True)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -104.20228335099686, delta=1e-4)

    def test_2018_TT_hil(self):
        lnL = PlanckLitePy(year=2018, spectra="TT", use_low_ell_bins=False)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -101.58123068722583, delta=1e-4)

    def test_2015_TT_hil(self):
        lnL = PlanckLitePy(year=2015, spectra="TT", use_low_ell_bins=False)
        loglikelihood = lnL.loglike(
            self.Dltt,
            self.Dlte,
            self.Dlee,
            self.ellmin,
        )
        self.assertAlmostEqual(loglikelihood, -102.34403873289027, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
