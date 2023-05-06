import numpy as np
from scipy.io import FortranFile
import scipy.linalg
from pathlib import Path



def _check_args(arg, allowed_args):
    if not arg in allowed_args:
        raise ValueError(f"Invalid value: {args}. Allowed values are {allowed_args}.")

class PlanckLogLike(object):
    DATADIR = Path(__file__).parent / "data"

    def __init__(self, year: int, spectra: str, include_low_ell_TT: bool):
        
        _check_args(year, {2015, 2022})
        _check_args(spectra, {"TT", "TTTEEE"})
        _check_args(include_low_ell_TT, {True, False})


        self.year = year
        self.spectra = spectra 
        self.include_low_ell_TT = include_low_ell_TT

        self._resolve_binning_setup()
        self._resolve_data_file_dependencies()


        self.initialize_W()
        self.initialize_C_ell_b()
        self.initialize_C_inv()

        return

    def __call__(self, C_ell_b):
        return

    def initialize_W(self):
        return

    def initialize_C_ell_b(self):
        return

    def initialize_C_inv(self):
        return

    def _get_C(self):
        return

    def _resolve_data_file_dependencies(self):

        if self.year == 2015:
            self.data_dir = self.DATA_DIR / "planck2015_plik_lite"
            self.version = 18
        elif self.year == 2018:
            self.data_dir = self.DATA_DIR / "planck2018_plik_lite"
            self.version = 22

        self.likelihood_file = self.data_dir / "cl_cmb_plik_v{version}.dat"
        
        
        return

    def _resolve_binning_setup(self):
        """
        This function sets up the binning scheme. It will resolve the
        number of bins in the TT, TE, and EE spectra, as well as the
        number of multipoles expected as input to likelihood function.
        """
        self.lmax = 2508
        self.N_bin_TT_high_ell = 215

        self.N_bin_TE = 199 if self.spectra == "TTTEEE" else 0
        self.N_bin_EE = 199 if self.spectra == "TTTEEE" else 0

        self.N_bin_TT_low_ell = 2 if self.include_low_ell_TT else 0
        self.lmin_TT = 2 if self.include_low_ell_TT else 30

        self.N_bin_TT = self.N_bin_TT_low_ell + self.N_bin_TT_high_ell
        self.N_bin_high_ell = self.N_bin_TT_high_ell + self.N_bin_TE + self.N_bin_EE

        return


if __name__ == "__main__":
    PlanckLogLike(year=2015, spectra="TT", include_low_ell_TT=False)
