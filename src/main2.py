import numpy as np
from scipy.io import FortranFile
import scipy.linalg
from pathlib import Path


def _check_args(arg, allowed_args):
    if not arg in allowed_args:
        raise ValueError(f"Invalid value: {arg}. Allowed values are {allowed_args}.")


class PlanckLogLike(object):
    DATA_DIR = Path(__file__).parent.parent / "data"

    def __init__(self, year: int, spectra: str, include_low_ell_TT: bool):
        _check_args(year, {2015, 2022})
        _check_args(spectra, {"TT", "TTTEEE"})
        _check_args(include_low_ell_TT, {True, False})

        self.year = year
        self.spectra = spectra
        self.include_low_ell_TT = include_low_ell_TT

        self._resolve_binning_setup()
        self._read_high_ell_data_dependencies()
        if include_low_ell_TT:
            self._read_low_ell_data_dependencies()
        self._handle_low_ell_TT_combination()

        self.initialize_W()
        self.C_inv = np.linalg.inv(self.C)

        return

    def __repr__(self):
        return f"""PlanckLogLike(year={self.year}, spectra={self.spectra}, include_low_ell_TT={self.include_low_ell_TT})"""

    def __str__(self):
        return f"""
        {self.__repr__()}

        Info:
        =====

        Year: {self.year}
        Spectra: {self.spectra}
        Use low ell TT data: {self.include_low_ell_TT}

        Number of TT bins: {self.N_bin_TT}
        Number of TE bins: {self.N_bin_TE}
        Number of EE bins: {self.N_bin_EE}

        Total number of data bins: {self.N_bin_total}
        Expected shape of input spectra: {self.N_multipoles}

        Lmin input spectra: {self.lmin_spectra}
        Lmax input spectra: {self.lmax}
        Lmin TT data: {self.lmin_TT_data}
        Lmin TE and EE data: {self.lmin_TE_EE_data}

        Shape of weights matrix W: {self.W.shape}
        Shape of inverse covariance matrix C_inv: {self.C_inv.shape}

        
        """

    def __call__(self, C_ell_b):
        R = self.C_hat_ell_b - self.W @ C_ell_b
        return - 0.5 * R.T @ self.C_inv @ R

    def initialize_W(self):
        # W is the weight / windowing matrix that operatres on input spectra.
        # The input spectra ar the three TT, TE, EE spectra at multipoles
        # 2 - 2508. Therefore, W will have shape (N_bin_total, 3 * 2507).
        self.lmin_spectra = 2
        W = np.zeros((self.N_bin_total, self.N_multipoles))
        for row in range(self.N_bin_TT):
            inds = slice(
                self.blmin_TT[row] + self.lmin_TT_data - self.lmin_spectra,
                self.blmax_TT[row] + self.lmin_TT_data - self.lmin_spectra + 1,
            )
            W[row][inds] = self.bin_w_TT[self.blmin_TT[row] : self.blmax_TT[row] + 1]

        if self.include_low_ell_TT:
            for row in range(self.N_bin_TE):
                inds = slice(
                    2507 + self.blmin[row] + self.lmin_TE_EE_data - self.lmin_spectra,
                    2507 + self.blmax[row] + self.lmin_TE_EE_data - self.lmin_spectra + 1,
                )
                W[row + self.N_bin_TT][inds] = self.bin_w[
                    self.blmin[row] : self.blmax[row] + 1
                ]

            for row in range(self.N_bin_EE):
                inds = slice(
                    2 * 2507 + self.blmin[row] + self.lmin_TE_EE_data - self.lmin_spectra,
                    2 * 2507 + self.blmax[row] + self.lmin_TE_EE_data - self.lmin_spectra + 1,
                )
                W[row + self.N_bin_TT + self.N_bin_TE][inds] = self.bin_w[
                    self.blmin[row] : self.blmax[row] + 1
                ]
        self.W = W

    def _handle_low_ell_TT_combination(self):
        # Append low ell quantities to high ell quantities.
        if self.include_low_ell_TT:
            self.blmin_TT = np.concatenate(
                (self.blmin_low_ell, self.blmin + len(self.bin_w_low_ell))
            )
            self.blmax_TT = np.concatenate(
                (self.blmax_low_ell, self.blmax + len(self.bin_w_low_ell))
            )
            self.bin_w_TT = np.concatenate((self.bin_w_low_ell, self.bin_w))
            self.C_hat_ell_b = np.concatenate(
                (self.C_hat_ell_b_low_ell, self.C_hat_ell_b)
            )

            self.C = np.zeros((self.N_bin_total, self.N_bin_total))
            self.C[0:2, 0:2] = np.diag(self.C_hat_ell_b_sigma_low_ell ** 2)
            self.C[2:, 2:] = self.covmat
        else:
            self.blmin_TT = self.blmin
            self.blmax_TT = self.blmax
            self.bin_w_TT = self.bin_w

    def _read_low_ell_data_dependencies(self):
        data_dir = self.DATA_DIR / f"planck{self.year}_low_ell"

        _, self.C_hat_ell_b_low_ell, self.C_hat_ell_b_sigma_low_ell = np.genfromtxt(
            data_dir / f"CTT_bin_low_ell_{self.year}.dat", unpack=True
        )

        self.blmin_low_ell = np.loadtxt(data_dir / "blmin_low_ell.dat").astype(int)
        self.blmax_low_ell = np.loadtxt(data_dir / "blmax_low_ell.dat").astype(int)
        self.bin_w_low_ell = np.loadtxt(data_dir / "bweight_low_ell.dat")

    def _read_high_ell_data_dependencies(self):
        if self.year == 2015:
            data_dir = self.DATA_DIR / "planck2015_plik_lite"
            version = 18
        elif self.year == 2018:
            data_dir = self.DATA_DIR / "planck2018_plik_lite"
            version = 22

        _, self.C_hat_ell_b, _ = np.genfromtxt(
            data_dir / f"cl_cmb_plik_v{version}.dat", unpack=True
        )
        self.cov_file = data_dir / f"c_matrix_plik_v{version}.dat"
        self.blmin = np.loadtxt(data_dir / "blmin.dat").astype(int)
        self.blmax = np.loadtxt(data_dir / "blmax.dat").astype(int)
        self.bin_w = np.loadtxt(data_dir / "bweight.dat")

        cov_file = data_dir / f"c_matrix_plik_v{version}.dat"
        f = FortranFile(cov_file, "r")
        self.covmat = f.read_reals(dtype=float).reshape(
            (self.N_bin_high_ell, self.N_bin_high_ell)
        )
        for i in range(self.N_bin_high_ell):
            for j in range(self.N_bin_high_ell):
                self.covmat[i, j] = self.covmat[j, i]

        if self.spectra == "TT":
            # For the special case of just TT, remove the TE and EE covariances.
            self.covmat = self.covmat[
                0 : self.N_bin_TT_high_ell, self.N_bin_TT_high_ell
            ]

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
        self.lmin_TT_data = 2 if self.include_low_ell_TT else 30
        self.lmin_TE_EE_data = 30

        self.N_bin_TT = self.N_bin_TT_low_ell + self.N_bin_TT_high_ell
        self.N_bin_high_ell = self.N_bin_TT_high_ell + self.N_bin_TE + self.N_bin_EE
        self.N_bin_total = self.N_bin_TT + self.N_bin_TE + self.N_bin_EE

        self.N_multipoles = 2507 if self.spectra == "TT" else 3 * 2507


if __name__ == "__main__":
    PlanckLogLike(year=2015, spectra="TT", include_low_ell_TT=False)
