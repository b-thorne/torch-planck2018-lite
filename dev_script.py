from src.main2 import PlanckLogLike
import numpy as np 
import matplotlib.pyplot as plt

loglike = PlanckLogLike(year=2015, include_low_ell_TT=True, spectra="TTTEEE")

ls_, dltt, dlte, dlee = np.genfromtxt("data/Dl_planck2015fit.dat", unpack=True)
fac = ls_ * (ls_ + 1) / 2 / np.pi

cltt = dltt / fac
clte = dlte / fac 
clee = dlee / fac

cl = np.concatenate((cltt, clte, clee))

binned_cl = loglike.W @ cl
bin_num_TT = np.arange(loglike.N_bin_TT)
bin_num_TE = np.arange(loglike.N_bin_TE)

ntt, nte = loglike.N_bin_TT, loglike.N_bin_TE


fig, ax = plt.subplots(1, 1)
ax.loglog(bin_num_TT, binned_cl[:ntt])
ax.loglog(bin_num_TT, loglike.C_hat_ell_b[:ntt])
plt.show()

fig, ax = plt.subplots(1, 1)
ax.loglog(np.arange(nte), binned_cl[ntt:ntt+nte])
ax.loglog(np.arange(nte), loglike.C_hat_ell_b[ntt:ntt+nte])
plt.show()

fig, ax = plt.subplots(1, 1)
ax.loglog(np.arange(nte), binned_cl[ntt + nte:ntt + nte + nte])
ax.loglog(np.arange(nte), loglike.C_hat_ell_b[ntt + nte: ntt + nte + nte])
plt.show()