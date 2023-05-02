# Torch Planck 2018 Lite

`torchplite` is a PyTorch implementation of the Planck 2018 Lite likelihood for the Cosmic Microwave Background (CMB) power spectra. This package provides a convenient and efficient way to compute the log-likelihood of CMB power spectra given a cosmological model.

## Installation

You can install `torchplite` via pip from PyPI:

```
pip install torchplite
```

This package requires Python 3.6 or later and PyTorch 1.7 or later.

## Usage

To use the `torchplite` package, you can import the `PlanckLitePy` class and create an instance with the desired settings:

```python
from torchplite import PlanckLitePy

# Initialize the PlanckLitePy object
planck = PlanckLitePy(year=2018, spectra="TTTEEE", use_low_ell_bins=False)

# Load the power spectra
import numpy as np
ls, Dltt, Dlte, Dlee = np.genfromtxt("path/to/your/data/Dl_planck2015fit.dat", unpack=True)

# Compute the log-likelihood
ellmin = int(ls[0])
loglikelihood = planck.loglike(Dltt, Dlte, Dlee, ellmin)
```

You can customize the behavior of the `PlanckLitePy` object by changing its constructor parameters:

- `year`: The Planck data release year (2015 or 2018).
- `spectra`: The CMB power spectra to use ("TTTEEE" for TT, TE, and EE or "TT" for TT only).
- `use_low_ell_bins`: Whether to include low-ell bins in the likelihood calculation (True or False).


## Running Tests 

To run the tests, you can use the `unittest` module: 

```
python -m unittest discover tests
```

this will run all the test cases defined in the `tests` directory. 

## Licesnse 

This project is under the MIT License. See the LICENSE file for more details.

## Credit

This is a PyTorch implementation of the `planck-lite-py` code by [Heather Prince](https://github.com/heatherprince/planck-lite-py). 