# standard library
from typing import List, Union


# dependent packages
import numpy as np


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# constants
h = 6.62607004 * 10**-34  # Planck constant
k = 1.38064852 * 10**-23  # Boltzmann constant
e = 1.60217662 * 10**-19  # electron charge
c = 299792458.0  # velocity of light


# main functions
def rad_trans(rad_in: ArrayLike, 
              medium: ArrayLike, 
              eta: ArrayLike) -> ArrayLike:
    """Calculates radiation transfer through a semi-transparent medium.

    One can also use the same function for Johnson-Nyquist PSD
    (power spectral density) instead of temperature.

    Parameters
    ----------
    rad_in
        Brightness temperature (or PSD) of the input. Units: K (or W/Hz).
    medium
        Brightness temperature (or PSD) of the lossy medium. Units: K (or W/Hz).
    eta
        Transmission of the lossy medium. Units: K (or W/Hz).

    Returns
    -------
    Brightness temperature (or PSD) of the output.

    """

    if len(eta.shape) > 1:
        rad_in = rad_in[:,None]
        medium = medium[:,None]
    return eta * rad_in + (1 - eta) * medium


def T_from_psd(F: ArrayLike, 
               psd: ArrayLike, 
               method: str = "Planck") -> ArrayLike:
    """Calculate Planck temperature from the PSD frequency (frequencies).

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    psd
        Power Spectral Density. Units: W / Hz.
    method
        Default: 'Planck'. Option: 'Rayleigh-Jeans'.

    Returns
    --------
    Planck temperature. Units: K.

    """
    if method == "Planck":
        return h * F / (k * np.log(h * F / psd + 1.0))
    elif method == "Rayleigh-Jeans":
        return psd / k
    else:
        raise ValueError("Method should be Planck or Rayleigh-Jeans.")


def johnson_nyquist_psd(F: ArrayLike, T: ArrayLike) -> ArrayLike:
    """Johnson-Nyquist power spectral density.

    Don't forget to multiply with bandwidth to caculate the total power in W.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    T
        Temperature. Units: K.

    Returns
    --------
    Power Spectral Density. Units: W / Hz.

    """
    return h * F * nph(F, T)


# helper functions
def nph(F: ArrayLike, T: ArrayLike) -> ArrayLike:
    """Photon occupation number of Bose-Einstein Statistics.

    If it is not single temperature, use nph = Pkid / (W_F * h * F).

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    T
        Temperature. Units: K.

    Returns
    --------
    Photon occupation number. Units: None.

    """
    return 1.0 / np.expm1(h * F / (k * T))
