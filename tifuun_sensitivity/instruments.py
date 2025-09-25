# standard library
from typing import List, Union, Tuple


# dependent packages
import numpy as np
from .physics import c, e, h,johnson_nyquist_psd 


# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]


# constants
Delta_Al = 188.0 * 10**-6 * e  # gap energy of Al
eta_pb = 0.4  # Pair breaking efficiency
eta_Al_ohmic_850 = 0.9975  # Ohmic loss of an Al surface at 850 GHz.
# Shitov+, ISSTT2008. https://www.nrao.edu/meetings/isstt/papers/2008/2008263266.pdf


# main functions
def D2HPBW(F: ArrayLike) -> ArrayLike:
    """Get half-power beam width of DESHIMA 2.0 at given frequency (frequencies).

    Parameters
    ----------
    F
        Frequency. Units: Hz.

    Returns
    -------
    hpbw
        Half-power beam width. Units: radian.

    """
    return 29.0 * 240.0 / F * np.pi / 180.0 / 60.0 / 60.0


def eta_ruze(F: ArrayLike, 
             sigma: float) -> ArrayLike:
    """Get main-beam efficiency by Ruze's equation.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    sigma
        Surface error. Units: m.

    Returns
    -------
    eta_ruze
        Ruze efficiency. Units: None.

    """
    return np.exp(-((4.0 * np.pi * sigma * F / c) ** 2.0))


def photon_NEP2_kid(F: ArrayLike, 
                   psd_KID: ArrayLike) -> ArrayLike:
    """NEP squared of the KID, with respect to the absorbed power.
    Note that the square of the NEP is returned, as this quantity is more convenient for the calculation.
    Also, we do not multiply by dF here, as we do that further downstream outside of the integral.

    Parameters
    -----------
    F
        Frequency of the signal responsible for loading. Units: Hz.
    psd_KID
        Power spectral density entering MKID the KID. Units: W / Hz.

    Returns
    -------
    NEP2_kid
        Noise-equivalent power squared of the KID.
    """
    poisson_term = 2 * psd_KID * h * F
    bunching_term = 2 * psd_KID**2
    r_term = 4 * Delta_Al * psd_KID / eta_pb
    return poisson_term + bunching_term + r_term

def window_trans(
    F: ArrayLike,
    thickness: ArrayLike,
    tandelta: float,
    neff: float,
    AR: bool,
    T_parasitic_refl: float,
    T_parasitic_refr: float
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the transmission/reflection from dielectric elements, such as lenses and windows.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    thickness
        Thickness of the lens/window. Units: m.
    tandelta
        Loss tangent of lens/window dielectric.
    neff
        Refractive index of lens/window dielectric. Units : None.
    AR
        Whether the lens/window is supposed to be coated by Anti-Reflective coating (True) or not (False).
    T_parasitic_refl
        Temperature of parasitic source seen in reflection, w.r.t. instrument.
    T_parasitic_refr
        Temperature of parasitic source seen in refraction..

    Returns
    -------
    List containing list of arrays of efficiencies as first element, and list of arrays of psd's seen by each stage.
    """

    eta = []
    psd = []
    
    refl = ((1 - neff) / (1 + neff)) ** 2 * np.ones(F.size)
    psd_refl = johnson_nyquist_psd(F, T_parasitic_refl)
    psd_refr = johnson_nyquist_psd(F, T_parasitic_refr)

    if AR == False:
        eta.append(1 - refl)
        psd.append(psd_refl)
    
    alpha = (2 * np.pi * F / c) * neff * tandelta
    eta_dielectric = np.exp(-alpha * thickness)

    eta.append(eta_dielectric)
    psd.append(psd_refr)

    if AR == False:
        eta.append(1 - refl)
        psd.append(psd_refl)

    return eta, psd

def average_over_filterbank(array_to_average: ArrayLike,
                            filterbank: ArrayLike) -> ArrayLike:
    """Averages a quantity defined on F_sky over as filterbank.

    Parameters
    ----------
    array_to_average
        Array defined over F_sky to average
    filterbank
        2D array containing filterbank

    Returns
    -------
    Averaged array.
    """
    sh_f = filterbank.shape
    assert array_to_average.size == sh_f[0]

    array_tiled = np.tile(array_to_average, (sh_f[-1], 1)) * filterbank.T
    return np.nansum(array_tiled, axis=1) / np.nansum(filterbank, axis=0)
