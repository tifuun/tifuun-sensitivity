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
    return 29.0 * 240.0 / (F / 1e9) * np.pi / 180.0 / 60.0 / 60.0


def eta_mb_ruze(F: ArrayLike, LFlimit: float, sigma: float) -> ArrayLike:
    """Get main-beam efficiency by Ruze's equation.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    LFlimit
        Main-beam efficiency at 0 Hz.
    sigma
        Surface error. Units: m.

    Returns
    -------
    eta_mb
        Main-beam efficiency. Units: None.

    """
    return LFlimit * np.exp(-((4.0 * np.pi * sigma * F / c) ** 2.0))


def photon_NEP_kid(F: ArrayLike, 
                   psd_KID: ArrayLike) -> ArrayLike:
    """NEP of the KID, with respect to the absorbed power.

    Parameters
    -----------
    F
        Frequency of the signal responsible for loading. Units: Hz.
    psd_kid
        Power spectral density entering MKID the KID. Units: W / Hz.
    filterbank
        Detection bandwidth, with respect to the power that sets the loading. Units: Hz.

    Returns
    -------
    NEP_kid
        Noise-equivalent power of the KID.

    Notes
    -----
    Pkid/(W_F * h * F) gives the occupation number.

    """
    # photon_term = 2 * Pkid * (h*F + Pkid/W_F)
    poisson_term = 2 * psd_KID * h * F
    bunching_term = 2 * psd_KID**2
    r_term = 4 * Delta_Al * psd_KID / eta_pb
    return poisson_term + bunching_term + r_term


def window_trans(
    F: ArrayLike,
    thickness: ArrayLike,
    tandelta: float,
    neffHDPE: float,
    window_AR: bool,
    T_parasitic_refl: float,
    T_parasitic_refr: float
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the window transmission.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    thickness
        Thickness of the HDPE window. Units: m.
    tandelta
        Loss tangent of window/lens dielectric.
    neffHDPE
        Refractive index of HDPE. Set to 1 to remove reflections. Units : None.
    window_AR
        Whether the window is supposed to be coated by Ar (True) or not (False).


    Returns
    -------
    psd_after_2nd_refl
        PSD looking into the window from the cold optics.
    eta_window
        Transmission of the window. Units: None.

    """
    # Parameters to calcualte the window (HDPE), data from Stephen
    # reflection. ((1-neffHDPE)/(1+neffHDPE))^2. Set to 0 for Ar coated.

    eta = []
    psd = []
    
    HDPErefl = ((1 - neffHDPE) / (1 + neffHDPE)) ** 2 * np.ones(F.size)
    psd_refl = johnson_nyquist_psd(F, T_parasitic_refl)
    psd_refr = johnson_nyquist_psd(F, T_parasitic_refr)

    if window_AR == False:
        eta.append(1 - HDPErefl)
        psd.append(psd_refl)

    eta_HDPE = np.exp(
        -thickness
        * 2
        * np.pi
        * neffHDPE
        * (tandelta * F / c + (tandelta * F / c) ** 2)
    )

    eta.append(eta_HDPE)
    psd.append(psd_refr)

    if window_AR == False:
        eta.append(1 - HDPErefl)
        psd.append(psd_refl)

    return eta, psd

def average_over_filterbank(array_to_average: ArrayLike,
                            filterbank: ArrayLike) -> ArrayLike:
    sh_f = filterbank.shape
    assert array_to_average.size == sh_f[0]

    array_tiled = np.tile(array_to_average, (sh_f[-1], 1)) * filterbank.T
    return np.nansum(array_tiled, axis=1) / np.nansum(filterbank, axis=0)
