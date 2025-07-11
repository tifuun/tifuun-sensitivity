import numpy as np
from typing import List, Union, Dict, Tuple

ArrayLike = Union[np.ndarray, List[float], List[int], float, int]

def Lorentzian(gamma: float, 
               F_sky: ArrayLike, 
               F: ArrayLike) -> ArrayLike:
    """Generate Lorentzians.

    Parameters
    ----------
    gamma
        FWHM of Lorentzian. Units: GHz.
    F_sky
        Array with sky frequencies. Units: GHz.
    F
        Array with center frequencies of Lorentzians.
    
    Returns
    ----------
    2D array with Lorentzians.
    """

    return (gamma**2 / ((F_sky.T - F)**2 + gamma**2))

def generateFilterbankFromR(R: Union[float, int], 
                            F: ArrayLike, 
                            thres: float = 0.0001) -> Tuple[ArrayLike, ArrayLike, float]:
    """Generate a filterbank containing pure Lorentzians.
    
    Parameters
    ----------
    R
        Resolving power of filterbank. Units: none.
    F
        Array with filter frequencies. Units: GHz.
    thres
        Threshold value to use in converge of F_sku
    
    Returns
    ----------
    2D array with filter efficiencies, 1D array with sky frequencies, and stepsize of sky frequencies.. 
    """

    gamma = F / (2 * R)

    mu_low = -gamma[0] * (thres**(-1) - 1)**(1/2) + F[0]
    mu_upp = gamma[-1] * (thres**(-1) - 1)**(1/2) + F[-1]

    area = 0.1
    nF_sky = 750

    area0 = 1e99
    area = 0

    if F.size > 1:
        while np.absolute(area0 - area) > thres:
            nF_sky *= 2
            F_sky = np.tile(np.linspace(mu_low, mu_upp, nF_sky), (2, 1))
            dF_sky = np.mean(np.diff(F_sky, axis=-1))
            filters = Lorentzian(gamma[0::F.size-1], F_sky, F[0::F.size-1])
            area0 = area
            area = np.sum(filters) * dF_sky
    else:
        while np.absolute(area0 - area) > thres:
            nF_sky *= 2
            F_sky = np.linspace(mu_low, mu_upp, nF_sky)
            dF_sky = np.mean(np.diff(F_sky))
            filters = Lorentzian(gamma[0], F_sky, F[0])
            area0 = area
            area = np.sum(filters) * dF_sky

    F_sky = np.tile(np.linspace(mu_low, mu_upp, nF_sky), (F.size, 1))

    filterbank = 4 / np.pi * Lorentzian(gamma, F_sky, F)
    
    #if cutoff := instrumentDict.get("cutoff") is not None:
    filterbank[F_sky[0,:] < 90,:] = 0
    
    return filterbank, F_sky[0,:], np.mean(np.diff(F_sky[0,:]))
