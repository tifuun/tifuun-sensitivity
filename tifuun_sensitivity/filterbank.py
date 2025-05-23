import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union, Dict

ArrayLike = Union[np.ndarray, List[float], List[int], float, int]

def Lorentzian(gamma, F_sky, F, order):
    return (gamma**2 / ((F_sky.T - F)**2 + gamma**2))**order

def generateFilterbankFromR(R: Union[float, int], 
                            F: ArrayLike, 
                            order: int = 1, 
                            thres: float = 0.0001, 
                            plot: bool = False) -> ArrayLike:
    """
    Generate a Lorentzian filterbank matrix from resolving power R.

    @param instrumentDict Instrument dictionary.

    @returns filterbank The Lorentzian filterbank.
    """

    gamma = F / (2 * R)

    mu_low = -gamma[0] * (thres**(-1/order) - 1)**(1/2) + F[0]
    mu_upp = gamma[-1] * (thres**(-1/order) - 1)**(1/2) + F[-1]

    area = 0.1
    nF_sky = 750

    area0 = 1e99
    area = 0

    if F.size > 1:
        while np.absolute(area0 - area) > thres:
            nF_sky *= 2
            F_sky = np.tile(np.linspace(mu_low, mu_upp, nF_sky), (2, 1))
            dF_sky = np.mean(np.diff(F_sky, axis=-1))
            filters = Lorentzian(gamma[0::F.size-1], F_sky, F[0::F.size-1], order)
            area0 = area
            area = np.sum(filters) * dF_sky
    else:
        while np.absolute(area0 - area) > thres:
            nF_sky *= 2
            F_sky = np.linspace(mu_low, mu_upp, nF_sky)
            dF_sky = np.mean(np.diff(F_sky))
            filters = Lorentzian(gamma[0], F_sky, F[0], order)
            area0 = area
            area = np.sum(filters) * dF_sky

    F_sky = np.tile(np.linspace(mu_low, mu_upp, nF_sky), (F.size, 1))

    filterbank = 4 / np.pi * Lorentzian(gamma, F_sky, F, order)

    return filterbank, F_sky[0,:], np.mean(np.diff(F_sky[0,:]))
