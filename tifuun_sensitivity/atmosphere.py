# standard library
from pathlib import Path
from typing import Callable, List, Union


# dependent packages
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]

# main functions
def eta_atm_func(
    F: ArrayLike, 
    pwv: float, 
    EL: float) -> ArrayLike:
    """Calculate eta_atm as a function of F by interpolation.

    Note that currently, the function can only accept scalar values for the precitpitable water vapor (pwv).

    Parameters
    ----------
    F
        Frequency of the astronomical signal.
        Units: Hz.
    pwv
        Precipitable water vapour. Units: mm.
    EL
        Telescope elevation angle. Units: degrees.

    Returns
    -------
    Atmospheric tranmsmission, interpolated on F and pwv. Units: None.
    """

    eta_atm_df = pd.read_csv(
        Path(__file__).parent / "data" / "atm.csv",
        skiprows=4,
        sep='\s+',
        header=0,
    )
    
    x = eta_atm_df["F"].values * 1e9
    y = np.array(list(eta_atm_df)[1:]).astype(np.float64)
    z = eta_atm_df.iloc[:, 1:].values

    eta_atm_interp = RectBivariateSpline(x, y, z, kx=1, ky=1)(F, pwv, grid=True)
    eta_el = eta_atm_interp ** (1 / np.sin(EL * np.pi / 180))
    return eta_el

