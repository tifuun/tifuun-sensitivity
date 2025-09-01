"""Sensitivity calculator for DESHIMA-type spectrometers.

FAQ
-----
Q.  Where is the point-source coupling phase and amplitude loss
    due to the mismatch between the
    beam in radiation and reception, that Shahab calculates
    at the lens surface?

A.  It is included in the main beam efficiency.
    These losses reduce the coupling to a point source,
    but the power (in transmission) couples to the sky.

"""

# flake8: noqa
__version__ = "0.1.4"


# modules
from . import atmosphere
from . import instruments
from . import physics
from . import simulator
