"""
Plugin overview
===============
"""
from .ops import (S, T, CCNOT, CPHASE, CSWAP, ISWAP, PSWAP)
from .qpu import QPUDevice
from .qvm import QVMDevice
from .wavefunction import WavefunctionDevice
from ._version import __version__
