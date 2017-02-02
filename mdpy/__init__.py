"""mdpy - Markov decision processes in Python"""

__version__ = '0.1.0'
__author__ = 'rldotai <>'

# importing modules
from . import discrete
from . import empirical
from . import linalg
from . import solve
from . import util

# setting up package namespace
from .discrete import categorical, drv
from .mdp import MarkovProcess
from .linalg import *
from .solve import *
from .util import from_dict

# define `__all__` for this package
__all__ = []

# don't export modules unless they're actually wanted
import inspect

_whitelist = []
__all__ = [name for name, x in locals().items() if not name.startswith('_') and
           (not inspect.ismodule(x) or x in _whitelist)]

__all__.append(__version__)
