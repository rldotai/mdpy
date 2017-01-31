"""mdpy - Markov decision processes in Python"""

__version__ = '0.0.2'
__author__ = 'rldotai <>'

# define your imports here
from . import linalg
from . import solve
from . import util
from . import discrete

from .linalg import *
from .solve import *
from .discrete import categorical, drv

# define `__all__` for this package
__all__ = []

# don't export modules unless they're actually wanted
import inspect

_whitelist = []
__all__ = [name for name, x in locals().items() if not name.startswith('_') and
           (not inspect.ismodule(x) or x in _whitelist)]

__all__.append(__version__)
