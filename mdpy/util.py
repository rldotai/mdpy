"""
Utility functions for working with MDPs
"""
import numpy as np
from functools import wraps


def as_array(func):
    """Wrap a function so that its first argument is converted to an array."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        first = np.array(args[0])
        rest = args[1:]
        return func(first, *rest, **kwargs)
    return wrapper