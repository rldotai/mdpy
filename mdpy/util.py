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

def from_dict(dct, shape, default=None):
    """Make an array from a dictionary whose keys are tuples corresponding
    to a single element's index in the array.

    Examples
    --------
    ```
    >>> dct = {(0, 0): 3, (0, 1): 2, (1, 1): 1}
    >>> from_dct(dct, (2, 2), default=4)
    # array([[3, 2], [4, 1]])
    ```
    """
    def func(key):
        return dct.get(key, default)
    return np.reshape([func(ix) for ix in np.ndindex(*shape)], shape)
