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

def as_diag(elem, n):
    """Convert `elem` to a diagonal matrix"""
    elem = np.squeeze(elem)
    length = elem.size
    ndim = elem.ndim
    if ndim == 0:
        return np.eye(n)*elem
    elif ndim == 1:
        if length == 1:
            return np.eye(n)*elem
        elif length != n:
            raise ValueError("Cannot make `elem` diagonal: need length %d, \
                got length %d"%(n, length))
        else:
            return np.diag(elem)
    elif ndim == 2:
        # Trust that it's already diagonal
        return elem
    else:
        raise ValueError("Too many dimensions to make`elem` diagonal: need \
            at most 2, got %d"%(ndim))

