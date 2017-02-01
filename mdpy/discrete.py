"""Implementing a categorical discrete random variable along the lines of
`scipy.rv_discrete`, which required a bit more effort and finesse than would
generally be considered preferable.
"""
import numpy as np
import scipy.linalg
import scipy.stats
import mdpy
from numbers import Number


class categorical(scipy.stats._distn_infrastructure.rv_sample):
    """
    A class created to express floating point categorical random variables in
    terms of `scipy` random variables.

    Note
    ----
    For reasons explored below, I am unsure if I've overridden everything that
    should be overridden.
    The `pmf`, `cdf`, `mean`,`var`, and `entropy` methods appear to work.

    Rant
    ----
    The (ab)use of metaprogramming in `scipy.stats` made this substantially more
    difficult than it had to be.

    If you're ever in the mood for a terrible time, read [the source code](https://github.com/scipy/scipy/blob/v0.18.1/scipy/stats/_distn_infrastructure.py)
    for the `rv_discrete` class.
    It appears that they have refactored `rv_sample` *into* `rv_discrete`, so
    that `rv_discrete` actually determines at class instantiation whether it
    will actually be `rv_discrete` or a `rv_sample`, which is a subclass of
    `rv_discrete`,
    """
    def __init__(self, outcomes, weights=None, **kwargs):
        if isinstance(outcomes, Number):
            outcomes = np.array([outcomes])
        if weights is None:
            weights = [1 for i in outcomes]

        # Assign probabilities according to weights
        probs = weights/np.sum(weights)

        # TODO: Check validity
        self.weights  = np.array(weights)
        self.outcomes = np.array(outcomes)
        self.probs    = np.array(probs)
        self.events   = list(zip(outcomes, probs))

        # Set the 'hidden' random variable mapping states to outcomes and hope
        # that scipy doesn't implement things that fail with non-integers
        self.__privaterv = scipy.stats.rv_discrete(values=(np.arange(len(probs)), probs))()
        super(categorical, self).__init__(values=(outcomes, probs), name='categorical', inc=0.5)

    def rvs(self, *args, **kwargs):
        """Overriding to provide support for non-integer, uh, support."""
        choices = self.__privaterv.rvs(*args, **kwargs)
        return self.outcomes[choices]

    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).
        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k


        Notes
        -----
        Overriding because `scipy.stats.rv_discrete` only supports integer
        values for some stupid reason.

        Changed:
        ```
        --- cond1 = (k >= self.a) & (k <= self.b) & self._nonzero(k, *args)
        +++ cond1 = (k >= self.a) & (k <= self.b)

        --- goodargs = argsreduce(cond, *((k,)+args))
        +++ goodargs = (k,) + args
        ```
        + some misc. comments

        Note that `self._nonzero(k, *args)` is exactly equivalent to `floor(k) == k`,
        so not only do you have higher overhead from function calls, but it's not even
        shorter! And it's only used in TWO places.
        Also, a lot of the internal functions appear to exist just to alias (and therefore
        slow down) existing `numpy` functions.
        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(scipy.asarray, (k, loc))
        args = tuple(map(scipy.asarray, args))
        k = scipy.asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k <= self.b)
        cond = cond0 & cond1
        output = scipy.zeros(scipy.shape(cond), 'd')
        scipy.place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            # I have no idea what `goodargs` does and I just don't care anymore
            goodargs = (k,) + args
            scipy.place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output


def drv(*args, **kwargs):
    """A function that creates and returns a `categorical` random variable, but
    is shorter to type.
    """
    return categorical(*args, **kwargs)
