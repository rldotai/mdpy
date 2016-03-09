"""Implementation of a Discrete MDP class."""
import numpy as np

from .linalg import unit


class DiscreteMDP:
    """Discrete MDP class."""
    def __init__(self, ptensor):
        ptensor = np.array(ptensor)
        assert(ptensor.ndim == 3)
        ns, na, nsp = ptensor.shape
        assert(ns == nsp)
        self.ns = ns
        self.na = na
        # check that it's valid

    @classmethod
    def random(cls, ns, na, rvar=None):
        """ Generate a random transition probability matrix for the given
        numbers of states `ns` and actions `na`.

        Ensure that each state has some probability of transitioning to a
        different state.
        Consider allowing user to specify random variable to draw from.
        Consider adding code to ensure that the matrix is properly ergodic.
        Consider having an adjustable sparsity parameter.
        """
        ret = np.zeros((ns, na, ns), dtype=np.float)
        for s, a in np.ndindex(ns, na):
            ret[s, a] = np.random.random(ns)
            ret[s, a] = ret[s, a]/np.sum(ret[s, a]) # normalize
        return cls(ret)


    def transition(self, s, a):
        """ Transition to a new state according to the probability matrix `pmat`,
        given action `a` was taken in state `s`.
        """
        ns, na, _ = pmat.shape
        # Allow specifying states/actions as vectors or integer indices
        if isinstance(s, int):
            s = unit(ns, s)
        if isinstance(a, int):
            a = unit(na, a)

        # Select and return the choice
        prob = np.dot(a, np.dot(s, pmat))
        choice = np.random.choice(np.arange(ns), p=prob)
        return unit(ns, choice)