"""Implementation of a Discrete MDP class."""
import numpy as np


class ArrayMDP:
    """MDP class, formulated in terms of multi-arrays.
    It requires an two arrays, one for the transition probabilities (`T`) and
    another of the same shape for the expected rewards (`R`).

    For example, given state `s`, action `a`, and next state `sp`, then the
    probability of the transition `(s, a, sp)` is `T[s,a,sp]` and the expected
    reward for undergoing that transition is `R[s,a,sp]`.
    """
    def __init__(self, transitions, rewards):
        """Create an ArrayMDP from the supplied transition and reward arrays."""
        T = np.array(transitions)
        R = np.array(rewards)
        # Check that shapes are valid
        assert(3 == T.ndim == R.ndim)
        assert(T.shape == R.shape)
        # Check that probabilities for `sp` given `s` and `a` sum to 1
        assert(np.allclose(1, np.einsum('ijk->ij', T))

        # Initialize the MDP
        self.T = T
        self.R = R

    def under_policy(self, policy):
        """Produce the Markov process that results from acting according to the
        given policy in the MDP."""
        pass

    def prob(self, s, a=None, sp=None):
        """Get the probability of supplied transition, or of the possible
        transitions conditioned accordingly.
        """
        pass
