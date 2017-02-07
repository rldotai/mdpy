"""Implementation of a Discrete MDP class."""
import numpy as np
import scipy.linalg
import scipy.stats
import mdpy
from numbers import Number


class MarkovProcess:
    """A class implementing Markov processes, which are like MDPs where you
    don't make any decisions.
    It requires two arrays, one for the transition probabilities (`T`) and
    another of the same shape for the expected rewards (`R`).

    For example, given state `s` and next state `sp`, the probability of the
    transition `(s, sp)` is `T[s, sp]`, with reward `R[s, sp]`.
    """
    def __init__(self, transitions, rewards):
        T = np.array(transitions)
        R = np.array(rewards)
        # Check that shapes are valid
        assert(2 == T.ndim == R.ndim)
        assert(T.shape == R.shape)
        assert(T.shape[0] == T.shape[1])
        # Check that transition probabilities sum to one
        assert(np.allclose(1, np.einsum('ij->i', T)))

        # Initialize variables
        self.T = T
        self.R = R
        self._states = np.arange(len(T))

    @classmethod
    def from_unnormalized(cls, transitions, rewards=None):
        """Create a Markov Process using an arbitrary transition matrix by
        taking the absolute value and normalizing the transition probabilities.
        """
        pass

    @property
    def states(self):
        return np.arange(len(self.T))

    def prob(self, s, sp=None):
        """Return the probability of the transition, or if `sp` is not given,
        instead return the probability of every transition from `s`.
        """
        return np.squeeze(self.T[s, sp])

    def transition(self, s):
        return np.random.choice(self._states, p=self.T[s])

    def step(self, s):
        """Transition from a state to its successor, returning `(sp, r)`."""
        sp = np.random.choice(self._states, p=self.T[s])
        r  = self.reward(s, sp)
        return (sp, r)

    def reward(self, s, sp):
        """Sample a reward from the transition `(s, sp)`."""
        r = self.R[s, sp]
        if isinstance(r, Number):
            return r
        elif isinstance(r, scipy.stats._distn_infrastructure.rv_frozen):
            return r.rvs()
        elif isinstance(r, scipy.stats._distn_infrastructure.rv_generic):
            return r.rvs()
        elif callable(r):
            return r(s, sp)
        else:
            raise TypeError("Reward for transition not understood: (%d, %d)"%(s, sp))

    def expected_reward(self, s, sp=None):
        """Compute the expected reward either given a state or a transition."""
        if sp is not None:
            return self._expectation(self.R[s, sp])
        else:
            return np.array([self._expectation(r) for r in self.R[s]])

    def _expectation(self, rwd):
            """Get the expected value of a reward."""
            if isinstance(rwd, Number):
                return rwd
            elif isinstance(rwd, scipy.stats._distn_infrastructure.rv_frozen):
                return rwd.mean()
            elif isinstance(rwd, scipy.stats._distn_infrastructure.rv_generic):
                return rwd.mean()
            else:
                raise TypeError("Unable to get expected value of reward: %s"%(rwd))

    def run(self, n, s0=None):
        """Run the Markov process for `n` steps, return a list of transitions.

        The result has the form:

            `[{'s': s, 'sp': sp, 'r': r}, ...]`

        So for `ret[t]`, 's' is the state at time `t`, 'r' is the reward, and
        'sp' is the next state.
        """
        if s0 is None:
            s0 = np.random.choice(len(self.states))

        # Set up and run the simulation
        ret = []
        s = s0
        for t in range(n):
            sp, r = self.step(s)
            ret.append({'s': s, 'sp': sp, 'r': r})

            # Set up for next iteration
            s = sp
        return ret


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
        assert(T.shape[0] == T.shape[2])
        # Check that probabilities for `sp` given `s` and `a` sum to 1
        assert(np.allclose(1, np.einsum('ijk->ij', T)))

        # TODO: Check ergodic (aperiodic and irreducible)

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

        TODO:
            Should I make this compatible with Baye's rule (so given `s`, `sp`)
            we can get the probability that each action was selected?
            Or will this be a confusing API?

            It would be as easy as:

            ```
            if a is None and sp is not None:
                return ret/np.sum(ret)
            ```
        """
        ret = np.squeeze(self.T[s, a, sp])
        return ret
