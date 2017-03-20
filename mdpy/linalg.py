import networkx as nx
import numpy as np
import scipy.stats
from functools import reduce
from math import gcd
from numpy.linalg import det, pinv, matrix_rank, norm

from .util import as_array


# Matrix/vector manipulation
def cols(mat):
    assert(is_matrix(mat))
    return [x for x in mat.T]

def colsum(mat):
    assert(is_matrix(mat))
    return np.einsum('ij->j', mat)

def rows(mat):
    assert(is_matrix(mat))
    return [x for x in mat]

def rowsum(mat):
    assert(is_matrix(mat))
    return np.einsum('ij->i', mat)

def someones(n, *ones):
    """Generate a vector of zeros except for ones at the given indices."""
    # TODO: a better name for this!
    ret = np.zeros(n)
    ret[[ones]] = 1
    return ret

def somezeros(n, *zeros):
    """Generate a vector of ones except at particular indices."""
    # TODO: get a better name for this!
    # TODO: fix issue where *zeros is empty
    ret = np.ones(n)
    ret[[zeros]] = 0
    return ret

def unit(ndim, ix, dtype=np.float):
    """Generate a unit vector, with the entry at `ix` set to 1."""
    ret = np.zeros(ndim, dtype=dtype)
    ret[ix] = 1
    return np.atleast_1d(ret)

@as_array
def normalize(array, axis=None):
    """Normalize an array along an axis."""
    def _normalize(vec):
        return vec/np.sum(vec)
    if axis:
        return np.apply_along_axis(_normalize, axis, array)
    else:
        return _normalize(array)

@as_array
def as_stochastic(mat):
    mat = np.squeeze(mat)
    if not is_square(mat):
        raise Exception("Stochastic matrices must be square.")
    if not is_nonnegative(mat):
        raise Exception("Stochastic matrices must have all nonnegative entries")
    return normalize(mat, axis=1)


def random_binary(num_states, num_features, num_active):
    """Create a matrix of random binary features, with `num_states` rows and
    `num_features` columns, each row having `num_active` entries equal to one
    and the rest equal to zero.
    """
    ixs = np.arange(num_features)
    ret = []
    for i in range(num_states):
        active = np.random.choice(ixs, num_active, replace=False)
        fvec = np.zeros(num_features)
        fvec[active] = 1
        ret.append(fvec)
    return np.array(ret)


# Vector properties
@as_array
def is_pvec(pvec, tol=1e-6):
    """Check if a vector represents a probability distribution."""
    vec = np.ravel(pvec)
    return (np.size(vec) == np.size(pvec)) \
    and (np.all(vec >= 0)) \
    and (1-tol <= np.sum(vec) <= 1+tol)


# Matrix properties

@as_array
def has_absorbing(mat):
    """Check if the transition matrix has absorbing states.

    A state is absorbing if its only outgoing transition is to itself.
    """
    return len(find_terminals(mat)) > 0

@as_array
def is_aperiodic(mat):
    """Check if a stochastic matrix is aperiodic."""
    graph = nx.DiGraph(mat)
    return nx.is_aperiodic(graph)

@as_array
def is_diagonal(mat):
    """Check if a matrix is diagonal."""
    if not is_square(mat):
        return False
    else:
        off_diagonals = np.extract(1 - np.eye(len(mat)), mat)
        return np.all(0 == off_diagonals)

@as_array
def is_ergodic(mat):
    """Check if the matrix is ergodic (irreducible and aperiodic)."""
    return not(is_reducible(mat) or is_periodic(mat))

@as_array
def is_matrix(mat):
    """Test that an array is a matrix."""
    return mat.ndim == 2

@as_array
def is_nonnegative(mat):
    """Check if a matrix is nonnegative."""
    return np.all(mat >= 0)

@as_array
def is_periodic(mat):
    """Check if the transition matrix is periodic.

    A matrix is periodic if it has a period greater than `2`, that is, if it
    """
    graph = nx.DiGraph(mat)
    return not(nx.is_aperiodic(graph))

@as_array
def is_distribution(vec):
    """Check if a vector is a probability distribution"""
    return np.all(vec >= 0) and np.isclose(1, np.sum(vec))

@as_array
def is_reducible(mat):
    """Check if the matrix is reducible. That is, if all states are part of the
    same communicating class (can be reached from each other).

    """
    assert(is_nonnegative(mat))
    #TODO: Find a better method for this
    P = np.copy(mat)
    S = np.zeros_like(mat)
    for i in range(len(mat)):
        S += P
        P = np.dot(P, mat)
    return np.any(np.isclose(0, S))

@as_array
def is_square(mat):
    """Ensure that an array is a 2-D square matrix."""
    return (mat.ndim == 2) and (mat.shape[0] == mat.shape[1])

@as_array
def is_stochastic(mat, tol=1e-6):
    """Check if a matrix is (right) stochastic."""
    return (mat.ndim == 2) \
    and (mat.shape[0] == mat.shape[1]) \
    and (np.all([row >= 0 for row in mat])) \
    and (all(1-tol <= np.sum(row) <= 1+tol for row in mat))

@as_array
def is_substochastic(mat, tol=1e-6):
    """Check if a matrix is (right) substochastic."""
    return (mat.ndim == 2) \
    and (mat.shape[0] == mat.shape[1]) \
    and (np.all([row >= 0 for row in mat])) \

## Information about MDPs utilities

@as_array
def find_terminals(mat):
    """Find terminal states in a transition matrix."""

    return [ix for ix, row in enumerate(mat) if row[ix] == 1]

@as_array
def find_nonterminals(mat):
    """Find nonterminal states in a transition matrix."""
    return [ix for ix, row in enumerate(mat) if row[ix] != 1]

@as_array
def get_period(mat):
    """Find the period of the stochastic matrix `mat`.

    Uses `networkx` to find the cycles in the graph and computes the GCD over
    their lengths.

    Notes
    -----
    The period is defined as the GCD of all possible return times to a state,
    or the integer that divides the length of every cycle in the transition
    graph.
    """
    import networkx as nx
    graph = nx.DiGraph(mat)
    return reduce(gcd, [len(x) for x in nx.simple_cycles(graph)])


@as_array
def approx_stationary(mat, s0=None, tol=1e-6, iterlimit=100000):
    """Compute the approximate stationary distribution of a stochastic matrix,
    by repeatedly multiplying a probability vector by the matrix until it
    converges.
    """
    assert(is_stochastic(mat,tol))
    if s0 is None:
        s0 = np.ones(len(mat))
        s0 = s0/len(mat)
    if not is_distribution(s0):
        raise Exception("Failed to converge to ")
    assert(is_distribution(s0))

    # Approximate the stationary distribution by repeated transitions
    s = np.copy(s0)
    for i in range(iterlimit):
        # TODO: is normalization here correct?
        sp = np.dot(s, mat)
        sp = sp/np.sum(np.abs(sp))
        if np.allclose(s, sp):
            return s#/np.sum(np.abs(s))
        s = sp
    else:
        raise Exception("Failed to converge within tolerance:", s, s0)

@as_array
def stationary(mat):
    """Compute the stationary distribution for transition matrix `mat`, via
    computing the solution to the system of equations (P.T - I)*\pi = 0.

    NB: Assumes `mat` is ergodic (aperiodic and irreducible).
    Could do with LU factorization -- c.f. 54-14 in Handbook of Linear Algebra
    """
    assert(is_stochastic(mat))

    P = (np.copy(mat).T - np.eye(len(mat)))
    P[-1,:] = 1
    b = np.zeros(len(mat))
    b[-1] = 1
    x = np.linalg.solve(P, b)
    return normalize(x)

@as_array
def stationary_matrix(mat):
    """Return the matrix `A` where each row is the stationary distribution for
    the given matrix `mat`.

    Computes the stationary distribution for transition matrix `mat`, via
    computing the solution to the system of equations (P.T - I)*\pi = 0.

    NB: Assumes `mat` is ergodic (aperiodic and irreducible).
    Could do with LU factorization -- c.f. 54-14 in Handbook of Linear Algebra
    """
    assert(is_stochastic(mat))
    ns = len(mat)
    P = (np.copy(mat).T - np.eye(ns))
    P[-1,:] = 1
    b = np.zeros(ns)
    b[-1] = 1
    x = np.linalg.solve(P, b)
    d_pi = normalize(x)
    return np.tile(d_pi, (ns, 1))


@as_array
def get_all_stationary(mat):
    """Compute /all/ stationary states for transition matrix `mat`, by
    finding the left eigenvectors with an associated eigenvalue of `1`.

    NB: Has a lot of transposing going on in order to accomodate numpy.
    NB: Uses `np.isclose` for checking whether eigenvalues are 1.
    NB: Tries to ensure it returns real-valued left eigenvectors.
    """
    assert(is_stochastic(mat))
    P = np.copy(mat).T
    vals, vecs = np.linalg.eig(P)
    states = [v/np.sum(v) for e, v in zip(vals, vecs.T) if np.isclose(e,1)]
    return [np.real_if_close(v) for v in states]

@as_array
def distribution_matrix(mat):
    """Compute the stationary distribution for a matrix, and return the
    diagonal matrix with the stationary distribution along its diagonal."""
    return np.diag(stationary(mat))

@as_array
def matrix_series(mat, n):
    """Compute the matrix series for `n` terms using matrix `mat`."""
    assert(is_square(mat))
    _I  = np.eye(len(mat))
    ret = np.copy(_I)
    for i in range(n):
        ret += mat @ ret
    return ret

@as_array
def matrix_power(mat, n):
    """Compute the matrix `mat` raised to the power `n`."""
    assert(is_square(mat))
    ret = np.eye(len(mat))
    for i in range(n):
        ret = ret @ mat
    return ret

