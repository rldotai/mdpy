import numpy as np
from functools import reduce
from numbers import Number
from numpy.linalg import det, pinv, matrix_rank, norm

from . import linalg
from .util import as_array


# td_key
# td_A
# td_b
# td_D
# td_e

# etd_key
# etd_A
# etd_b
# etd_m
# etd_f
# etd_i

# common

@as_array
def propagator(A):
    """ A |--> (I - A)

    NB: Somewhat iffy terminology`
    """
    assert(linalg.is_square(A))
    I = np.eye(len(P))
    return (I - A)

@as_array
def potential(A, tol=1e-6):
    """Compute the potential matrix
    ret = (I - M1*M2*...)^{-1}
    """
    assert(linalg.is_square(A))
    assert(isinstance(tol, Number))
    I = np.eye(len(A))
    ret = np.linalg.inv(I - A)
    ret[np.abs(ret) < tol] = 0 # zero values within tolerance
    return ret

def bellman(P,G,r):
    """Compute the solution to the Bellman equation."""
    assert(linalg.is_stochastic(P))
    assert(linalg.is_diagonal(G))
    I = np.eye(len(P))
    return np.dot(np.linalg.inv(I - np.dot(G,P)), r)


def least_squares(P, G, X, r):
    """Compute the optimal weights via least squares."""
    assert(linalg.is_stochastic(P))
    v = bellman(P, G, r)
    D = np.diag(linalg.stationary(P))
    return np.linalg.pinv(X.T @ D @ X) @ X.T @ D @ v


def warp(P, G, L):
    """
    The matrix which warps the distribution due to gamma and lambda.
    warp = (I - P_{\pi} \Gamma \Lambda)^{-1}
    NB: "warp matrix" is non-standard terminology.

    P : The transition matrix (under a policy)
    G : Diagonal matrix, diag([gamma(s_1), ...])
    L : Diagonal matrix, diag([lambda(s_1), ...])
    """
    assert(linalg.is_stochastic(P))
    assert(linalg.is_diagonal(G))
    assert(linalg.is_diagonal(L))
    return potential(P @ G @ L)

# TD

def td_weights(P, G, L, X, r):
    """The weights found by TD(lambda)."""
    assert(linalg.is_stochastic(P))
    I = np.eye(len(P))
    D = np.diag(linalg.stationary(P))
    A = X.T @ D @ (I - P @ G) @ X
    A_inv = np.linalg.pinv(A)
    b = X.T @ D @ r

    return np.dot(A_inv, b)

def td_solution(P, G, L, X, r):
    """The state values found by TD(lambda)

    TODO: General TD(lambda)
    """
    theta = td_weights(P, G, L, X, r)
    return X @ theta

# ETD
def etd_weights(P, G, L, X, ivec, r):
    """The weight vector found by ETD."""
    # compute intermediate quantities (could be more efficient)
    assert(linalg.is_stochastic(P))
    I = np.eye(len(P))
    di = linalg.stationary(P) * ivec
    m = potential(L @ G @ P.T) @ potential(G @ P.T) @ di
    M = np.diag(m)

    # solve the equation
    A = X.T @ M @ potential(P @ G @ L) @ (I - P @ G) @ X
    A_inv = np.linalg.pinv(A)
    b = X.T @ M @ potential(P @ G @ L) @ r
    return np.dot(A_inv, b)


def etd_solution(P, G, L, X, ivec, r):
    """The solution found by ETD(lambda)"""
    # compute intermediate quantities (could be more efficient)
    theta = etd_weights(P, G, L, X, ivec, r)
    return np.dot(X, theta)

def followon(P, G, di):
    """Compute the followon trace's expected value for each state."""
    assert(linalg.is_stochastic(P))
    assert(linalg.is_diagonal(G))
    I = np.eye(len(P))

    return np.dot(np.linalg.inv(I - G @ P.T), di)