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


# ETD
def etd_weights(P, G, L, X, ivec, r):
    """The weight vector found by ETD."""
    # compute intermediate quantities (could be more efficient)
    assert(linalg.is_stochastic(P))
    I = np.eye(len(P))
    di = linalg.stationary(P) * ivec
    m = pinv(I - L @ G @ P.T) @ (I - G @ P.T) @ di
    M = np.diag(m)

    # solve the equation
    A = X.T @ M @ pinv(I - P @ G @ L) @ (I - P @ G) @ X
    A_inv = np.linalg.pinv(A)
    b = X.T @ M @ pinv(I - P @ G @ L) @ r
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

##############################################################################
# The following may throw errors, have yet to test w/ rest of repo
##############################################################################
# TODO: Document these functions in more detail
# TODO: add tests
# TODO: de-duplicate/normalize with rest of repo style
def mc_return(P, r, Γ):
    """Compute the Monte-Carlo return given transition matrix `P`, expected 
    reward `r`, and state-discount matrix `Γ`."""
    assert(linalg.is_stochastic(P))
    I = np.eye(len(P))
    return np.linalg.pinv(I - P @ Γ) @ r

def ls_weights(P, r, Γ, X):
    """Compute the least-squares weights for transition matrix `P`, expected
    reward `r`, and state-discount matrix `Γ`, and feature matrix `X`."""
    assert(linalg.is_stochastic(P))
    assert(X.ndim == 2)
    assert(len(X) == len(P))
    value = mc_return(P, r, Γ)
    dist  = linalg.stationary(P)
    D     = np.diag(dist)
    return np.linalg.pinv(X.T @ D @ X) @ X.T @ D @ value

def ls_values(P, r, Γ, X):
    """Compute the state-values under least-squares function approximation for 
    transition matrix `P`, expected reward vector `r`, discount matrix `Γ`, 
    and feature matrix `X`."""
    weights = ls_weights(P, r, Γ, X)
    return X @ weights

def td_weights(P, r, Γ, Λ, X):
    """Compute the weights found at the TD fixed point under linear function 
    approximation given transition matrix `P`, expected reward vector `r`, 
    discount matrix `Γ`, bootstrapping matrix `Λ, and feature matrix `X`."""
    assert(linalg.is_stochastic(P))
    assert(X.ndim == 2)
    assert(len(X) == len(P))
    assert(linalg.is_diagonal(Γ))
    assert(linalg.is_diagonal(Λ))
    I    = np.eye(len(P))
    dist = linalg.stationary(P)
    D    = np.diag(dist)
    r_lm = (I - P @ Γ @ Λ) @ r
    P_lm = I - pinv(I - P @ Γ @ Λ) @ (I - P @ Γ)
    A = X.T @ D @ (I - P_lm) @ X
    b = X.T @ D @ r_lm
    return np.linalg.pinv(A) @ b

def td_values(P, r, Γ, Λ, X):
    """Compute state values found at the TD fixed point under linear function
    approximation given transition matrix `P`, expected reward vector `r`, 
    discount matrix `Γ`, bootstrapping matrix `Λ, and feature matrix `X`."""
    return X @ td_weights(P, r, Γ, Λ, X)
    
def lambda_return(P, r, Γ, Λ, v_hat):
    """Compute the λ-return given transition matrix `P`, expected reward vector
    `r`, discount matrix `Γ`, bootstrapping matrix `Λ,  and approximate 
    state-values `v_hat`."""
    assert(linalg.is_stochastic(P))
    assert(linalg.is_diagonal(Γ))
    assert(linalg.is_diagonal(Λ))
    I = np.eye(len(P))
    # Incorporate next-state's value into expected reward
    r_hat = r + P @ Γ @ (I - Λ) @ v_hat
    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Λ) @ r_hat

def sobel_variance(P, R, Γ):
    """Compute the variance of the return using Sobel's method, given 
    transition matrix `P`, expected transition reward matrix `R`, and 
    state-dependent discount matrix `Γ`."""
    assert(linalg.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(linalg.is_diagonal(Γ))
    ns = len(P)
    I = np.eye(len(P))
    r = (P * R) @ np.ones(ns)
    v_pi = mc_return(P, r, Γ)
    
    # Set up Bellman equation
    q = -v_pi**2
    for i in range(ns):
        for j in range(ns):
            q[i] += P[i,j]*(R[i,j] + Γ[j,j]*v_pi[j])**2
    # Solve Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ) @ q

def second_moment(P, R, Γ):
    """Compute the second moment of the return using the method from White and
    White, given transition matrix `P`, expected transition reward matrix `R`, 
    and state-dependent discount matrix `Γ`."""
    assert(linalg.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(linalg.is_diagonal(Γ))
    ns = len(P)
    I = np.eye(len(P))
    # Compute expected state values
    r = (P*R) @ np.ones(ns)
    v_pi = mc_return(P, r, Γ)
    γ = np.diag(Γ)
    
    # Compute reward-like transition matrix
    R_bar = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            R_bar[i,j] = R[i,j]**2 + 2*( γ[j] * R[i,j] * v_pi[j])
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)
    
    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ) @ r_bar


def lambda_second_moment(P, R, Γ, Λ, v_hat):
    """Compute the second moment of the λ-return using the method from White &
    White, given transition matrix `P`, expected transition reward matrix `R`, 
    state-dependent discount matrix `Γ`, state-dependent bootstrapping matrix 
    `Λ`, and approximate values `v_hat`."""
    assert(linalg.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(linalg.is_diagonal(Γ))
    assert(linalg.is_diagonal(Λ))
    ns = len(P)
    I = np.eye(len(P))
    # Expected immediate reward
    r = (P * R) @ np.ones(ns)
    # Lambda return may be different from approximate lambda return
    v_lm = lambda_return(P, r, Γ, Λ, v_hat)
    
    # Get per-state discount and bootstrapping
    γ = np.diag(Γ)
    λ = np.diag(Λ)
    
    # Compute reward-like transition matrix
    R_bar = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            R_bar[i,j] = R[i,j]**2 \
                + (γ[j] * (1-λ[j])*v_lm[j])**2 \
                + 2*( γ[j] * (1 - λ[j]) * R[i,j] * v_hat[j] ) \
                + 2*( γ[j] * λ[j] * R[i,j] * v_lm[j]) \
                + 2*( (γ[j]**2)*λ[j]*(1-λ[j]) * (v_hat[j]*v_lm[j]) )
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)
    
    # Solve the Bellman equation
    return pinv(I - P @ Γ @ Γ @ Λ @ Λ) @ r_bar