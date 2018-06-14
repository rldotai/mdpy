import numpy as np
from functools import reduce
from numbers import Number
from numpy.linalg import det, pinv, matrix_rank, norm
from . import linalg
from .util import as_array, as_diag


# TODO: Change assertions to exceptions
# TODO: Add tests
# TODO: Convert to new matrix multiplication operator

# TODO: Allow specifying Γ as a vector, constant, or maybe even dict?
def mc_return(P, r, Γ):
    """Compute the expected Monte-Carlo return for the Markov chain defined by
    `P` with expected reward `r`.
    This is the result of solving the Bellman equation.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    """
    assert linalg.is_stochastic(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    I = np.eye(ns)
    return np.linalg.pinv(I - P @ Γ) @ r


# TODO: Allow specifying Γ as a vector, constant, or maybe even dict?
def ls_weights(P, r, Γ, X):
    """Compute the least-squares weights for the MDP given feature matrix `X`.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    """
    assert linalg.is_stochastic(P)
    assert X.ndim == 2
    assert len(X) == len(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    value = mc_return(P, r, Γ)
    dist = linalg.stationary(P)
    D = np.diag(dist)
    return np.linalg.pinv(X.T @ D @ X) @ X.T @ D @ value


# TODO: Allow specifying Γ as a vector, constant, or maybe even dict?
def ls_values(P, r, Γ, X):
    """Compute the state-values under least-squares function approximation.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    """
    ns = len(P)
    Γ = as_diag(Γ, ns)
    weights = ls_weights(P, r, Γ, X)
    return X @ weights


# TODO: Allow specifying Γ, Λ, as a vector, constant, or maybe even dict?
def td_weights(P, r, Γ, Λ, X):
    """Compute the weights found at the TD fixed point for the MDP under
    linear function approximation.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.

    Notes
    -----
    If the feature matrix `X` is of the same rank as `P`, then the result should
    be the same as computing the exact value function.
    If `Λ = diag([1, 1, ..., 1])`, then the result should be the same as
    computing the weights under least-squares.
    """
    assert linalg.is_stochastic(P)
    assert X.ndim == 2
    assert len(X) == len(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)

    # Calculate intermediate quantities
    I = np.eye(ns)
    dist = linalg.stationary(P)
    D = np.diag(dist)

    # Set up and solve the equation
    r_lm = pinv(I - P @ Γ @ Λ) @ r
    P_lm = I - pinv(I - P @ Γ @ Λ) @ (I - P @ Γ)
    A = X.T @ D @ (I - P_lm) @ X
    b = X.T @ D @ r_lm
    return np.linalg.pinv(A) @ b


# TODO: Allow specifying Γ, Λ, as a vector, constant, or maybe even dict?
def td_values(P, r, Γ, Λ, X):
    """Compute state values found at the TD fixed point for the MDP under
    linear function approximation.


    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    """
    return X @ td_weights(P, r, Γ, Λ, X)


def delta_matrix(R, Γ, v):
    """Returns the matrix whose (i,j)-th entry represents the expected TD-error
    for transitioning to state `j` from state `i`.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    R : Matrix[float]
        The reward matrix, with `R[i,j]` the expected reward for transitioning
        to state `j` from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    v : The value approximate function, with `v[i]` the value assigned to state
        `i`. If `v` is the true value function, the expected TD-error will be 0.

    Returns
    -------
    Δ : Matrix[float]
        The expected TD-error matrix, with `Δ[i,j]` the expected value of
        `(R_{t+1} + γ_t+1 v(S_{t+1}) - v(S_{t})` given that state `S_{t} = i`,
        and state `S_{t+1} = j`
    """
    assert linalg.is_square(R)
    ns = len(R)
    Γ = as_diag(Γ, ns)
    ret = np.zeros((ns, ns))
    for i, j in np.ndindex(*ret.shape):
        ret[i, j] = R[i, j] + Γ[j, j] * v[j] - v[i]
    return ret


def expected_delta(P, R, Γ, v):
    """The expected TD-error given transitions `P`, reward matrix `R`,
    discount matrix `Γ`, and values `v`.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    R : Matrix[float]
        The reward matrix, with `R[i,j]` the expected reward for transitioning
        to state `j` from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    v : The value approximate function, with `v[i]` the value assigned to state
        `i`. If `v` is the true value function, the expected TD-error will be 0.

    Returns
    -------
    δ : Vector[float]
        The expected TD-error vector, with `δ[i]` the expected value of
        `(R_{t+1} + γ_t+1 v(S_{t+1}) - v(S_{t})` given that state `S_{t} = i`.
    """
    assert linalg.is_stochastic(P)
    Δ = delta_matrix(R, Γ, v)
    return (P * Δ).sum(axis=1)


def expected_reward(P, R):
    """Expected immediate reward given transition matrix `P` and
    expected reward matrix `R`.
    """
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    return np.multiply(P, R).sum(axis=1)


# TODO: Allow specifying Γ, Λ, as a vector, constant, or maybe even dict?
def lambda_return(P, r, Γ, Λ, v_hat):
    """Compute the expected λ-return for the MDP.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    r : Vector[float].
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.

    Notes
    -----
    If `v_hat` is the "true" value function (i.e., the values found by solving
    the Bellman equation) then the λ-return will be the same as the Monte-Carlo
    return (which in expectation *is* the true value function).

    The λ-return is defined via:

        G_{t}^{λ} = R_{t+1} + γ_{t+1}( (1-λ_{t+1}) v(S_{t+1}) + λ_{t+1}G_{t+1}^{λ}
    """
    assert linalg.is_stochastic(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)
    I = np.eye(ns)
    # Incorporate next-state's value into expected reward
    r_hat = r + P @ Γ @ (I - Λ) @ v_hat
    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Λ) @ r_hat


def etd_weights(P, r, Γ, Λ, X, ivec):
    """Compute the fixed-point of ETD(λ) by solving its Bellman equation.
    The weight vector returned corresponds to the asymptotic weights for found
    by Emphatic TD(λ).

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    ivec : Vector[float]
        The per-state "interest" vector.
        For example, `ivec[i]` is the interest allocated to state `i`.
    """
    assert linalg.is_stochastic(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)

    # compute intermediate quantities (could be more efficient)
    I = np.eye(ns)
    di = linalg.stationary(P) * ivec
    m = pinv(I - Λ @ Γ @ P.T) @ (I - Γ @ P.T) @ di
    M = np.diag(m)

    # solve the equation
    A = X.T @ M @ pinv(I - P @ Γ @ Λ) @ (I - P @ Γ) @ X
    A_inv = np.linalg.pinv(A)
    b = X.T @ M @ pinv(I - P @ Γ @ Λ) @ r
    return np.dot(A_inv, b)


def etd_values(P, r, Γ, Λ, X, ivec):
    """Compute the state-values found by Emphatic TD(λ) by solving the
    appropriate Bellman equation.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    r : The expected reward vector.
        Element `r[i]` is defined to be the expected reward over the
        transitions from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.
    ivec : Vector[float]
        The per-state "interest" vector.
        For example, `ivec[i]` is the interest allocated to state `i`.
    """
    # compute intermediate quantities (could be more efficient)
    theta = etd_weights(P, Γ, Λ, X, ivec, r)
    return np.dot(X, theta)


def followon(P, Γ, ivec):
    """Compute the followon trace's expected value for each state."""
    assert linalg.is_stochastic(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)

    I = np.eye(ns)
    di = linalg.stationary(P) * ivec
    return np.dot(np.linalg.pinv(I - Γ @ P.T), di)


@as_array
def potential(A, tol=1e-6):
    """Compute the potential matrix for `A`, which is the sum of the matrix
    geometric series (also referred to as the "Neumann series".

        B = \sum_{k=0}^{\infty} A^k = (I - A)^{-1}

    Parameters
    ----------
    A : Matrix[float]
        A square matrix such that `(I - A)` is invertible.
    """
    assert linalg.is_square(A)
    assert isinstance(tol, Number)
    I = np.eye(len(A))
    ret = np.linalg.inv(I - A)
    ret[np.abs(ret) < tol] = 0  # zero values within tolerance
    return ret


def warp(P, Γ, Λ):
    """
    The matrix which warps the distribution due to gamma and lambda.
    warp = (I - P_{\pi} \Gamma \Lambda)^{-1}


    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning to state `j` from state `i`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].

    Notes
    -----
    The term "warp matrix" is non-standard terminology, but is somewhat
    appropriate because it represents the asymptotic result of bootstrapping
    and discounting in the MDP.
    The i-th row-sum reflects the influence of the subsequent states on state
    `i`, while the j-th column sum reflects the influence of state `j` on its
    successors.
    """
    assert linalg.is_stochastic(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)
    return potential(P @ Γ @ Λ)

def lspi_weights(P, r, Γ, X):
    """Least-squares policy iteration fixed-point weights.

    TODO: Need to actually go through the details to make sure this is right.
    """
    assert linalg.is_stochastic(P)
    assert X.ndim == 2
    assert len(X) == len(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)

    # Calculate intermediate quantities
    I = np.eye(ns)
    dist = linalg.stationary(P)
    D = np.diag(dist)
    Π = X @ pinv(X.T @ D @ X) @ X.T @ D

    A = X.T @ ( X - P @ Γ @ Π @ X)
    b = X.T @ r
    return pinv(A) @ b

def lspi_values(P, r, Γ, X):
    """Least-squares policy iteration fixed-point values."""
    return X @ lspi_weights(P, r, Γ, X)

def brm_weights(P, r, Γ, X):
    """Bellman-residual minimization fixed-point weights.

    TODO: Need to actually go through the details to make sure this is right
    """
    assert linalg.is_stochastic(P)
    assert X.ndim == 2
    assert len(X) == len(P)
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)

    # Calculate intermediate quantities
    I = np.eye(ns)
    dist = linalg.stationary(P)
    D = np.diag(dist)
    Π = X @ pinv(X.T @ D @ X) @ X.T @ D

    A = (X - P @ Γ @ Π @ X).T @ (X - P @ Γ @ Π @ X)
    b = (X - P @ Γ @ Π @ X).T @ r
    return pinv(A) @ b

def brm_values(P, r, Γ, X):
    """Bellman-residual minimization fixed-point values.

    TODO:
        Need to go through the math to check that this is right; as it currently
        is it seems kinda terrible, which surely couldn't be the case given that
        TD(λ) has existed since the 80s?
    """
    return X @ brm_weights(P, r, Γ, X)


###############################################################################
# Variance and Second Moment
###############################################################################

# TODO: Allow specifying Γ as a vector, constant, or maybe even dict?
def sobel_variance(P, R, Γ):
    """Compute the variance of the return using Sobel's method for a Markov
    process.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning from state `i` to state `j` .
    R : Matrix[float]
        Element `R[i,j]` is defined to be the expected reward for transitioning
        from state `i` to state `j`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].

    TODO
    -----
    This function doesn't work if rewards are a function of state, action, and
    the successor state.
    It is easy to fix in a haphazard way, via summing over (s,a,s') for P and
    R, but I would prefer to handle it via something more generic like numpy's
    `einsum`.
    """
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    ns = len(P)
    Γ = as_diag(Γ, ns)

    I = np.eye(ns)
    r = (P * R) @ np.ones(ns)
    v_pi = mc_return(P, r, Γ)

    # Set up Bellman equation
    q = -v_pi ** 2
    for i in range(ns):
        for j in range(ns):
            q[i] += P[i, j] * (R[i, j] + Γ[j, j] * v_pi[j]) ** 2
    # Solve Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ) @ q


# TODO: Allow specifying Γ as a vector, constant, or maybe even dict?
def second_moment(P, R, Γ):
    """Compute the second moment of the return using the method from White and
    White for a Markov process.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning from state `i` to state `j` .
    R : Matrix[float]
        The expected reward matrix.
        Element `R[i,j]` is defined to be the expected reward for transitioning
        from state `i` to state `j`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].

    TODO
    -----
    This function doesn't work if rewards are a function of state, action, and
    the successor state.
    It is easy to fix in a haphazard way, via summing over (s,a,s') for P and
    R, but I would prefer to handle it via something more generic like numpy's
    `einsum`.
    """
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    ns = len(P)
    Γ = as_diag(Γ, ns)
    I = np.eye(ns)

    # Compute expected state values
    r = (P * R) @ np.ones(ns)
    v_pi = mc_return(P, r, Γ)
    γ = np.diag(Γ)

    # Compute reward-like transition matrix
    R_bar = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            R_bar[i, j] = R[i, j] ** 2 + 2 * (γ[j] * R[i, j] * v_pi[j])
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)

    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ) @ r_bar


# TODO: Allow specifying Γ, Λ, as a vector, constant, or maybe even dict?
def lambda_second_moment(P, R, Γ, Λ, v_hat):
    """Compute the second moment of the λ-return using the method from White &
    White for a Markov process.

    Parameters
    ----------
    P : Matrix[float]
        The transition matrix, with `P[i,j]` defined as the probability of
        transitioning from   state `i` to state `j` .
    R : Matrix[float]
        The expected reward matrix.
        Element `R[i,j]` is defined to be the expected reward for transitioning
        from state `i` to state `j`.
    Γ : Matrix[float]
        The state-dependent discount matrix, a diagonal matrix whose (i,i)-th
        entry is the discount applied to state `i`.
        All entries should be in the interval [0, 1].
    Λ : Matrix[float]
        The state-dependent bootstrapping matrix, a diagonal matrix whose
        (i,i)-th entry is the bootstrapping (λ value) for state `i`.
        All entries should be in the interval [0, 1].
    X : Matrix
        The feature matrix, whose rows correspond to the feature representation
        for each state.
        For example, `X[i]` provides the features for state `i`.

    Notes
    -----
    Because we are using the λ-return, the choice of `v_hat` influences the
    second moment.
    The λ-return is defined via:

        G_{t}^{λ} = R_{t+1} + γ_{t+1}( (1-λ_{t+1}) v(S_{t+1}) + λ_{t+1}G_{t+1}^{λ}


    TODO
    -----
    This function doesn't work if rewards are a function of state, action, and
    the successor state.
    It is easy to fix in a haphazard way, via summing over (s,a,s') for P and
    R, but I would prefer to handle it via something more generic like numpy's
    `einsum`.
    """
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Λ = as_diag(Λ, ns)

    I = np.eye(ns)
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
            R_bar[i, j] = (
                R[i, j] ** 2
                + (γ[j] * (1 - λ[j]) * v_hat[j]) ** 2
                + 2 * (γ[j] * (1 - λ[j]) * R[i, j] * v_hat[j])
                + 2 * (γ[j] * λ[j] * R[i, j] * v_lm[j])
                + 2 * ((γ[j] ** 2) * λ[j] * (1 - λ[j]) * (v_hat[j] * v_lm[j]))
            )
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)

    # Solve the Bellman equation
    return pinv(I - P @ Γ @ Γ @ Λ @ Λ) @ r_bar


###############################################################################
# Objective/Error Functions
###############################################################################


def square_error(P, R, Γ, v):
    """Square error (SE)."""
    bias = value_error(P, R, Γ, v)
    variance = sobel_variance(P, R, Γ)
    return variance + bias ** 2


def value_error(P, R, Γ, v):
    """Value error (VE)."""
    assert linalg.is_ergodic(P)
    r = (P * R).sum(axis=1)
    v_pi = mc_return(P, r, Γ)
    return v_pi - v


def bellman_error(P, R, Γ, v):
    """Bellman error (BE)."""
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Δ = delta_matrix(R, Γ, v)
    return (P * Δ) @ np.ones(ns)


def projected_bellman_error(P, R, Γ, X, v):
    """Projected Bellman error."""
    assert linalg.is_ergodic(P)
    assert P.shape == R.shape
    r = (P * R).sum(axis=1)
    d_pi = linalg.stationary(P)
    D = np.diag(d_pi)
    proj = X @ np.linalg.pinv(X.T @ D @ X) @ X.T @ D

    # Bellman operator
    Tv = r + P @ Γ @ v
    return v - proj @ Tv


def square_td_error(P, R, Γ, v):
    """Squared temporal difference error (STDE)."""
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    ns = len(P)
    Γ = as_diag(Γ, ns)
    Δ = delta_matrix(R, Γ, v)
    return (P * Δ ** 2) @ np.ones(ns)


def expected_update(P, R, Γ, X, v):
    """Expected update."""
    assert linalg.is_stochastic(P)
    assert P.shape == R.shape
    d_pi = linalg.stationary(P)
    D = np.diag(d_pi)
    δ = expected_delta(P, R, Γ, v)
    return X.T @ D @ δ

# -----------------------------------------------------------------------------
# Weighted/normed versions of the errors
# -----------------------------------------------------------------------------

def mse(P, R, Γ, v):
    """Mean-squared error (MSE)."""
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(d_pi * square_error(P, R, Γ, v))


def msve(P, R, Γ, v):
    """Mean-squared value error (MSVE)."""
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(d_pi * value_error(P, R, Γ, v) ** 2)


def msbe(P, R, Γ, v):
    """Mean squared Bellman error (MSBE)."""
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(d_pi * bellman_error(P, R, Γ, v) ** 2)


def mspbe(P, R, Γ, X, v):
    """Mean squared projected Bellman error (MSPBE)."""
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(d_pi * projected_bellman_error(P, R, Γ, X, v) ** 2)


def mstde(P, R, Γ, v):
    """Mean squared temporal difference error (MSTDE)."""
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(d_pi * square_td_error(P, R, Γ, v))


def neu(P, R, Γ, X, v):
    """Norm of the expected update (NEU).

    NEU(v) = 0 is the fixed-point of TD(0).
    """
    assert linalg.is_ergodic(P)
    d_pi = linalg.stationary(P)
    return np.mean(expected_update(P, R, Γ, X, v) ** 2)
