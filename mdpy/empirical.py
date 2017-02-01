"""
Code for simulating MDPs and analyzing the results.
"""
import numpy as np


def calculate_return(rewards, gamma):
    """Calculate return from a list of rewards and a list of gammas.

    Notes
    -----
    The discount parameter `gamma` should be the discount for the *next* state,
    if you are using general value functions.
    This is because (in the episodic setting) the terminal state has a discount
    factor of zero, but the state preceding it has a normal discount factor,
    as does the state following.

    So as we compute G_{t} = R_{t+1} + γ_{t+1}*G_{t+1}
    """
    ret = []
    g = 0
    # Allow for gamma to be specified as a sequence or a constant
    if not hasattr(gamma, '__iter__'):
        gamma = itertools.repeat(gamma)
    # Work backwards through the
    for r, gm in reversed(list(zip(rewards, gamma))):
        g *= gm
        g += r
        ret.append(g)
    # inverse of reverse
    ret.reverse()
    return np.array(ret)

def calculate_squared_return(rewards, gammas, returns):
    """Calculate squared return from a list of rewards, a list of gammas,
    and a list of returns.

    Notes
    -----
    The discount parameter `gamma` should be the discount for the *next* state,
    if you are using general value functions.
    """
    ret = []
    G_sq = 0
    G_next = 0
    for rwd, gm, G in reversed(list(zip(rewards, gammas, returns))):
        G_sq *= gm**2
        G_sq += rwd**2 + 2*gm*rwd*G_next
        ret.append(G_sq)
        G_next = G
    ret.reverse()
    return ret
