# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Granada, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------

import numpy as np
import random
import sparse

def randint(low: int, high: int, sparsity: float, shape: tuple or list, fill_value = 0.0) -> sparse.array:
    """
    Generate random sparse matrix between and interval.

    Arguments:
    ----------
    \tlow {int} -- Lower level
    \thigh {int} -- Higher level
    \tsparsity {float} -- Density of the matrix
    \tshape {tuple|list} -- Output shape

    Keyword Arguments:
    ------------------
    \tfill value {int} -- Default value (default: {0.0})

    Returns:
    --------
    \tsparse.array -- Array object

    Examples:
    ---------
        >>> import sparse
    """
    assert(fill_value not in range(low,high)), 'Error: Uniform distribution not valid {a} in ({b},{c})'.format(
        a = fill_value,
        b = low,
        c = high
    )
    M = sparse.array(shape = shape, fill_value = fill_value)
    # Number of values different from fill_value
    n = int(np.floor(np.array(shape).prod() * sparsity))
    # Random indexes
    index = np.array(random.sample(range(np.array(shape).prod()), n))
    values = np.random.randint(low, high, n)
    M.T = np.zeros((n,len(shape) + 1))
    # Application mod
    for i in range(len(shape)):
        M.T[:,len(shape) - i - 1] = np.mod(index, shape[len(shape) - i - 1])
        index = np.floor_divide(index, shape[len(shape) - i - 1])
    # Paste values in last column
    M.T[:,-1] = values
    return M