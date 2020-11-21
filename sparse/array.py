# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Grenade, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------

import numpy as np
from itertools import product
from functools import reduce

import sparse

from sparse.functions import (
    cartesian_product,
    groupby
)

class array:
    """
    array
    =====
    """
    def __init__(self, shape: list or tuple, dtype = np.float64, fill_value = 0.0):
        """
        Arguments:
        ----------
        \tshape {tuple|list} -- Output shape
            
        Keyword Arguments:
        ------------------
        \tdtype -- Default type (default: {numpy.float64})
        \tfill value {int} -- Default value (default: {0.0})
        
        Examples:
        ---------
        >>> import sparse
        >>> M1 = sparse.array([20,20], np.float64, 3)
        >>> M2 = sparse.array([20,20,20], np.float64)
        >>> M3 = sparse.array([20,20,20], fill_value = 3.5)
        """
        self.shape = np.array(shape)
        self.dtype = dtype
        self.fill_value = fill_value
        self.T = np.zeros(
            shape = (0,self.shape.shape[0] + 1), 
            dtype = self.dtype
        )

    def __getitem__(self, args):
        """
        Operator overload brackets.

        Arguments:
        ----------
        \targs {np.array|list} -- Indices

        Returns:
        --------
        \tsparse.array|dtype -- Submatrix or one value

        Examples:
        ---------
        >>> import sparse
        >>> M = sparse.zeros((20,20))
        >>> print(M[2:10,:15].shape)
        array([8,15])
        >>> print(M[[1,2,3],[1,2,3]].shape)
        array([3,3])
        >>> print(M[5,8])
        0
        """
        raise NotImplementedError

    def __setitem__(self, args, value):
        raise NotImplementedError

    def __getindexes__(self, args):
        raise NotImplementedError

    def to_numpy(self) -> np.array:
        """
        Casting from sparse.array to numpy.array

        Examples:
        ---------
        
        """
        raise NotImplementedError

    def dot(self, obj):
        raise NotImplementedError

    def sum(self, axis = None):
        raise NotImplementedError

    def prod(self, axis = None):
        raise NotImplementedError

    def mean(self, axis = None):
        raise NotImplementedError

    def var(self, axis = None):
        raise NotImplementedError

    def std(self, axis = None):
        raise NotImplementedError

def from_numpy(obj: np.array, fill_value = 0.0) -> array:
    """
    Casting from numpy.array to sparse.array.

    Arguments:
    ----------
    \tobj {numpy.array} -- A numpy.array object

    Keyword Arguments:
    ------------------
    \tfill value {int} -- Default value (default: {0.0})
    
    Returns:
    --------
    \tsparse.array -- Array object
    
    Examples:
    ---------
    >>> import sparse
    >>> M 
    """
    M = array(
        shape = obj.shape,
        dtype = obj.dtype,
        fill_value = fill_value
    )
    M.T = np.vstack((
        # Indexes of the matrix
        np.where(obj != fill_value),
        # Values
        obj[obj > fill_value]
    )).T
    return M

def randint(low: int, high: int, sparsity: float, shape: tuple or list, fill_value = 0.0) -> array:
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
    M = array(shape = shape, fill_value = fill_value)
    # Number of values different from fill_value
    n = int(np.floor(np.array(shape).prod() * sparsity))
    M.T = np.vstack(
        # Indexes of the matrix
        [np.random.randint(0,shape[i], size = n) for i in range(len(shape))] +
        # Values
        [np.random.randint(low,high,n)]
    ).T
    return M

def zeros(shape: list or tuple, dtype = np.float64):
    """
    Create sparse matrix with all values equal to 0.

    Arguments:
    ----------
    \tshape {tuple|list} -- Output shape
    \tdtype -- 

    """
    M = array(
        shape = shape, 
        fill_value = dtype(0)
    )
    return M

def ones(shape: list or tuple, dtype = np.float64):
    """
    Create sparse matrix with all values equal to 1.

    Arguments:
    ----------
    \tshape {tuple|list} -- Output shape
    \tdtype -- 

    """
    M = array(
        shape = shape, 
        fill_value = dtype(1)
    )
    return M