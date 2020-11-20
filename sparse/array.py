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

from functions import (
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
        __init__(self, shape: list or tuple, dtype = np.float64, fill_value = 0.0)
        
        Parameters
        ----------
        shape: list or tuple of ints
            Shape of the new array, e.g., `(2, 3)` or `[2,3]`.
        dtype: data-type
            The desired data-type for the array, e.g., `numpy.int8`. 
            By default: np.float64
        fill_value: initialize value with same dtype.
            By default: 0.0
        
        Examples::
            >>> import SparseMatrix.SparseMatrix as sm
            >>> M1 = sm([20,20], np.float64, 3)
            >>> M2 = sm([20,20,20], np.float64)
            >>> M3 = sm([20,20,20], fill_value = 3.5)
        """
        self.shape = np.array(shape)
        self.dtype = dtype
        self.fill_value = fill_value
        self.T = np.zeros(
            shape = (0,self.shape.shape[0] + 1), 
            dtype = self.dtype
        )

    def __getitem__(self, args: np.array or list):
        """
        __getitem__(self, args: np.array or list)

        Parameters
        ----------
        args: np.array or
            Multi 

        """
        pass

def from_numpy(array: np.array, fill_value = 0.0):
    """
    from_numpy(cls, array: np.array, fill_value = 0.0)

    Casting from numpy.array to sparse.array.

    Arguments:
        array {numpy.array} -- A numpy.array object

    Keyword Arguments:
        fill_value {integer} -- Default value (default: {0.0})
    
    Returns:
        [sparse.array] -- Array object
    """
    M = array(
        shape = array.shape,
        dtype = array.dtype,
        fill_value = fill_value
    )
    M.T = np.vstack((
        # Indexes of the matrix
        np.where(array != fill_value),
        # Values
        array[array > fill_value]
    )).T
    return M

def randint(low: int, high: int, sparsity: float, shape: tuple or list, fill_value = 0.0):
    """
    randint(low: int, high: int, sparsity: float, shape: tuple or list, fill_value = 0.0)

    Generate random sparse matrix.

    Arguments:
        low {integer} -- Lower level
        high {integer} -- Higher level
        sparsity {float} -- 1 - density of the matrix
        shape {tuple|list} -- Output shape

    Keyword Arguments:
        fill_value {integer} -- Default value (default: {0.0})

    Raises:
        TypeError -- [description]

    Returns:
        [sparse.array] -- Array object
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