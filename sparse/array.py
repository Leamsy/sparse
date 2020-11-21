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
import random

import sparse

from sparse.functions import (
    cartesian_product,
    groupby,
    min_max
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

    def get_indexes(self, args):
        """
        Get generator of indexes numpy.array and fill args

        Arguments:
        ----------
        \targs {np.array|list} -- Indices

        Return:
        \tGenerator object
        \tArguments args with slice filled
        """
        self.__check_indexes__(args)
        args = self.__fill_slice(args)
        args = np.array(args)
        lindex = [
            # Slice case
            np.where(
                (self.T[:,i] >= a.start) &
                (self.T[:,i] < a.stop) &
                (np.mod(self.T[:,i] - a.start, a.step) == 0)
            )[0] if isinstance(a, slice) else
            # List case
            np.where(
                np.isin(self.T[:,i], a)
            )[0] if isinstance(a, list) or isinstance(a, np.ndarray) else
            # Int case
            np.where(
                self.T[:,i] == a
            )[0] for i,a in enumerate(args)
        ]
        indexes = reduce(np.intersect1d, lindex)
        return indexes, args

    def to_numpy(self) -> np.array:
        """
        Casting from sparse.array to numpy.array

        Examples:
        ---------
        
        """
        X = np.zeros(shape = self.shape) + self.fill_value
        for row in self.T:
            X[tuple(row[:-1].astype(int))] = row[-1]
        return X

    

    # Bracket operator

    def __getitem__(self, args):
        """
        Operator overload brackets.

        Arguments:
        ----------
        \targs {np.array|list} -- Indices

        Returns:
        --------
        \tsparse.array|dtype -- Submatrix or one value

        Examples::
            >>> import sparse
            >>> M = sparse.zeros((20,20))
            >>> print(M[2:10,:15].shape)
            array([8,15])
            >>> print(M[[1,2,3],[1,2,3]].shape)
            array([3,3])
            >>> print(M[5,8])
            0
        """
        indexes, args = self.get_indexes(args)
        mask = np.array([isinstance(a, slice) or isinstance(a,list) or isinstance(a,np.ndarray) for a in args])
        # Caso de retornar un solo valor
        if np.count_nonzero(mask) == 0:
            if indexes.shape[0] == 0:
                return self.fill_value
            else:
                return self.T[indexes[0],-1]
        # Generador de tuplas de indices
        generator = cartesian_product(args)
        # Calcular máximo y mínimo de los índices
        minimum, maximum = min_max(generator, len(args))
        # Generar pasos para slice
        step = np.array([
            a.step if isinstance(a, slice) else
            0 if isinstance(a, list) else 1
        for a in args])
        # Calcular shape para listas y slices
        shape = np.array([
            len(args[i]) if isinstance(args[i], list) or isinstance(args[i], np.ndarray) else
            np.ceil((maximum[i] - minimum[i] + 1) / step[i])
        for i in range(len(args)) if mask[i]]).astype(np.int64)
        # Generar objeto de la clase
        M = array(
            shape = shape
        )
        if indexes.shape[0] != 0:
            M.T = self.T[:,np.where(np.append(mask,True))[0]][indexes]
            mask = np.array([i for i,a in enumerate(args) if isinstance(a, slice)])
            if mask.shape[0] != 0:
                M.T[:,mask] = np.floor(
                    (M.T[:,mask] - minimum[mask]) / step[mask]
                ).astype(np.int64)
            mask = np.array([i for i,a in enumerate(args) if isinstance(a, list) or isinstance(a, np.ndarray)])
            for i in mask:
                change = {}
                for k,j in enumerate(args[i]):
                    change[j] = k
                for k,j in enumerate(M.T[:,i]):
                    M.T[k,i] = change[M.T[k,i]]
            M.T = M.T.astype(np.int64)
        return M

    def __setitem__(self, args, value):
        raise NotImplementedError

    # Arithmetic operators

    def __add__(self, obj):
        raise NotImplementedError

    def __sub__(self, obj):
        raise NotImplementedError

    def __mul__(self, obj):
        raise NotImplementedError

    # Logical operators

    def __lt__(self, obj):
        raise NotImplementedError

    def __le__(self, obj):
        raise NotImplementedError

    def __eq__(self, obj):
        raise NotImplementedError

    def __ne__(self, obj):
        raise NotImplementedError

    def __gt__(self, obj):
        raise NotImplementedError

    def __ge__(self, obj):
        raise NotImplementedError

    # Str operator

    def __str__(self):
        string = '<sparse: shape={shape}>, dtype={dtype}, n0v={n0v}, fill_value={fill_value}'.format(
            shape = self.shape,
            dtype = self.dtype,
            n0v = self.T.shape[0],
            fill_value = self.fill_value
        )
        return string

    # Check arguments

    def __check_indexes__(self, args):
        """
        Check dimensions of incoming indexes.
        
        Arguments:
        ----------
        \targs {list of {slice|list|int}} -- Indexes ranges
        """
        assert len(args) == len(self.shape), 'Error: number of indexes {input} != {obj}.'.format(
            input = len(args),
            obj = len(self.shape)
        )
        for i in range(len(args)):
            if isinstance(args[i], slice):
                assert (args[i].start == None) or (args[i].start >= 0), 'Start slice {i} out of range.'.format(
                    i = i
                )
                assert (args[i].stop == None) or (args[i].stop < self.shape[i]), 'Stop slice {i} out of range.'.format(
                    i = i
                )
                assert (args[i].step == None) or (args[i].step < self.shape[i]), 'Step slice {i} out of range.'.format(
                    i = i
                )
            if isinstance(args[i], list):
                assert all([v in range(self.shape[i]) for v in args[i]]), 'List values {i} out of range.'.format(
                    i = i
                )
            if isinstance(args[i], int):
                assert args[i] in range(self.shape[i]), 'Value {i} out of range.'.format(
                    i = i
                )

    def __fill_slice(self, args):
        """
        Complete values of default slice

        Arguments:
        ----------
        \targs {list of {slice|list|int}} -- Indexes ranges

        Return:
        -------
        \tlist -- List obj
        """
        args = [slice(
            arg.start if arg.start != None else 0,
            arg.stop if arg.stop != None else self.shape[i],
            arg.step if arg.step != None else 1,
        ) if isinstance(arg,slice) else arg for i,arg in enumerate(args)]
        return args

    # Numpy functions

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

    def min(self, axis = None):
        raise NotImplementedError

    def max(self, axis = None):
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
    # 
    index = np.array(random.sample(range(np.array(shape).prod()), n))
    values = np.random.randint(low, high, n)
    M.T = np.zeros((n,len(shape) + 1))
    for i in range(len(shape)):
        M.T[:,len(shape) - i - 1] = np.mod(index, shape[len(shape) - i - 1])
        index = np.floor_divide(index, shape[len(shape) - i - 1])
    M.T[:,-1] = values
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