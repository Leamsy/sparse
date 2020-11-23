# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Granada, Spain
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
    min_max,
    nindex_to_oneindex,
    oneindex_to_nindex,
    max_oneindex,
    inverse_filter_coordinates,
    filter_coordinates
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
            X[tuple(row[:-1].astype(np.int64))] = row[-1]
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
            shape = shape,
            dtype = self.dtype,
            fill_value = self.fill_value
        )
        # Algún valor distinto de fill_value
        if indexes.shape[0] != 0:
            M.T = self.T[:,np.where(np.append(mask,True))[0]][indexes]
            # Completar slices
            mask = np.array([i for i,a in enumerate(args) if isinstance(a, slice)])
            if mask.shape[0] != 0:
                M.T[:,mask] = np.floor(
                    (M.T[:,mask] - minimum[mask]) / step[mask]
                )
            # Completar list
            mask = np.array([i for i,a in enumerate(args) if isinstance(a, list) or isinstance(a, np.ndarray)])
            for i in mask:
                change = {}
                for k,j in enumerate(args[i]):
                    change[j] = k
                for k,j in enumerate(M.T[:,i]):
                    M.T[k,i] = change[M.T[k,i]]
        return M

    def __setitem__(self, args, value):
        # Índices sobre self.T
        indexes, args = self.get_indexes(args)
        # Generador de tuplas de indices
        self_gen = cartesian_product(args)
        # Si value es int
        if isinstance(value,int):
            # Índices nuevos
            T = np.array([np.append(index,value) for index in self_gen])
            if T.shape[0] != 0: 
                self.T = np.vstack([self.T,T])
                func = lambda x,axis: value if value in x else x[0,0]
                self.T = groupby(
                    X = self.T, 
                    by = [i for i in range(len(self.shape))], 
                    func = func,
                    output = [len(self.shape)]
                )
            else: self.T = np.vstack([[np.append(args,value)]])
        
        # Si value es sparse.array
        if isinstance(value,array):
            self.T = np.delete(self.T, indexes, 0)
            if self.fill_value == value.fill_value:
                self.T = np.vstack([
                    self.T,
                    np.hstack([
                        inverse_filter_coordinates(value.T[:,:-1],args),
                        value.T[:,-1].reshape(-1,1)
                    ])
                ])
            else:
                # Indices de dimensión n a unidimensional
                oneindex = nindex_to_oneindex(value.T[:,:-1],value.shape)
                # Optimizar búsqueda logarítmica
                opt = set(oneindex)
                # fill_value a indices densos
                oneindex_fill_value = np.array(
                    [i for i in range(max_oneindex(value.shape)) if i not in opt
                ])
                # Indices densos contraidos con fill_value
                B = np.hstack([
                    oneindex_to_nindex(oneindex_fill_value,value.shape),
                    np.repeat(value.fill_value,oneindex_fill_value.shape[0]).reshape(-1,1)
                ])
                # Unión de todos los conjuntos
                self.T = np.vstack([
                    self.T,
                    np.hstack([
                        inverse_filter_coordinates(value.T[:,:-1],args),
                        value.T[:,-1].reshape(-1,1)
                    ]),
                    np.hstack([
                        inverse_filter_coordinates(B[:,:-1],args),
                        B[:,-1].reshape(-1,1)
                    ])
                ])


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
        string = '<sparse: shape={shape}>, dtype={dtype}, n0v={n0v}, fill_value={fill_value}>'.format(
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