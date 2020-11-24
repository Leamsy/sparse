# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Granada, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------

from itertools import product
import numpy as np

def cartesian_product(X: list):
    """
    Cartesian product applied to different type of objects:
    1. slices
    2. list
    3. int

    Arguments:
    ----------
    \tX {list} -- List of slices,int or list

    Returns:
    --------
    \titertools.product -- Generator object with shape (len(X),prod([len(x) for x in X]))
    
    Examples:
    ---------
        >>> import sparse
        >>> generator = sparse.functions.cartesian_product([3,[1,2],slice(2,4,1)])
        >>> print([g for g in generator])
        >>> [(3, 1, 2), (3, 1, 3), (3, 2, 2), (3, 2, 3)]
    """
    generator = product(*[
        range(x.start,x.stop,x.step) if isinstance(x,slice) else 
        x if isinstance(x,range) else
        x if isinstance(x,list) else
        [x] for x in X
    ])
    return (generator)

def groupby(X: np.array, by: int or list, func, output = None):
    """
    groupby(X: np.array, by: int or list, func, output = None)

    Parameters
    ----------
    X: numpy array
        Input data structure
    by: int or list of ints
        Columns to group data
    func: function(X: np.array, axis = 0)
        Function applied
    output: int or list of ints
        Columns  
    Return
    ------

    Example
    -------
    ... hola

    """
    by = by if isinstance(by,list) else [by]
    if output == None:
        output = [i for i in range(X.shape[1]) if i not in by]
    else:
        output = output if isinstance(output,list) else [output]
    _, index, inverse = np.unique(X[:,by], axis = 0, return_index = True , return_inverse = True)
    Y = np.zeros((index.shape[0],len(by) + len(output)))
    for i,j in enumerate(index):
        Y[i] = np.append(X[j,by],func(X[inverse == i][:,output], axis = 0))
    return Y

def min_max(tuples, size):
    minimum = np.full(size, np.inf)
    maximum = np.zeros(size)

    for t in tuples:
        minimum = np.minimum(minimum,np.array(t))
        maximum = np.maximum(maximum,np.array(t))
    
    return minimum, maximum

def minus_row_index(X: np.array, Y: np.array):
    return np.where([False if any([np.all(x == y) for y in Y]) else True for x in X])

def nindex_to_oneindex(nindex: np.array, shape: np.array):
    I = np.zeros(shape = nindex.shape[0])
    for i in range(shape.shape[0]):
        I += nindex[:,i] * shape[:i].prod()
    return I.astype(np.int64)

def oneindex_to_nindex(oneindex: np.array, shape: np.array):
    I = np.zeros(shape = (oneindex.shape[0],shape.shape[0]), dtype = np.int64)
    iacc = oneindex
    for i in range(shape.shape[0]-1,-1,-1):
        I[:,i] = np.floor_divide(iacc,shape[:i].prod())
        iacc = np.mod(iacc,shape[:i].prod())
    return I.astype(np.int64)

def max_oneindex(shape: np.array):
    return shape.prod()

def inverse_filter_coordinates(X,args):
    Y = np.zeros((X.shape[0],len(args)))
    for i,a in enumerate(args):
        if isinstance(a,list) or isinstance(a,np.ndarray):
            Y[:,i] = np.array([a[x] for x in X[:,i]])
        if isinstance(a,slice):
            Y[:,i] = (X[:,i] + a.start) * a.step
        if isinstance(a,int):
            Y[:,i] = a
    return Y.astype(np.int64)

def filter_coordinates(X,args):
    args = np.array(args)
    f = lambda x: isinstance(x,list) or isinstance(x,np.ndarray) or isinstance(x,slice)
    mask = [i for i,a in enumerate(args) if f(a)]
    Y = np.zeros((X.shape[0],len(mask)))
    for i,a in enumerate(args[mask]):
        if isinstance(a,list) or isinstance(a,np.ndarray):
            if len(a) != len(set(a)): Warning('Error transform')
            A = dict(zip(a,np.arange(len(a))))
            Y[:,i] = np.array([A[x] for x in X[:,i]])
        if isinstance(a,slice):
            Y[:,i] = (X[:,i] - a.start) / a.step
    return Y.astype(np.int64)

    