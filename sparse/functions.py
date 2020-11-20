# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Grenade, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------

from itertools import product
import numpy as np

def cartesian_product(X: list):
    """
    cartesian_product(X: list)

    Parameters
    ----------
    X: list of slices, ranges or list

    Returns
    -------
    Numpy array object with shape (len(X),prod([len(x) for x in X]))

    Descripcion
    -----------
    Cartesian product applied to different type of objects.

    Reference
    ---------
    <https://www.math.uvic.ca/faculty/gmacgill/guide/RF.pdf>

    Examples
    --------
        >>> cartesian_product([3,[1,2],2:3,range(1,5,2)])
    """
    tuples = product(*[
        range(x.start,x.stop,x.step) if isinstance(x,slice) else 
        x if isinstance(x,range) else
        x if isinstance(x,list) else
        [x] for x in X
    ])
    return np.array([t for t in tuples])

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