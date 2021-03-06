# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Granada, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------
"""
Module Sparse
=============

Work with multidimensional sparse matrix using numpy style.
"""
import sys
import warnings

from sparse.array import (
    array,
    from_numpy,
    zeros,
    ones
)

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

import sparse.utils
import sparse.random