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
    randint,
    zeros,
    ones
)
import sparse.utils