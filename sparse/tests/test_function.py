# -----------------------------------------------------------------------------
# 
# 
# (C) 2020 Ernesto Martínez del Pino, Granada, Spain
# Released under GNU Public Licence (GPL)
# email: ernestomar1997@hotmail.com
# -----------------------------------------------------------------------------

# python -m unittest sparse.tests.test_array --verbose

import unittest
import numpy as np

import sparse



class FunctionTest(unittest.TestCase):
    def test_min_max(self):
        from itertools import product
        tuples = product([1,2,3],[4,5,6])
        """for t in tuples:
            print(t)"""
        minimum, maximum = sparse.min_max(tuples,2)
        
        print(minimum)
        print(maximum)

    def test_nindex_and_oneindex(self):
        from sparse.functions import (
            nindex_to_oneindex,
            oneindex_to_nindex,
            max_oneindex
        )
        ncolumns = 2
        nrows = 5000
        max_index_value = 10
        nindex = np.random.randint(0,max_index_value, size = (nrows,ncolumns))
        shape = np.array([max_index_value for i in range(ncolumns)])
        oneindex = nindex_to_oneindex(nindex,shape)
        nindex_calculated = oneindex_to_nindex(oneindex,shape)
        self.assertTrue(np.all(nindex == nindex_calculated))
        self.assertTrue(np.all(oneindex < max_oneindex(shape)))

    def test_filter_coordinates(self):
        from sparse.functions import (
            inverse_filter_coordinates,
            filter_coordinates
        )
        args = [slice(0,10,1),[0,2,4,6,8,1,3,5,7,9],4]
        mask = [0,1]
        X = np.random.randint(0,10,(5,3))
        Y = filter_coordinates(X,args)
        X_new = inverse_filter_coordinates(Y,args)
        self.assertTrue(np.all(X[:,mask] == X_new[:,mask]))