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

class test_function(unittest.TestCase):
    def test_min_max(self):
        from itertools import product
        tuples = product([1,2,3],[4,5,6])
        """for t in tuples:
            print(t)"""
        minimum, maximum = sparse.min_max(tuples,2)
        
        print(minimum)
        print(maximum)