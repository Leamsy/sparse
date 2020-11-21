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

class test_array(unittest.TestCase):
    def test_randint(self):
        M = sparse.randint(
            low = 0,
            high = 10,
            sparsity = 0.2,
            shape = (30,10)
        )
        self.assertEqual(0,0)
    
    def test_from_numpy(self):
        M = sparse.from_numpy(np.ones(shape = (30,10)))
        self.assertEqual(0,0)

    def test_to_numpy(self):
        M = sparse.randint()
    
    def test_zeros(self):
        M1 = sparse.zeros(shape = (30,30))
        M2 = np.zeros((30,30))
        self.assertTrue(np.array_equal(
            M1.to_numpy(),
            M2
        ))