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
        M = sparse.randint(
            low = 0,
            high = 10,
            sparsity = 0.1,
            shape = (10,10),
            fill_value = 0
        )
    
    def test_zeros(self):
        M1 = sparse.zeros(shape = (30,30))
        M2 = np.zeros((30,30))
        self.assertTrue(np.array_equal(
            M1.to_numpy(),
            M2
        ))

    def test_getitem(self):
        pass

    def test_getitem_one_value(self):
        shape = (5,5)
        M1 = sparse.randint(
            low = 1,
            high = 10,
            sparsity = 0.9,
            shape = shape,
            fill_value = 0
        )
        M2 = M1.to_numpy()
        index = np.random.randint(0,shape[0],2)
        self.assertEqual(M1[index[0],index[1]],M2[index[0],index[1]])
    
    def test_getitem_slice(self):
        shape = (50,50,50)
        M1 = sparse.randint(
            low = 1,
            high = 10,
            sparsity = 0.5,
            shape = shape,
            fill_value = 0
        )
        M2 = M1.to_numpy()
        self.assertTrue(np.array_equal(
            M1[1:40:2,:4,:10].to_numpy(),
            M2[1:40:2,:4,:10]
        ))

    def test_getitem_list(self):
        shape = (40,40)
        M1 = sparse.randint(
            low = 1,
            high = 10,
            sparsity = 0.5,
            shape = shape,
            fill_value = 0
        )
        M2 = M1.to_numpy()
        L1 = [0,2]
        L2 = [1,3]
        self.assertTrue(np.array_equal(
            M1[L1,L2].to_numpy(),
            M2[0:4:2,1:5:2]
        ))