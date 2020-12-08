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

class IndexTest(unittest.TestCase):
    def test_getitem_one_value(self):
        shape = (5,5)
        M1 = sparse.random.randint(
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
        M1 = sparse.random.randint(
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
        M1 = sparse.random.randint(
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

    def test_getitem_all(self):
        shape = (10,10,10)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.5,
            shape = shape,
            fill_value = 0
        )
        M2 = M1.to_numpy()
        self.assertTrue(np.array_equal(
            M1[:2,[0,1],0].to_numpy(),
            M2[:2,:2,0]
        ))

    def test_setitem_value2value(self):
        shape = (10,10,10)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 0
        )
        M1[1,2,3] = 5
        self.assertEqual(M1[1,2,3],5)
        M2 = M1.to_numpy()
        self.assertEqual(M2[1,2,3],5)

    def test_setitem_array2value(self):
        shape = (10,10)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 0
        )
        M1[:3,:6] = 5
        M2 = M1.to_numpy()
        self.assertTrue(np.all(M2[:3,:6] == 5))

    def test_setitem_array2array_fill_value_equal(self):
        shape = (100,100)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 0
        )
        shape = (4,4)
        M2 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 0
        )
        M1[:4,:4] = M2
        self.assertTrue(np.all(
            M1[:4,:4].to_numpy() == M2.to_numpy()
        ))

    def test_setitem_array2array_fill_value_not_equal(self):
        shape = (10,10)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 0
        )
        shape = (2,2)
        M2 = sparse.random.randint(
            low = 2,
            high = 10,
            sparsity = 0.8,
            shape = shape,
            fill_value = 1
        )
        M1[0:4:2,0:4:2] = M2
        self.assertTrue(np.all(M1[0:4:2,0:4:2].to_numpy() == M2.to_numpy()))

    def test_indexrandom1(self):
        M1 = sparse.zeros(shape = (10,10,10))
        M1[:10:2,5,0:10:3] = 2
        M1[:10:2,5,5:10:2] = 3
        print(M1.to_numpy())
        M2 = np.zeros(shape = (10,10,10))
        M2[:10:2,5,0:10:3] = 2
        M2[:10:2,5,5:10:2] = 3
        print(M2)
        self.assertTrue(np.array_equal(M1.to_numpy(),M2))
    
    def test_indexrandom2(self):
        M1 = sparse.zeros(shape = 10)
        M1[:10] = 2
        print(M1.to_numpy())
        M2 = sparse.zeros(shape = (10,10))
        M2[:,3] = M1
        print(M2.to_numpy())
        M2[3,:] = M1
        print(M2.to_numpy())

class GeneralTest(unittest.TestCase):
    def test_randint(self):
        shape = (30,10)
        low = 1
        high = 10
        sparsity = 0.2
        M = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        self.assertLess(M.T[:,-1].max(), high)
        self.assertGreaterEqual(M.T[:,-1].min(), low)
        self.assertTrue(np.array_equal(M.shape,np.array(shape)))
        self.assertAlmostEqual(sparsity,M.T.shape[0]/float(M.shape.prod()))
    
    def test_from_numpy(self):
        shape = (30,10)
        M1 = np.ones(shape = shape)
        M2 = sparse.from_numpy(M1)
        self.assertTrue(np.array_equal(np.array(M1.shape),M2.shape))
        self.assertEqual(np.count_nonzero(M1 != 0), M2.T.shape[0])

    def test_to_numpy(self):
        shape = (10,10)
        M1 = sparse.random.randint(
            low = 1,
            high = 10,
            sparsity = 0.1,
            shape = shape,
            fill_value = 0
        )
        M2 = M1.to_numpy()
        self.assertTrue(np.array_equal(
            np.array(M2.shape),
            np.array(shape)
        ))
    
    def test_zeros(self):
        M1 = sparse.zeros(shape = (30,30))
        M2 = np.zeros((30,30))
        self.assertTrue(np.array_equal(
            M1.to_numpy(),
            M2
        ))
        self.assertEqual(
            M1.T.shape[0],
            0
        )

class Operator(unittest.TestCase):
    def test_sum_sparse_value(self):
        shape = (30,10)
        low = 1
        high = 10
        sparsity = 0.2
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        value = 5
        self.assertTrue(np.array_equal(
            M1.to_numpy() + value,
            (M1 + value).to_numpy()
        ))

    def test_sum_sparse_sparse(self):
        shape = (3,3)
        low = 1
        high = 10
        sparsity = 0.4
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        M2 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        self.assertTrue(np.array_equal(
            M1.to_numpy() + M2.to_numpy(),
            (M1 + M2).to_numpy()
        ))

    def test_subtract_sparse_value(self):
        shape = (30,10)
        low = 1
        high = 10
        sparsity = 0.2
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        value = 5
        self.assertTrue(np.array_equal(
            M1.to_numpy() - value,
            (M1 - value).to_numpy()
        ))

    def test_subtract_sparse_sparse(self):
        shape = (3,3)
        low = 1
        high = 10
        sparsity = 0.4
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        M2 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        self.assertTrue(np.array_equal(
            M1.to_numpy() - M2.to_numpy(),
            (M1 - M2).to_numpy()
        ))

    def test_product_sparse_value(self):
        shape = (30,10)
        low = 1
        high = 10
        sparsity = 0.2
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        value = 5
        self.assertTrue(np.array_equal(
            M1.to_numpy() * value,
            (M1 * value).to_numpy()
        ))

    def test_product_sparse_sparse(self):
        shape = (3,3)
        low = 1
        high = 10
        sparsity = 0.4
        M1 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        M2 = sparse.random.randint(
            low = low,
            high = high,
            sparsity = sparsity,
            shape = shape
        )
        print(M1.to_numpy())
        print(M2.to_numpy())
        print((M1 * M2).to_numpy())
        print(M1.to_numpy() * M2.to_numpy())
        self.assertTrue(np.array_equal(
            M1.to_numpy() * M2.to_numpy(),
            (M1 * M2).to_numpy()
        ))
        


    
    
    

    

    

    
        
        