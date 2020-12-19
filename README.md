**sparse** is a Python module for multidimensional sparse matrix built over [NumPy](https://numpy.org/) package.

Documentation
-------------
For the moment, the only documentation available can be found in doc strings associated with functions and methods. The development of the package follows the same structure as NumPy.

Dependencies
------------
* Python (>=3.8.6)
* NumPy (>=1.18.5)

Instalation
-----------
The easiest way to install sparse is using pip:
```
pip install sparse_pkg_rojo1997
```

Project Submodules
------------------
* Random: different ways to generate random sparse matrices.
* Utils: set of functions to transform indexes.
* Tests: test set for the project.
* Array: core file of the project.

Description
-----------

Dense matrix (2,2) example in memory:

|   | 0 | 1 |
|---|:-:|:-:|
| 0 | 1 | 2 |
| 1 | 3 | 0 |

Sparse matrix with *fill_value = 0* in memory:

|   | 1º dim | 2º dim | value |
|---|:------:|:------:|:-----:|
| 0 | 0      | 0      | 1     |
| 0 | 0      | 1      | 2     |
| 0 | 1      | 0      | 3     |

1) The order of the rows in the matrix does not affect the implementation of the algorithms.
2) In some case, algorithms use an mathematical application to reduce the computational cost translating all the coordinates into a single one. This means that the product of the dimensions must enter the *dtype* range.
$$ \prod_{s\in shape}s< max(dtype) $$
