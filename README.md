**sparse** is a Python module for multidimensional sparse matrix built over [NumPy](https://numpy.org/) package.

Documentation
-------------
The documentation associated with the module can be found in the doc strings of the functions and methods.

Dependencies
------------

* Python (>=3.8.6)
* NumPy (>=1.18.5)

Instalation
-----------
The easiest way to install sparse is using pip:
```
pip install sparse
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

Sparse matrix with *fill_value = 1* in memory:

|   | 1º dim | 2º dim | value |
|---|:------:|:------:|:-----:|
| 0 | 0      | 0      | 1     |
| 0 | 0      | 1      | 2     |
| 0 | 1      | 0      | 3     |
