# python setup.py sdist bdist_wheel
# python -m pip install .\dist\sparse_pkg_rojo1997-0.0.1-py3-none-any.whl --ignore-installed
# python -m twine check dist/*
# python -m twine upload --repository pypi dist/* --verbose
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "sparse-pkg-rojo1997",
    version = "0.0.2",
    author = "Ernesto Martínez del Pino",
    author_email = "ernestomar1997@hotmail.com",
    description = "sparse is a Python module for multidimensional sparse matrix built over NumPy package.",
    license = "GNU General Public License v3 or later (GPLv3+)",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/rojo1997/sparse",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent"
    ],
    python_requires = '>=3.8'
)

if __name__ == '__main__':
    pass