from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='llpro',
    ext_modules=cythonize('llpro/**/*.pyx'),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
