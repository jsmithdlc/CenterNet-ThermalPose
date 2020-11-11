from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('colormap2gray_cy.pyx'),include_dirs=[numpy.get_include()])