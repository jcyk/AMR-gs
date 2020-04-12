#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(Extension("_smatch", sources=["_smatch.pyx", "_gain.cc"], language="c++",extra_compile_args=["-std=c++11"])))
