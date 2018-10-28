from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("current_disk.pyx"))
setup(ext_modules=cythonize("internal_field.pyx"))