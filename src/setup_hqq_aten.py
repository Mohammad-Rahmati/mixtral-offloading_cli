from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hqq_aten',
	ext_modules=[cpp_extension.CppExtension('hqq_aten', ['hqq_aten.cpp'])],
	extra_compile_args=['-O3'],
 	cmdclass={'build_ext': cpp_extension.BuildExtension})

#python3 setup_hqq_aten.py bdist_wheel
#pip install dist/hqq_aten-<version>-<tags>.whl
