from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='gala_goo',
    ext_modules=[
        CUDAExtension(
                name='gala_model',
                sources=['gala.cu'],
                include_dirs=['/home/abakst/mocha/GALA-GNN-Acceleration-LAnguage/codegen_python/../src/'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']},
                extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
