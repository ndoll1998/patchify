from distutils.core import setup

setup(
    name="pypatchify",
    version="0.1.2",
    description="Fast and easy image and n-dimensional volume patchification",
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    author="Niclas Doll",
    author_email="niclas@amazonis.net",
    url="https://github.com/ndoll1998/patchify/tree/master",
    packages=['pypatchify'],
    package_dir={'pypatchify': 'pypatchify'},
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing"
    ]
) 
