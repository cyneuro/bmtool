from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bmtool",
    version="0.7.4",
    author="Neural Engineering Laboratory at the University of Missouri",
    author_email="gregglickert@mail.missouri.edu",
    description="BMTool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyneuro/bmtool",
    download_url="",
    license="MIT",
    install_requires=[
        "neuron==8.2.4",  # hard code version
        "bmtk",
        "click",
        "clint",
        "h5py",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "questionary",
        "pynmodlt",
        "xarray",
        "fooof",
        "requests",
        "pyyaml",
        "PyWavelets",
        "numba",
        "tqdm"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),  # Automatically finds all packages
    include_package_data=True,
    entry_points={"console_scripts": ["bmtool = bmtool.manage:cli"]},
)
