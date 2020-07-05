import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Stolen from: https://packaging.python.org/tutorials/packaging-projects/ and
# https://github.com/pypa/sampleproject/blob/master/setup.py

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galsim_timeit",
    version="0.0.1",
    author="Skanda Kaashyap",
    author_email="skaashya@andrew.cmu.edu",
    description="A small add on to the GalSim tool to perform timing experiments.",
    long_description=long_description,
    url="https://github.com/kaashmonee/galsim-timeit",
    packages=setuptools.find_packages(where='./'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.6",
)



