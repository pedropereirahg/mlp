"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from glob import glob

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

octaveFiles = glob('./*.m')

setup(
    name='mlp',
    version='1.0.0',
    description='Implements a multilayer perceptron (MLP) with k-fold cross-validation in Octave.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/pypa/sampleproject',

    # Author details
    author='Pedro Pereira',
    author_email='pedrogoncalvesp.95@gmail.com',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='artificial intelligence mlp kfold cross validation',

    py_modules=["mlp"],
    data_files=[('mlp', octaveFiles)]
)
