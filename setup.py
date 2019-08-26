from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path
import fdia_simulation

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fdia_simulation',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=fdia_simulation.__version__,

    description='False Data injection attacks simulation on state estimators (i.e. Kalman)',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/QDucasse/fdia_simulation',

    # Author details
    author='Quentin Ducasse',
    author_email='quentin.ducasse@ensta-bretagne.org',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',


        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',

        'Operating System :: Microsoft :: Windows'
    ],

    # What does your project relate to?
    keywords='Kalman filters filtering optimal estimation tracking attacks false data injection',

    packages=find_packages(),

    install_requires=['numpy', 'scipy', 'matplotlib','filterpy'],

)
