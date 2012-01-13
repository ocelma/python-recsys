import os.path
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

VERSION = "0.2"

setup(
    name = "python-recsys",
    version = VERSION,
    description="A simple recommender system in python",
    author='Oscar Celma',
    author_email='ocelma@bmat.com',
    maintainer='Oscar Celma',
    maintainer_email='ocelma@bmat.com',
    license = "http://www.gnu.org/copyleft/gpl.html",
    platforms = ["any"],    
    url="http://www.dtic.upf.edu/~ocelma/software/python-recsys",
    package_dir={'recsys':'recsys'},
    packages=['recsys', 'recsys.algorithm', 'recsys.datamodel', 'recsys.evaluation', 'recsys.utils'],
    install_requires = ["numpy", "scipy", "divisi2", "csc-pysparse"],
)
