import os.path
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

VERSION = "1.0"

setup(
    name = "python-itunes",
    version = VERSION,
    description="A simple python wrapper to access iTunes Store API",
    author='Oscar Celma',
    author_email='ocelma@bmat.com',
    maintainer='Oscar Celma',
    maintainer_email='ocelma@bmat.com',
    license = "http://www.gnu.org/copyleft/gpl.html",
    platforms = ["any"],    
    url="https://github.com/ocelma/python-itunes",
    packages=['itunes'],
)
