========
pyrecsys
========

A python library for implementing a recommender system.

Installation
============

Dependencies
~~~~~~~~~~~~

**pyrecsys** is build on top of `Divisi2`_, with csc-pysparse (Divisi2 also requires `NumPy`_).

.. _`Divisi2`: http://csc.media.mit.edu/docs/divisi2/install.html
.. _`NumPy`: http://numpy.scipy.org

**pyrecsys** also requires `SciPy`_.

.. _`SciPy`: http://numpy.scipy.org

To install the dependencies do something like this (Ubuntu):

.. code-block:: python

    sudo apt-get install python-scipy
    sudo apt-get install python-numpy
    sudo pip install divisi2 csc-pysparse

    # If you don't have pip installed then do:
    # sudo easy_install csc-pysparse
    # sudo easy_install divisi2

Download
~~~~~~~~

Download **pyrecsys** from `github`_.

.. _`github`: http://github.com/ocelma/python-recsys

Install
~~~~~~~

.. code-block:: python

    tar xvfz pyrecsys.tar.gz
    cd pyrecsys
    sudo python setup.py install
