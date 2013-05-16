Installation
============

Dependencies
~~~~~~~~~~~~

**python-recsys** is build on top of `Divisi2`_, with csc-pysparse (Divisi2 also requires `NumPy`_, and uses Networkx).

.. _`Divisi2`: http://csc.media.mit.edu/docs/divisi2/install.html
.. _`NumPy`: http://numpy.scipy.org

**python-recsys** also requires `SciPy`_.

.. _`SciPy`: http://numpy.scipy.org

To install the dependencies do something like this (Ubuntu):

.. code-block:: python

    sudo apt-get install python-scipy python-numpy
    sudo apt-get install python-pip
    sudo pip install csc-pysparse networkx divisi2

    # If you don't have pip installed 
    # (i.e. the last command, sudo pip install, fails)
    # then do:
    # sudo easy_install csc-pysparse
    # sudo easy_install networkx
    # sudo easy_install divisi2

.. note::
    If you get an error like this one while compiling csc-pysparse:

    .. error::
        (...) error: Python.h: No such file or directory"

    then you need to also install python-dev package:

    .. code-block:: python

        sudo apt-get install python-dev

Download
~~~~~~~~

Download **python-recsys** from `github`_.

.. _`github`: http://github.com/ocelma/python-recsys

Install
~~~~~~~

.. code-block:: python

    tar xvfz python-recsys.tar.gz
    cd python-recsys
    sudo python setup.py install
