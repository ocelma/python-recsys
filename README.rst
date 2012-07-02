=============
python-recsys
=============

A python library for implementing a recommender system.

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

::

    sudo apt-get install python-scipy python-numpy
    sudo apt-get install python-pip
    sudo pip install csc-pysparse networkx divisi2

    # If you don't have pip installed then do:
    # sudo easy_install csc-pysparse
    # sudo easy_install networkx
    # sudo easy_install divisi2

Download
~~~~~~~~

Download **python-recsys**  from `github`_.

.. _`github`: http://github.com/ocelma/python-recsys

Install
~~~~~~~

::

    tar xvfz python-recsys.tar.gz
    cd python-recsys
    sudo python setup.py install

Documentation
~~~~~~~~~~~~~

Documentation and examples available `here`_.

.. _`here`: http://ocelma.net/software/python-recsys/build/html

To create the HTML documentation files from doc/source do:

::

    cd doc
    make html

HTML files are created here: 

::

    doc/build/html/index.html


