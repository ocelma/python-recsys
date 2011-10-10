Tests
=====

**python-recsys** provides a set of tests, to ensure that my ugly code is not
broken each time I pretend to implement something new.

Yet, I've found out that sniffing at those (nose) tests is good way to understand how to use this
library...

These tests make use of the `nosetest`_ library.
Install python-nose (in Ubuntu):

.. _`nosetest`: http://code.google.com/p/python-nose/

.. code-block:: python

    sudo apt-get install python-nose

How to run the tests
--------------------

To run the set of tests (and see some output ugly *print's*) just do:

.. code-block:: python

    cd PATH_TO/python-recsys
    nosetests -s -v

.. note::
    To run test.test_algorithm you will need to download the `Movielens`_ 1M
    Ratings Data Set, and save it here: 
    
    .. code-block:: python

        PATH_TO/python-recsys/recsys/tests/data/movielens

.. _`Movielens`: http://www.grouplens.org/node/73

.. warning:: 
    It takes a few minutes to run the tests.test_algorithm, as it computes 
    the SVD of the input matrix.

If you want to run only one package tests, do:

.. code-block:: python

    cd PATH_TO/python-recsys
    nosetests -s -v tests.test_evaluation


