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

Example
~~~~~~~

1. Load Movielens dataset:

::

    from recsys.algorithm.factorize import SVD
    svd = SVD()
    svd.load_data(filename='./data/movielens/ratings.dat', 
                sep='::', 
                format={'col':0, 'row':1, 'value':2, 'ids': int})

2. Compute Singular Value Decomposition (SVD), M=U Sigma V^t:

::

    k = 100
    svd.compute(k=k, 
                min_values=10, 
                pre_normalize=None, 
                mean_center=True, 
                post_normalize=True, 
                savefile='/tmp/movielens')

3. Get similarity between two movies:

::

    ITEMID1 = 1    # Toy Story (1995)
    ITEMID2 = 2355 # A bug's life (1998)

    svd.similarity(ITEMID1, ITEMID2)
    # 0.67706936677315799

4. Get movies similar to *Toy Story*:

::

    svd.similar(ITEMID1)

    # Returns: <ITEMID, Cosine Similarity Value>
    [(1,    0.99999999999999978), # Toy Story
     (3114, 0.87060391051018071), # Toy Story 2
     (2355, 0.67706936677315799), # A bug's life
     (588,  0.5807351496754426),  # Aladdin
     (595,  0.46031829709743477), # Beauty and the Beast
     (1907, 0.44589398718134365), # Mulan
     (364,  0.42908159895574161), # The Lion King
     (2081, 0.42566581277820803), # The Little Mermaid
     (3396, 0.42474056361935913), # The Muppet Movie
     (2761, 0.40439361857585354)] # The Iron Giant

5. Predict the rating a user (USERID) would give to a movie (ITEMID):

::

    MIN_RATING = 0.0
    MAX_RATING = 5.0
    ITEMID = 1
    USERID = 1

    svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
    # Predicted value 5.0 

    svd.get_matrix().value(ITEMID, USERID)
    # Real value 5.0 

6. Recommend (non-rated) movies to a user:

::

    svd.recommend(USERID, is_row=False) #cols are users and rows are items, thus we set is_row=False

    # Returns: <ITEMID, Predicted Rating>
    [(2905, 5.2133848204673416), # Shaggy D.A., The
     (318,  5.2052108435956033), # Shawshank Redemption, The
     (2019, 5.1037438278755474), # Seven Samurai (The Magnificent Seven)
     (1178, 5.0962756861447023), # Paths of Glory (1957)
     (904,  5.0771405690055724), # Rear Window (1954)
     (1250, 5.0744156653222436), # Bridge on the River Kwai, The
     (858,  5.0650911066862907), # Godfather, The
     (922,  5.0605327279819408), # Sunset Blvd.
     (1198, 5.0554543765500419), # Raiders of the Lost Ark
     (1148, 5.0548789542105332)] # Wrong Trousers, The

7. Which users should *see* Toy Story? (e.g. which users -that have not rated Toy
   Story- would give it a high rating?)

::

    svd.recommend(ITEMID)

    # Returns: <USERID, Predicted Rating>
    [(283,  5.716264440514446),
     (3604, 5.6471765418323141),
     (5056, 5.6218800339214496),
     (446,  5.5707524860615738),
     (3902, 5.5494529168484652),
     (4634, 5.51643364021289),
     (3324, 5.5138903299082802),
     (4801, 5.4947999354188548),
     (1131, 5.4941438045650068),
     (2339, 5.4916048051511659)]
    

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


