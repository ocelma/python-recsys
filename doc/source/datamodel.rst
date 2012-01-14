Data model
==========

**pyrecsys** data model includes: users, items, and its interaction.

Items
-----

Create an item, with an **Id**, and some metadata:

.. code-block:: python
    
    from recsys.datamodel.item import Item

    ITEMID = 'a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432'
    item = Item(ITEMID)
    # ...plus any other info you'd like to add
    name = 'U2'
    genres = ['rock', 'irish', '80s']
    popularity = 8.89
    item.add_data({'name': name, 'genres': genres, 'popularity': popularity})

Create a dict of items, reading from a file. This example actually reads the `Movielens`_ 1M
Ratings Movies file (*movies.dat*):

.. _`Movielens`: http://www.grouplens.org/node/73

.. code-block:: python

    # Read movie info
    def read_items(filename):
        items = dict()
        for line in open(filename):
            #1::Toy Story (1995)::Animation|Children's|Comedy
            data =  line.strip('\r\n').split('::')
            item_id = int(data[0])
            item_name = data[1]
            genres = data[2].split('|')
            item = Item(item_id)
            item.add_data({'name': item_name, 'genres': genres})
            items[item_id] = item
        return items

    # Call it!
    filename = './data/movielens/movies.dat'
    items = read_items(filename)

Users
-----

Create a user, with a given **Id**:

.. code-block:: python

    from recsys.datamodel.user import User

    USERID = 'ocelma' 
    user = User(USERID)

Link an item with a user, plus its interaction (rating, number of plays, views, etc.):

.. code-block:: python

    from recsys.datamodel.user import User
    from recsys.datamodel.item import Item

    ITEMID = 'a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432'
    item = Item(ITEMID)
    # ...plus any other info you'd like to add
    name = 'U2'
    item.add_data({'name': name})

    USERID = 'ocelma' 
    PLAYS = 256
    user = User(USERID)
    user.add_item(ITEMID, PLAYS) #Instead of PLAYS, one can add the classical [1..5] stars (rating)


Data
----

Data class manages "users rate items" information.

Loading user data, and adding tuples to Data:

.. code-block:: python
    
    from recsys.datamodel.data import Data

    data = Data()
    for PLAYS, ITEMID in user.get_items():
        data.add_tuple((PLAYS, ITEMID, user.get_id())) # Tuple format is: <value, row, column>

Loading a train/test dataset from a file. This example actually reads the Movielens 1M Ratings Data Set (ratings.dat) file:

.. code-block:: python

    from recsys.datamodel.data import Data

    filename = './data/movielens/ratings.dat'

    data = Data()
    format = {'col':0, 'row':1, 'value':2, 'ids': 'int'}
        # About format parameter:
        #   'row': 1 -> Rows in matrix come from column 1 in ratings.dat file
        #   'col': 0 -> Cols in matrix come from column 0 in ratings.dat file
        #   'value': 2 -> Values (Mij) in matrix come from column 2 in ratings.dat file
        #   'ids': int -> Ids (row and col ids) are integers (not strings)
    data.load(filename, sep='::', format=format)
    train, test = data.split_train_test(percent=80) # 80% train, 20% test

Getting data from the test dataset:

.. code-block:: python

    for rating, item_id, user_id in test:
        pass # Do something, like evaluating how well we can predict the ratings in this test dataset

Accessing the test dataset as if it were a list:

.. code-block:: python

    test[3] 

The Data class can also store the information to disk:

.. code-block:: python

    data.save(FILENAME)

Or even load or save data using the pickle format:

.. code-block:: python

    data.load_pickle(FILENAME)
    data.save_pickle(FILENAME)

