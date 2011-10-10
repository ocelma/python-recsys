class User:
    """
    User information, including her interaction with the items

    :param id: user id
    :type id: string or int
    :returns: a user instance

    """
    def __init__(self, id):
        self._id = id
        self._items = []

    def __repr__(self):
        return str(self._id)

    def get_id(self):
        """
        Returns the User id
        """
        return self._id

    def add_item(self, item_id, weight): 
        """
        :param item_id: An item ID
        :param weight: The weight (rating, views, plays, etc.) of the item_id for this user
        """
        self._items.append((item_id, weight))

    def get_items(self):
        """
        Returns the list of items for the user
        """
        return self._items
