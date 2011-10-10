class Item:
    """
    An item, with its related metadata information

    :param id: item id
    :type id: string or int
    :returns: an item instance

    """
    def __init__(self, id):
        self._id = id
        self._data = None

    def __repr__(self):
        return str(self._id)

    def get_id(self):
        """
        Returns the Item id
        """
        return self._id

    def get_data(self):
        """
        Returns the associated information of the item
        """
        return self._data

    def add_data(self, data):
        """
        :param data: associated data for the item
        :type data: dict() or list()
        """
        self._data = data
