#!/usr/bin/python
"""A python interface to search iTunes Store"""
import os
import urllib2, urllib
import urlparse
import re
try: 
    import simplejson as json
except ImportError: 
    import json
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

__name__ = 'pyitunes'
__doc__ = 'A python interface to search iTunes Store'
__author__ = 'Oscar Celma'
__version__ = '0.1'
__license__ = 'GPL'
__maintainer__ = 'Oscar Celma'
__email__ = 'ocelma@bmat.com'
__status__ = 'Beta'

API_VERSION = '2'        # iTunes API version
COUNTRY = 'US'           # ISO Country Store
HOST_NAME = 'http://itunes.apple.com/'

__cache_enabled = False  # Enable cache? if set to True, make sure that __cache_dir exists! (e.g. $ mkdir ./cache)
__cache_dir = './cache'  # Set cache directory

class ServiceException(Exception):
    """Exception related to the web service."""
    
    def __init__(self, type, message):
        self._type = type
        self._message = message
    
    def __str__(self):
        return self._type + ': ' + self._message
    
    def get_message(self):
        return self._message

    def get_type(self):
        return self._type

class _Request(object):
    """Representing an abstract web service operation."""

    def __init__(self, method_name, params):
        self.params = params
        self.method = method_name

    def _download_response(self):
        """Returns a response"""
        data = []
        for name in self.params.keys():
            value = self.params[name]
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, long):
                value = str(value)
            try:
                data.append('='.join((name, urllib.quote_plus(value.replace('&amp;', '&').encode('utf8')))))
            except UnicodeDecodeError:
                data.append('='.join((name, urllib.quote_plus(value.replace('&amp;', '&')))))
        data = '&'.join(data)

        url = HOST_NAME
        parsed_url = urlparse.urlparse(url)
        if not parsed_url.scheme:
            url = "http://" + url
        url += self.method + '?'
        url += data
        #print url

        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        return response.read() 

    def execute(self, cacheable=False):
        try:
            if is_caching_enabled() and cacheable:
                response = self._get_cached_response()
            else:
                response = self._download_response()
            return json.loads(response)
        except urllib2.HTTPError, e:
            raise self._get_error(e.fp.read())

    def _get_cache_key(self):
        """Cache key""" 
        keys = self.params.keys()[:]
        keys.sort()
        string = self.method
        for name in keys:
            string += name
            if isinstance(self.params[name], int) or isinstance(self.params[name], float):
                self.params[name] = str(self.params[name])
            string += self.params[name]
        return get_md5(string)

    def _is_cached(self):
        """Returns True if the request is available in the cache."""
        return os.path.exists(os.path.join(_get_cache_dir(), self._get_cache_key()))

    def _get_cached_response(self):
        """Returns a file object of the cached response."""
        if not self._is_cached():
            response = self._download_response()
            response_file = open(os.path.join(_get_cache_dir(), self._get_cache_key()), "w")
            response_file.write(response)
            response_file.close()
        return open(os.path.join(_get_cache_dir(), self._get_cache_key()), "r").read()

    def _get_error(self, text):
        return ServiceException(type='Error', message=text)
        raise

# Webservice BASE OBJECT
class _BaseObject(object):
    """An abstract webservices object."""
        
    def __init__(self, method):
        self._method = method
        self._search_terms = dict()
    
    def _request(self, method_name=None, params = None, cacheable = False):
        if not method_name:
            method_name = self._method
        if not params:
            params = self._get_params()    
        return _Request(method_name, params).execute(cacheable)
    
    def _get_params(self):
        params = {}
        for key in self._search_terms.keys():
            params[key] = self._search_terms[key]
        return params

    def get(self):
        self._json_results = self._request(cacheable=is_caching_enabled())
        if self._json_results.has_key('errorMessage'):
            raise ServiceException(type='Error', message=self._json_results['errorMessage'])
        self._num_results = self._json_results['resultCount']
        l = []
        for json in self._json_results['results']:
            type = None
            if json.has_key('wrapperType'):
                type = json['wrapperType']
            elif json.has_key('kind'):
                type = json['kind']

            if type == 'artist':
                id = json['artistId']
                item = Artist(id)
            elif type == 'collection':
                id = json['collectionId']
                item = Album(id)
            elif type == 'track':
                id = json['trackId']
                item = Track(id)
            elif type == 'audiobook':
                id = json['collectionId']
                item = Audiobook(id)
            elif type == 'software':
                id = json['trackId']
                item = Software(id)
            else:
                if json.has_key('collectionId'):
                    id = json['collectionId']
                elif json.has_key('artistId'):
                    id = json['artistId']
                item = Item(id)
            item._set(json)
            l.append(item)
        return l

# SEARCH
class Search(_BaseObject):
    """ Search iTunes Store """

    def __init__(self, query, country=COUNTRY, media='all', entity=None, attribute=None, limit=50, lang='en_us', version=API_VERSION, explicit='Yes'):
        _BaseObject.__init__(self, 'search')

        self._search_terms = dict()
        self._search_terms['term'] = query
        self._search_terms['country'] = country   # ISO Country code for iTunes Store
        self._search_terms['media'] = media       # The media type you want to search for
        if entity:
            self._search_terms['entity'] = entity # The type of results you want returned, relative to the specified media type
        if attribute:
            self._search_terms['attribute'] = attribute # The attribute you want to search for in the stores, relative to the specified media type
        self._search_terms['limit'] = limit       # Results limit
        self._search_terms['lang'] = lang         # The language, English or Japanese, you want to use when returning search results
        self._search_terms['version'] = version   # The search result key version you want to receive back from your search
        self._search_terms['explicit'] = explicit # A flag indicating whether or not you want to include explicit content in your search results

        self._json_results = None
        self._num_results = None

    def num_results(self):
        return self._num_results

# LOOKUP
class Lookup(_BaseObject):
    """ Lookup """

    def __init__(self, id, entity=None, limit=50):
        _BaseObject.__init__(self, 'lookup')
        self.id = id
        self._search_terms['id'] = id
        if entity:
            self._search_terms['entity'] = entity# The type of results you want returned, relative to the specified media type
        self._search_terms['limit'] = limit      # Results limit


# RESULT ITEM 
class Item(object):
    """ Item result class """

    def __init__(self, id):
        self.id = id
        self.name = None
        self.url = None

    # JSON SETTERs
    def _set(self, json):
        self.json = json
        #print json
        if json.has_key('kind'):
            self.type = json['kind']
        else:
            self.type = json['wrapperType']
        # Item information
        self._set_genre(json)
        self._set_release(json)
        self._set_country(json)
        self._set_artwork(json)
        self._set_url(json)

    def _set_genre(self, json):
        self.genre = json.get('primaryGenreName', None)

    def _set_release(self, json):
        self.release_date = None
        if json.has_key('releaseDate') and json['releaseDate']:
            self.release_date = json['releaseDate'].split('T')[0]

    def _set_country(self, json):
        self.country_store = json.get('country', None)

    def _set_artwork(self, json):
        self.artwork = dict()
        if json.has_key('artworkUrl30'): 
            self.artwork['30'] = json['artworkUrl30']
        if json.has_key('artworkUrl60'): 
            self.artwork['60'] = json['artworkUrl60']
        if json.has_key('artworkUrl100'): 
            self.artwork['100'] = json['artworkUrl100']
        if json.has_key('artworkUrl512'): 
            self.artwork['512'] = json['artworkUrl512']

    def _set_url(self, json):
        self.url = None
        if json.has_key('trackViewUrl'):
            self.url = json['trackViewUrl']
        elif json.has_key('collectionViewUrl'):
            self.url = json['collectionViewUrl']
        elif json.has_key('artistViewUrl'):
            self.url = json['artistViewUrl']

    # REPR, EQ, NEQ
    def __repr__(self):
        if not self.name:
            if self.json.has_key('collectionName'):
                self._set_name(self.json['collectionName'])
            elif self.json.has_key('artistName'):
                self._set_name(self.json['artistName'])
        return self.name.encode('utf8')

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def _set_name(self, name):
        self.name = name

    # GETTERs
    def get_id(self):
        if not self.id:
            if self.json.has_key('collectionId'):
                self.id = self.json['collectionId']
            elif self.json.has_key('artistId'):
                self.id = self.json['artistId']
        return self.id

    def get_name(self):
        """ Returns the Item's name """
        return self.__repr__()

    def get_url(self):
        """ Returns the iTunes Store URL of the Item """
        return self.url

    def get_genre(self):
        """ Returns the primary genre of the Item """
        return self.genre

    def get_release_date(self):
        """ Returns the release date of the Item """
        return self.release_date

    def get_artwork(self):
        """ Returns the artwork (a dict) of the item """
        return self.artwork

    def get_tracks(self, limit=500):
        """ Returns the tracks of the Item """
        if self.type == 'song':
            return self
        items = Lookup(id=self.id, entity='song', limit=limit).get()
        if not items:
            raise ServiceException(type='Error', message='Nothing found!')
        return items[1:]

    def get_albums(self, limit=200):
        """ Returns the albums of the Item """
        if self.type == 'collection':
            return self
        if self.type == 'song':
            return self.get_album()
        items = Lookup(id=self.id, entity='album', limit=limit).get()[1:]
        if not items:
            raise ServiceException(type='Error', message='Nothing found!')
        return items[1:]

    def get_album(self):
        """ Returns the album of the Item """
        if self.type == 'collection':
            return self
        items = Lookup(id=self.id, entity='album', limit=1).get()
        if not items or len(items) == 1:
            raise ServiceException(type='Error', message='Nothing found!')
        return items[1]

# ARTIST
class Artist(Item):
    """ Artist class """
    def __init__(self, id):
        Item.__init__(self, id)

    def _set(self, json):
        super(Artist, self)._set(json)
        self.name = json['artistName']
        self.amg_id = json.get('amgArtistId', None)
        self.url = json.get('artistViewUrl', json.get('artistLinkUrl', None))

    # GETTERs
    def get_amg_id(self):
        return self.amg_id

# ALBUM
class Album(Item):
    """ Album class """
    def __init__(self, id):
        Item.__init__(self, id)

    def _set(self, json):
        super(Album, self)._set(json)
        # Collection information
        self.name = json['collectionName']
        self.url = json.get('collectionViewUrl', None)
        self.amg_id = json.get('amgAlbumId', None)

        self.price = round(json['collectionPrice'] or 0, 4)
        self.price_currency = json['currency']
        self.track_count = json['trackCount']
        self.copyright = json.get('copyright', None)

        self._set_artist(json)

    def _set_artist(self, json):
        self.artist = None
        if json.get('artistId'):
            id = json['artistId']
            self.artist = Artist(id)
            self.artist._set(json)

    # GETTERs
    def get_amg_id(self):
        return self.amg_id

    def get_copyright(self):
        return self.copyright

    def get_price(self):
        return self.price

    def get_track_count(self):
        return self.track_count

    def get_artist(self):
        return self.artist

# TRACK
class Track(Item):
    """ Track class """
    def __init__(self, id):
        Item.__init__(self, id)

    def _set(self, json):
        super(Track, self)._set(json)
        # Track information
        self.name = json['trackName']
        self.url = json.get('trackViewUrl', None)
        self.preview_url = json.get('previewUrl', None)
        self.price = None
        if json.has_key('trackPrice') and json['trackPrice'] is not None:
            self.price = round(json['trackPrice'], 4)
        self.number = json.get('trackNumber', None)
        self.duration = None
        if json.has_key('trackTimeMillis') and json['trackTimeMillis'] is not None:
            self.duration = round(json.get('trackTimeMillis', 0.0)/1000.0, 2)
        try:
            self._set_artist(json)
        except KeyError:
            self.artist = None
        try:
            self._set_album(json)
        except KeyError:
            self.album = None

    def _set_artist(self, json):
        self.artist = None
        if json.get('artistId'):
            id = json['artistId']
            self.artist = Artist(id)
            self.artist._set(json)

    def _set_album(self, json):
        if json.has_key('collectionId'):
            id = json['collectionId']
            self.album = Album(id)
            self.album._set(json)

    # GETTERs
    def get_preview_url(self):
        return self.preview_url

    def get_disc_number(self):
        return self.number

    def get_duration(self):
        return self.duration

    def get_artist(self):
        return self.artist

    def get_price(self):
        return self.price

# Audiobook
class Audiobook(Album):
    """ Audiobook class """
    def __init__(self, id):
        Album.__init__(self, id)

# Software
class Software(Track):
    """ Audiobook class """
    def __init__(self, id):
        Track.__init__(self, id)

    def _set(self, json):
        super(Software, self)._set(json)
        self._set_version(json)
        self._set_price(json)
        self._set_description(json)
        self._set_screenshots(json)
        self._set_genres(json)
        self._set_seller_url(json)
        self._set_languages(json)
        self._set_avg_rating(json)
        self._set_num_ratings(json)

    def _set_version(self, json):
        self.version = json.get('version', None)

    def _set_price(self, json):
        self.price = json.get('price', None)

    def _set_description(self, json):
        self.description = json.get('description', None)

    def _set_screenshots(self, json):
        self.screenshots = json.get('screenshotUrls', None)

    def _set_genres(self, json):
        self.genres = json.get('genres', None)

    def _set_seller_url(self, json):
        self.seller_url = json.get('sellerUrl', None)

    def _set_languages(self, json):
        self.languages = json.get('languageCodesISO2A', None)

    def _set_avg_rating(self, json, only_current_version=False):
        if only_current_version:
            self.avg_rating = json.get('averageUserRatingForCurrentVersion', None)
        else:
            self.avg_rating = json.get('averageUserRating', None)

    def _set_num_ratings(self, json, only_current_version=False):
        if only_current_version:
            self.num_ratings = json.get('userRatingCountForCurrentVersion', None)
        else:
            self.num_ratings = json.get('userRatingCount', None)

    # GETTERs
    def get_version(self):
        return self.version
    def get_description(self):
        return self.description
    def get_screenshots(self):
        return self.screenshots
    def get_genres(self):
        return self.genres
    def get_seller_url(self):
        return self.seller_url
    def get_languages(self):
        return self.languages
    def get_avg_rating(self):
        return self.avg_rating
    def get_num_ratings(self):
        return self.num_ratings

# CACHE
def enable_caching(cache_dir = None):
    global __cache_dir
    global __cache_enabled

    if cache_dir == None:
        import tempfile
        __cache_dir = tempfile.mkdtemp()
    else:
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        __cache_dir = cache_dir
    __cache_enabled = True

def disable_caching():
    global __cache_enabled
    __cache_enabled = False

def is_caching_enabled():
    """Returns True if caching is enabled."""
    global __cache_enabled
    return __cache_enabled

def _get_cache_dir():
    """Returns the directory in which cache files are saved."""
    global __cache_dir
    global __cache_enabled
    return __cache_dir

def get_md5(text):
    """Returns the md5 hash of a string."""
    hash = md5()
    try:
        hash.update(text.encode('utf8'))
    except UnicodeDecodeError:
        hash.update(text)
    return hash.hexdigest()

#SEARCHES
def search_track(query, limit=100, store=COUNTRY):
    return Search(query=query, media='music', entity='song', limit=limit, country=store).get()

def search_album(query, limit=100, store=COUNTRY):
    return Search(query=query, media='music', entity='album', limit=limit, country=store).get()

def search_artist(query, limit=100, store=COUNTRY):
    return Search(query=query, media='music', entity='musicArtist', limit=limit, country=store).get()

def search(query, media='all', limit=100, store=COUNTRY):
    return Search(query=query, media=media, limit=limit, country=store).get()

#LOOKUP
def lookup(id):
    items = Lookup(id).get()
    if not items:
        raise ServiceException(type='Error', message='Nothing found!')
    return items[0]
