# -*- coding: utf-8 -*-
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true
import itunes

U2 = 'U2'
U2_ONE = 'One'
U2_ACHTUNGBABY = 'Achtung Baby (Deluxe Edition) [Remastered]' # 'Achtung Baby'

U2_ONE_ID = 475391315 # Before it was 368617
U2_ACHTUNGBABY_ID = 475390461 # Before it was 368713
U2_ID = 78500

U2_URL = 'http://itunes.apple.com/us/artist/u2/id%s?uo=4' % U2_ID
U2_ACHTUNGBABY_URL = 'http://itunes.apple.com/us/album/achtung-baby-deluxe-edition/id%s?uo=4' % U2_ACHTUNGBABY_ID
U2_ONE_URL = 'http://itunes.apple.com/us/album/one/id%s?i=%s&uo=4' % (U2_ACHTUNGBABY_ID, U2_ONE_ID)

#SEARCHES
def test_search_track():
    assert_equal(itunes.search_track('u2 achtung baby one')[0].get_id(), U2_ONE_ID)

def test_search_album():
    assert_equal(itunes.search_album('u2 achtung baby')[0].get_id(), U2_ACHTUNGBABY_ID)

def test_search_artist():
    assert_equal(itunes.search_artist('u2')[0].get_id(), U2_ID)

def test_search_artist_store():
    U2_URL_ES = 'http://itunes.apple.com/es/artist/u2/id78500?l=en&uo=4'
    assert_equal(itunes.search_artist('u2', store='ES')[0].get_id(), U2_ID)
    assert_equal(itunes.search_artist('u2', store='ES')[0].get_url(), U2_URL_ES)

#LOOKUPS
def test_lookup_track():
    item = itunes.lookup(U2_ONE_ID)
    assert_true(isinstance(item, itunes.Track))
    assert_equal(item.get_id(), U2_ONE_ID)
    assert_equal(item.get_name(), U2_ONE)

    assert_equal(item.get_album().get_id(), U2_ACHTUNGBABY_ID)
    assert_equal(item.get_artist().get_id(), U2_ID)

def test_lookup_album():
    item = itunes.lookup(U2_ACHTUNGBABY_ID)
    assert_true(isinstance(item, itunes.Album))
    assert_equal(item.get_id(), U2_ACHTUNGBABY_ID)
    assert_equal(item.get_name(), U2_ACHTUNGBABY)

    assert_equal(item.get_artist().get_id(), U2_ID)

def test_lookup_artist():
    item = itunes.lookup(U2_ID)
    assert_true(isinstance(item, itunes.Artist))
    assert_equal(item.get_id(), U2_ID)
    assert_equal(item.get_name(), U2)

def test_lookup_notfound():
    UNKNOWN_ID = 0
    assert_raises(itunes.ServiceException, itunes.lookup, UNKNOWN_ID)

#METHODS
def test_artist_url():
    item = itunes.lookup(U2_ID)
    assert_equal(item.get_url(), U2_URL)

def test_album_url():
    item = itunes.lookup(U2_ACHTUNGBABY_ID)
    assert_equal(item.get_url(), U2_ACHTUNGBABY_URL)

def test_track_url():
    item = itunes.lookup(U2_ONE_ID)
    assert_equal(item.get_url(), U2_ONE_URL)

def test_album_length():
    item = itunes.lookup(U2_ACHTUNGBABY_ID)
    assert_true(len(item.get_tracks()) == 26) # 12)

#TEXT: Unicode
def test_unicode():
    assert_equal(itunes.search_artist('Björk')[0].get_id(), itunes.search_artist(u'Bj\xf6rk')[0].get_id())

def test_unicode2():
    assert_equal(itunes.search_artist('Björk')[:5], itunes.search_artist(u'Bj\xf6rk')[:5])

def test_movie_as_track():
    item = itunes.search(query='the godfather', media='movie')[0]
    assert_equal(item.get_artist(), None)


