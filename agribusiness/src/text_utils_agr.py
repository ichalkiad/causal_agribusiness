import json
import logging
import pickle
from argparse import ArgumentParser
import numpy as np
from plotly import io as pio
from scipy import sparse as sp
import matplotlib.pyplot as plt
import string
from flashtext import KeywordProcessor
import re
from collections import OrderedDict
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from spacy.vocab import Vocab
from spacy.tokens import Doc
import warnings
import html
import string
import unicodedata
from plotly import graph_objects as go

POS_EMOJIS = [u'😂', u'❤', u'♥', u'😍', u'😘', u'😊', u'👌', u'💕',
              u'👏', u'😁', u'☺', u'♡', u'👍', u'✌', u'😏', u'😉', u'🙌', u'😄']
NEG_EMOJIS = [u'😭', u'😩', u'😒', u'😔', u'😱']
NEUTRAL_EMOJIS = [u'🙏']

NORMALIZE_RE = re.compile(r"([a-zA-Z])\1\1+")
ALPHA_DIGITS_RE = re.compile(r"[a-zA-Z0-9_]+")
QUOTES_RE = re.compile(r'^".*"$')
BREAKS_RE = re.compile(r"[\r\n]+")
URLS_RE = re.compile(r"http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
UTF_CHARS = r'a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff'
TAG_EXP = r'(^|[^0-9A-Z&/]+)(#|\uff03)([0-9A-Z_]*[A-Z_]+[%s]*)' % UTF_CHARS
HASHTAGS_RE = re.compile(TAG_EXP, re.UNICODE | re.IGNORECASE)
URL_SHORTENERS = ['t', 'bit', 'goo', 'tinyurl']

# Order of patterns in DECONTRACTION matters as they are matched in the order they appear here
DECONTRACTIONS = OrderedDict([("ain't", "am not"), ("won't", "will not"), ("can't", "cannot"),
                              ("n't", " not"), ("'re", " are"), ("'s", " is"),
                              ("'d", " would"), ("'ll", " will"),
                              ("'t", " not"), ("'ve", " have"),
                              ("'m", " am")])

EMOJIS_UTF = {"\\xf0\\x9f\\x94\\xae": "\ud83d\udd2e", "\\xf0\\x9f\\x93\\xb0": "\ud83d\udcf0",
              "\\xf0\\x9f\\x94\\xb4": "\ud83d\udd34", "\\xe2\\x99\\x90": "\u2650",
              "\\xf0\\x9f\\x8d\\x81": "\ud83c\udf41", "\\xf0\\x9f\\x86\\x9a": "\ud83c\udd9a",
              "\\xe2\\x8f\\xb3": "\u23f3", "\\xf0\\x9f\\x98\\xaf": "\ud83d\ude2f",
              "\\xf0\\x9f\\x93\\xa7": "\ud83d\udce7", "\\xf0\\x9f\\x8d\\x8a": "\ud83c\udf4a",
              "\\xf0\\x9f\\x9a\\x9e": "\ud83d\ude9e", "\\xf0\\x9f\\x93\\xae": "\ud83d\udcee",
              "\\xf0\\x9f\\x94\\x93": "\ud83d\udd13", "\\xe2\\x9b\\xb5": "\u26f5",
              "\\xf0\\x9f\\x9a\\xa3": "\ud83d\udea3", "\\xf0\\x9f\\x91\\xa0": "\ud83d\udc60",
              "\\xf0\\x9f\\x9a\\x92": "\ud83d\ude92", "\\xe2\\x98\\xba": "\u263a",
              "\\xf0\\x9f\\x99\\x8f": "\ud83d\ude4f", "\\xf0\\x9f\\x8d\\x94": "\ud83c\udf54",
              "\\xf0\\x9f\\x94\\x9e": "\ud83d\udd1e", "\\xf0\\x9f\\x91\\xb6": "\ud83d\udc76",
              "\\xf0\\x9f\\x91\\xb3": "\ud83d\udc73", "\\xf0\\x9f\\x9a\\x9a": "\ud83d\ude9a",
              "\\xf0\\x9f\\x85\\xb0": "\ud83c\udd70", "\\xf0\\x9f\\x90\\x98": "\ud83d\udc18",
              "\\xf0\\x9f\\x93\\x91": "\ud83d\udcd1", "\\xf0\\x9f\\x93\\x9a": "\ud83d\udcda",
              "\\xf0\\x9f\\x91\\x80": "\ud83d\udc40", "\\xf0\\x9f\\x94\\x80": "\ud83d\udd00",
              "\\xf0\\x9f\\x8c\\x94": "\ud83c\udf14", "\\xf0\\x9f\\x9b\\x80": "\ud83d\udec0",
              "\\xf0\\x9f\\x90\\xbc": "\ud83d\udc3c", "\\xe2\\x99\\x8c": "\u264c",
              "\\xf0\\x9f\\x91\\xba": "\ud83d\udc7a", "\\x38\\xe2\\x83\\xa3": "8\u20e3", "\\xe2\\x99\\x91": "\u2651",
              "\\xf0\\x9f\\x8c\\x89": "\ud83c\udf09", "\\xf0\\x9f\\x98\\x96": "\ud83d\ude16",
              "\\xe2\\x9d\\x8c": "\u274c", "\\xf0\\x9f\\x98\\x98": "\ud83d\ude18",
              "\\xf0\\x9f\\x93\\x96": "\ud83d\udcd6", "\\xf0\\x9f\\x98\\x94": "\ud83d\ude14",
              "\\xf0\\x9f\\x91\\x9b": "\ud83d\udc5b", "\\x36\\xe2\\x83\\xa3": "6\u20e3",
              "\\xf0\\x9f\\x99\\x8e": "\ud83d\ude4e", "\\xf0\\x9f\\x91\\xad": "\ud83d\udc6d",
              "\\xf0\\x9f\\x98\\xb7": "\ud83d\ude37", "\\xf0\\x9f\\x98\\xaa": "\ud83d\ude2a",
              "\\xf0\\x9f\\x93\\x94": "\ud83d\udcd4",
              "\\xf0\\x9f\\x87\\xb0\\xf0\\x9f\\x87\\xb7": "\ud83c\uddf0\ud83c\uddf7",
              "\\xf0\\x9f\\x8e\\xbb": "\ud83c\udfbb", "\\xf0\\x9f\\x91\\xa9": "\ud83d\udc69",
              "\\xe2\\x86\\x96": "\u2196", "\\xf0\\x9f\\x80\\x84": "\ud83c\udc04",
              "\\xf0\\x9f\\x8d\\x93": "\ud83c\udf53", "\\xf0\\x9f\\x93\\xa4": "\ud83d\udce4",
              "\\xe2\\xad\\x90": "\u2b50", "\\xf0\\x9f\\x8d\\x90": "\ud83c\udf50",
              "\\xf0\\x9f\\x98\\x84": "\ud83d\ude04", "\\xf0\\x9f\\x93\\x98": "\ud83d\udcd8",
              "\\xf0\\x9f\\x91\\x92": "\ud83d\udc52", "\\xf0\\x9f\\x8d\\xaa": "\ud83c\udf6a",
              "\\xf0\\x9f\\x93\\xa5": "\ud83d\udce5", "\\xf0\\x9f\\x9a\\xa9": "\ud83d\udea9",
              "\\xf0\\x9f\\x87\\xae\\xf0\\x9f\\x87\\xb9": "\ud83c\uddee\ud83c\uddf9",
              "\\xf0\\x9f\\x90\\x84": "\ud83d\udc04", "\\xe2\\x9c\\x8c": "\u270c",
              "\\xf0\\x9f\\x8d\\xa3": "\ud83c\udf63", "\\xf0\\x9f\\x92\\xae": "\ud83d\udcae",
              "\\xf0\\x9f\\x8f\\x80": "\ud83c\udfc0", "\\xe3\\x80\\xbd": "\u303d",
              "\\xf0\\x9f\\x8c\\x83": "\ud83c\udf03", "\\xf0\\x9f\\x8d\\xba": "\ud83c\udf7a",
              "\\xe3\\x8a\\x99": "\u3299", "\\xe2\\x9b\\xbd": "\u26fd", "\\xe2\\x99\\x8b": "\u264b",
              "\\xe2\\x98\\x80": "\u2600", "\\xf0\\x9f\\x95\\xa6": "\ud83d\udd66",
              "\\xf0\\x9f\\x90\\xb4": "\ud83d\udc34", "\\xe2\\x9a\\xa1": "\u26a1",
              "\\xf0\\x9f\\x92\\xb0": "\ud83d\udcb0", "\\xf0\\x9f\\x98\\xbe": "\ud83d\ude3e",
              "\\xf0\\x9f\\x95\\x99": "\ud83d\udd59", "\\xf0\\x9f\\x92\\xb5": "\ud83d\udcb5",
              "\\xe2\\x9c\\xb3": "\u2733", "\\xf0\\x9f\\x90\\xb8": "\ud83d\udc38",
              "\\xf0\\x9f\\x94\\x82": "\ud83d\udd02", "\\xf0\\x9f\\x9a\\x83": "\ud83d\ude83",
              "\\xf0\\x9f\\x8c\\xbc": "\ud83c\udf3c", "\\xe2\\x99\\x8a": "\u264a",
              "\\xf0\\x9f\\x92\\xa4": "\ud83d\udca4", "\\xf0\\x9f\\x8c\\x96": "\ud83c\udf16",
              "\\xf0\\x9f\\x94\\xb5": "\ud83d\udd35", "\\xf0\\x9f\\x90\\x9e": "\ud83d\udc1e",
              "\\xf0\\x9f\\x94\\xb2": "\ud83d\udd32",
              "\\xf0\\x9f\\x87\\xac\\xf0\\x9f\\x87\\xa7": "\ud83c\uddec\ud83c\udde7",
              "\\xf0\\x9f\\x90\\x80": "\ud83d\udc00", "\\xf0\\x9f\\x9a\\x97": "\ud83d\ude97",
              "\\xf0\\x9f\\x99\\x8d": "\ud83d\ude4d", "\\xf0\\x9f\\x94\\xbb": "\ud83d\udd3b",
              "\\xf0\\x9f\\x98\\x9c": "\ud83d\ude1c", "\\xf0\\x9f\\x92\\xb6": "\ud83d\udcb6",
              "\\xf0\\x9f\\x92\\x85": "\ud83d\udc85", "\\xf0\\x9f\\x90\\x8d": "\ud83d\udc0d",
              "\\xf0\\x9f\\x9a\\x81": "\ud83d\ude81", "\\xf0\\x9f\\x8f\\xa7": "\ud83c\udfe7",
              "\\xf0\\x9f\\x92\\x97": "\ud83d\udc97", "\\xf0\\x9f\\x94\\xb1": "\ud83d\udd31",
              "\\x33\\xe2\\x83\\xa3": "3\u20e3", "\\xf0\\x9f\\x91\\xac": "\ud83d\udc6c",
              "\\xf0\\x9f\\x91\\x8c": "\ud83d\udc4c", "\\xf0\\x9f\\x8d\\x88": "\ud83c\udf48",
              "\\xf0\\x9f\\x90\\x94": "\ud83d\udc14", "\\xf0\\x9f\\x8d\\xb4": "\ud83c\udf74",
              "\\xf0\\x9f\\x8d\\xb3": "\ud83c\udf73", "\\xf0\\x9f\\x9a\\x82": "\ud83d\ude82",
              "\\xe2\\x9a\\xbe": "\u26be", "\\xf0\\x9f\\x9b\\x81": "\ud83d\udec1",
              "\\xf0\\x9f\\x8f\\x83": "\ud83c\udfc3", "\\xe2\\x8f\\xa9": "\u23e9",
              "\\xf0\\x9f\\x8d\\xa7": "\ud83c\udf67", "\\xf0\\x9f\\x95\\x95": "\ud83d\udd55",
              "\\xf0\\x9f\\x93\\xb2": "\ud83d\udcf2", "\\xf0\\x9f\\x92\\x8e": "\ud83d\udc8e",
              "\\xf0\\x9f\\x99\\x8c": "\ud83d\ude4c", "\\xf0\\x9f\\x95\\x97": "\ud83d\udd57",
              "\\xf0\\x9f\\x91\\xa7": "\ud83d\udc67", "\\xf0\\x9f\\x98\\xad": "\ud83d\ude2d",
              "\\xe2\\x86\\x99": "\u2199", "\\xf0\\x9f\\x8c\\x95": "\ud83c\udf15",
              "\\xf0\\x9f\\x92\\x95": "\ud83d\udc95", "\\xf0\\x9f\\x98\\xb5": "\ud83d\ude35",
              "\\xe2\\xa4\\xb5": "\u2935", "\\xf0\\x9f\\x8f\\xa2": "\ud83c\udfe2",
              "\\xf0\\x9f\\x8c\\xb0": "\ud83c\udf30", "\\xf0\\x9f\\x93\\x85": "\ud83d\udcc5",
              "\\xf0\\x9f\\x90\\xbd": "\ud83d\udc3d", "\\xf0\\x9f\\x8d\\x8d": "\ud83c\udf4d",
              "\\xf0\\x9f\\x8d\\xb2": "\ud83c\udf72", "\\xf0\\x9f\\x9a\\xbd": "\ud83d\udebd",
              "\\xf0\\x9f\\x93\\x92": "\ud83d\udcd2", "\\xf0\\x9f\\x88\\xb2": "\ud83c\ude32",
              "\\xf0\\x9f\\x92\\xa1": "\ud83d\udca1", "\\xf0\\x9f\\x90\\x88": "\ud83d\udc08",
              "\\xf0\\x9f\\x91\\x89": "\ud83d\udc49", "\\xf0\\x9f\\x90\\x8b": "\ud83d\udc0b",
              "\\xf0\\x9f\\x94\\x8b": "\ud83d\udd0b", "\\xf0\\x9f\\x8f\\xac": "\ud83c\udfec",
              "\\xf0\\x9f\\x91\\x93": "\ud83d\udc53", "\\xe2\\x9d\\x87": "\u2747",
              "\\xf0\\x9f\\x90\\x85": "\ud83d\udc05", "\\xf0\\x9f\\x95\\x93": "\ud83d\udd53",
              "\\xf0\\x9f\\x94\\xb7": "\ud83d\udd37", "\\xf0\\x9f\\x93\\xba": "\ud83d\udcfa",
              "\\xf0\\x9f\\x8e\\xbc": "\ud83c\udfbc", "\\xf0\\x9f\\x9a\\xa0": "\ud83d\udea0",
              "\\xe2\\x98\\x91": "\u2611", "\\xf0\\x9f\\x98\\x8d": "\ud83d\ude0d",
              "\\xf0\\x9f\\x92\\x9e": "\ud83d\udc9e", "\\xf0\\x9f\\x8d\\xab": "\ud83c\udf6b",
              "\\xf0\\x9f\\x91\\xb7": "\ud83d\udc77", "\\xf0\\x9f\\x92\\xaf": "\ud83d\udcaf",
              "\\xf0\\x9f\\x94\\xa1": "\ud83d\udd21", "\\xe2\\x96\\xaa": "\u25aa", "\\x37\\xe2\\x83\\xa3": "7\u20e3",
              "\\xf0\\x9f\\x8f\\x84": "\ud83c\udfc4", "\\xf0\\x9f\\x88\\xb6": "\ud83c\ude36",
              "\\x31\\xe2\\x83\\xa3": "1\u20e3", "\\xf0\\x9f\\x83\\x8f": "\ud83c\udccf",
              "\\xf0\\x9f\\x93\\x9b": "\ud83d\udcdb", "\\xf0\\x9f\\x9a\\xb6": "\ud83d\udeb6",
              "\\xe2\\x8f\\xaa": "\u23ea", "\\xf0\\x9f\\x8f\\xa6": "\ud83c\udfe6",
              "\\xf0\\x9f\\x99\\x89": "\ud83d\ude49", "\\xf0\\x9f\\x89\\x91": "\ud83c\ude51",
              "\\xf0\\x9f\\x94\\x95": "\ud83d\udd15", "\\xf0\\x9f\\x88\\xb3": "\ud83c\ude33",
              "\\xf0\\x9f\\x93\\x80": "\ud83d\udcc0", "\\xf0\\x9f\\x8c\\x85": "\ud83c\udf05",
              "\\xf0\\x9f\\x8d\\x97": "\ud83c\udf57", "\\xf0\\x9f\\x90\\xad": "\ud83d\udc2d",
              "\\xf0\\x9f\\x98\\xa8": "\ud83d\ude28", "\\xf0\\x9f\\x85\\xbf": "\ud83c\udd7f",
              "\\xf0\\x9f\\x9a\\x96": "\ud83d\ude96", "\\xf0\\x9f\\x90\\x83": "\ud83d\udc03",
              "\\xf0\\x9f\\x92\\x9a": "\ud83d\udc9a", "\\xf0\\x9f\\x90\\xa8": "\ud83d\udc28",
              "\\xe2\\x99\\xa6": "\u2666", "\\xf0\\x9f\\x98\\x8b": "\ud83d\ude0b",
              "\\xf0\\x9f\\x92\\x88": "\ud83d\udc88", "\\xf0\\x9f\\x94\\x83": "\ud83d\udd03",
              "\\xf0\\x9f\\x98\\x83": "\ud83d\ude03", "\\xf0\\x9f\\x98\\x8a": "\ud83d\ude0a",
              "\\xe2\\x9b\\xb3": "\u26f3", "\\xf0\\x9f\\x90\\xb1": "\ud83d\udc31", "\\xe2\\x99\\xa3": "\u2663",
              "\\xf0\\x9f\\x92\\xb7": "\ud83d\udcb7", "\\xf0\\x9f\\x92\\xb4": "\ud83d\udcb4",
              "\\xf0\\x9f\\x91\\xb4": "\ud83d\udc74", "\\xf0\\x9f\\x8e\\xbd": "\ud83c\udfbd",
              "\\xf0\\x9f\\x90\\x92": "\ud83d\udc12", "\\xf0\\x9f\\x8e\\xb1": "\ud83c\udfb1",
              "\\xf0\\x9f\\x8c\\xb3": "\ud83c\udf33", "\\xf0\\x9f\\x8c\\x84": "\ud83c\udf04",
              "\\xf0\\x9f\\x98\\x8c": "\ud83d\ude0c", "\\xf0\\x9f\\x92\\x87": "\ud83d\udc87",
              "\\xe2\\x9a\\x93": "\u2693", "\\xf0\\x9f\\x8f\\xa5": "\ud83c\udfe5",
              "\\xf0\\x9f\\x9a\\xab": "\ud83d\udeab", "\\xe2\\x9c\\x82": "\u2702",
              "\\xf0\\x9f\\x8f\\x81": "\ud83c\udfc1", "\\xf0\\x9f\\x98\\xb8": "\ud83d\ude38",
              "\\xf0\\x9f\\x94\\xab": "\ud83d\udd2b", "\\xf0\\x9f\\x9a\\xbb": "\ud83d\udebb",
              "\\xf0\\x9f\\x98\\xa0": "\ud83d\ude20", "\\xf0\\x9f\\x8f\\x89": "\ud83c\udfc9",
              "\\xf0\\x9f\\x98\\xbc": "\ud83d\ude3c", "\\xf0\\x9f\\x8d\\x9b": "\ud83c\udf5b",
              "\\xe2\\x9e\\xa1": "\u27a1", "\\xf0\\x9f\\x92\\x8c": "\ud83d\udc8c",
              "\\xf0\\x9f\\x94\\x81": "\ud83d\udd01", "\\xf0\\x9f\\x8f\\xae": "\ud83c\udfee",
              "\\xf0\\x9f\\x8f\\xaa": "\ud83c\udfea", "\\xf0\\x9f\\x98\\xbf": "\ud83d\ude3f",
              "\\xf0\\x9f\\x93\\x8b": "\ud83d\udccb", "\\xf0\\x9f\\x99\\x88": "\ud83d\ude48",
              "\\xf0\\x9f\\x8f\\x87": "\ud83c\udfc7", "\\xf0\\x9f\\x94\\x99": "\ud83d\udd19",
              "\\xf0\\x9f\\x90\\xbe": "\ud83d\udc3e", "\\xf0\\x9f\\x98\\xb1": "\ud83d\ude31",
              "\\xf0\\x9f\\x8d\\x8c": "\ud83c\udf4c", "\\xf0\\x9f\\x91\\x86": "\ud83d\udc46",
              "\\xf0\\x9f\\x93\\x9d": "\ud83d\udcdd", "\\xf0\\x9f\\x91\\x90": "\ud83d\udc50",
              "\\xf0\\x9f\\x90\\x90": "\ud83d\udc10", "\\xe2\\x99\\x89": "\u2649",
              "\\xf0\\x9f\\x92\\x84": "\ud83d\udc84", "\\xf0\\x9f\\x90\\xa2": "\ud83d\udc22",
              "\\xf0\\x9f\\x8c\\x8e": "\ud83c\udf0e", "\\xf0\\x9f\\x86\\x96": "\ud83c\udd96",
              "\\xf0\\x9f\\x91\\x87": "\ud83d\udc47", "\\xf0\\x9f\\x98\\xa2": "\ud83d\ude22",
              "\\xe2\\x9c\\x8a": "\u270a", "\\xf0\\x9f\\x92\\x9f": "\ud83d\udc9f",
              "\\xf0\\x9f\\x8c\\x86": "\ud83c\udf06", "\\xf0\\x9f\\x8e\\xa0": "\ud83c\udfa0",
              "\\xf0\\x9f\\x8c\\x98": "\ud83c\udf18", "\\xf0\\x9f\\x8d\\xa0": "\ud83c\udf60",
              "\\xf0\\x9f\\x92\\x83": "\ud83d\udc83", "\\xe2\\x86\\x95": "\u2195",
              "\\xf0\\x9f\\x98\\x82": "\ud83d\ude02", "\\xf0\\x9f\\x8d\\xb9": "\ud83c\udf79",
              "\\xf0\\x9f\\x92\\xa6": "\ud83d\udca6", "\\xe2\\xac\\x86": "\u2b06",
              "\\xf0\\x9f\\x9a\\x84": "\ud83d\ude84", "\\xf0\\x9f\\x9a\\xbe": "\ud83d\udebe",
              "\\xf0\\x9f\\x91\\xaa": "\ud83d\udc6a", "\\xf0\\x9f\\x8d\\xb8": "\ud83c\udf78",
              "\\xf0\\x9f\\x8e\\xa3": "\ud83c\udfa3", "\\xf0\\x9f\\x86\\x99": "\ud83c\udd99",
              "\\xf0\\x9f\\x94\\xaf": "\ud83d\udd2f", "\\xf0\\x9f\\x93\\x8e": "\ud83d\udcce",
              "\\xe2\\x93\\x82": "\u24c2", "\\xf0\\x9f\\x88\\xb5": "\ud83c\ude35",
              "\\xf0\\x9f\\x9a\\xa5": "\ud83d\udea5", "\\xf0\\x9f\\x93\\xb7": "\ud83d\udcf7",
              "\\xf0\\x9f\\x94\\x84": "\ud83d\udd04", "\\xf0\\x9f\\x8e\\x83": "\ud83c\udf83",
              "\\xf0\\x9f\\x98\\x8e": "\ud83d\ude0e", "\\xf0\\x9f\\x8e\\xb5": "\ud83c\udfb5",
              "\\xf0\\x9f\\x98\\xbb": "\ud83d\ude3b", "\\xe2\\x99\\x8d": "\u264d",
              "\\xf0\\x9f\\x92\\x8b": "\ud83d\udc8b", "\\xf0\\x9f\\x90\\x93": "\ud83d\udc13",
              "\\xf0\\x9f\\x94\\xac": "\ud83d\udd2c", "\\xc2\\xa9": "\u00a9", "\\xf0\\x9f\\x9a\\xa2": "\ud83d\udea2",
              "\\xf0\\x9f\\x8e\\xb7": "\ud83c\udfb7", "\\xf0\\x9f\\x93\\xb1": "\ud83d\udcf1",
              "\\xf0\\x9f\\x8e\\x89": "\ud83c\udf89", "\\xf0\\x9f\\x92\\xb1": "\ud83d\udcb1",
              "\\xf0\\x9f\\x89\\x90": "\ud83c\ude50", "\\xf0\\x9f\\x8d\\x89": "\ud83c\udf49",
              "\\xf0\\x9f\\x9a\\xb1": "\ud83d\udeb1", "\\xf0\\x9f\\x95\\x9e": "\ud83d\udd5e",
              "\\xf0\\x9f\\x94\\xb9": "\ud83d\udd39", "\\xe2\\x9c\\x94": "\u2714",
              "\\xf0\\x9f\\x8c\\x92": "\ud83c\udf12", "\\xf0\\x9f\\x9a\\xb5": "\ud83d\udeb5",
              "\\xf0\\x9f\\x93\\x89": "\ud83d\udcc9", "\\xf0\\x9f\\x95\\x9f": "\ud83d\udd5f",
              "\\xf0\\x9f\\x98\\x80": "\ud83d\ude00", "\\xf0\\x9f\\x8d\\xaf": "\ud83c\udf6f",
              "\\xf0\\x9f\\x90\\xae": "\ud83d\udc2e", "\\xf0\\x9f\\x8e\\xa4": "\ud83c\udfa4",
              "\\xf0\\x9f\\x93\\xb9": "\ud83d\udcf9", "\\xf0\\x9f\\x97\\xbb": "\ud83d\uddfb",
              "\\xf0\\x9f\\x94\\x97": "\ud83d\udd17", "\\xf0\\x9f\\x9a\\x98": "\ud83d\ude98",
              "\\xf0\\x9f\\x98\\xb3": "\ud83d\ude33", "\\xf0\\x9f\\x8e\\x87": "\ud83c\udf87",
              "\\xe2\\x9b\\x94": "\u26d4", "\\xf0\\x9f\\x8c\\x80": "\ud83c\udf00",
              "\\xf0\\x9f\\x8c\\x90": "\ud83c\udf10", "\\xf0\\x9f\\x98\\xa4": "\ud83d\ude24",
              "\\xf0\\x9f\\x9a\\xaa": "\ud83d\udeaa", "\\xf0\\x9f\\x8c\\x9a": "\ud83c\udf1a",
              "\\xf0\\x9f\\x8d\\xa2": "\ud83c\udf62", "\\xf0\\x9f\\x86\\x95": "\ud83c\udd95",
              "\\xf0\\x9f\\x91\\x8e": "\ud83d\udc4e", "\\xf0\\x9f\\x90\\xa1": "\ud83d\udc21",
              "\\xe2\\x9c\\x88": "\u2708", "\\xf0\\x9f\\x8d\\xa9": "\ud83c\udf69",
              "\\xf0\\x9f\\x8d\\xb0": "\ud83c\udf70", "\\xf0\\x9f\\x94\\xa4": "\ud83d\udd24",
              "\\xf0\\x9f\\x94\\xba": "\ud83d\udd3a", "\\xf0\\x9f\\x9a\\x88": "\ud83d\ude88",
              "\\xf0\\x9f\\x93\\x87": "\ud83d\udcc7", "\\xf0\\x9f\\x91\\xb5": "\ud83d\udc75",
              "\\xf0\\x9f\\x98\\x88": "\ud83d\ude08", "\\xf0\\x9f\\x90\\x8c": "\ud83d\udc0c",
              "\\xf0\\x9f\\x8e\\xaa": "\ud83c\udfaa", "\\xe2\\x99\\xbb": "\u267b",
              "\\xf0\\x9f\\x8c\\x8f": "\ud83c\udf0f", "\\xf0\\x9f\\x91\\xb0": "\ud83d\udc70",
              "\\xf0\\x9f\\x93\\xa1": "\ud83d\udce1", "\\xf0\\x9f\\x92\\xbf": "\ud83d\udcbf",
              "\\xf0\\x9f\\x8e\\x84": "\ud83c\udf84", "\\xf0\\x9f\\x90\\xa7": "\ud83d\udc27",
              "\\xf0\\x9f\\x9a\\x95": "\ud83d\ude95", "\\xf0\\x9f\\x8d\\x9f": "\ud83c\udf5f",
              "\\xe2\\x98\\x94": "\u2614", "\\xf0\\x9f\\x98\\x91": "\ud83d\ude11",
              "\\xf0\\x9f\\x94\\xbc": "\ud83d\udd3c", "\\xf0\\x9f\\x97\\xbc": "\ud83d\uddfc",
              "\\xf0\\x9f\\x98\\xa7": "\ud83d\ude27", "\\xf0\\x9f\\x91\\xb8": "\ud83d\udc78",
              "\\xf0\\x9f\\x9a\\xa8": "\ud83d\udea8", "\\xf0\\x9f\\x90\\x91": "\ud83d\udc11",
              "\\xf0\\x9f\\x92\\xa2": "\ud83d\udca2", "\\xf0\\x9f\\x90\\xbb": "\ud83d\udc3b",
              "\\xf0\\x9f\\x93\\x84": "\ud83d\udcc4", "\\xf0\\x9f\\x98\\xb2": "\ud83d\ude32",
              "\\xf0\\x9f\\x92\\xa8": "\ud83d\udca8", "\\xf0\\x9f\\x91\\xbd": "\ud83d\udc7d",
              "\\xe2\\x9b\\x84": "\u26c4", "\\xf0\\x9f\\x91\\xb1": "\ud83d\udc71",
              "\\xf0\\x9f\\x8d\\xa8": "\ud83c\udf68", "\\xf0\\x9f\\x98\\x9d": "\ud83d\ude1d",
              "\\xf0\\x9f\\x8e\\xa1": "\ud83c\udfa1", "\\xf0\\x9f\\x9a\\xb0": "\ud83d\udeb0",
              "\\xf0\\x9f\\x91\\xa5": "\ud83d\udc65", "\\xe2\\x86\\xa9": "\u21a9",
              "\\xf0\\x9f\\x9a\\x93": "\ud83d\ude93", "\\xf0\\x9f\\x92\\x98": "\ud83d\udc98",
              "\\xf0\\x9f\\x99\\x85": "\ud83d\ude45", "\\xf0\\x9f\\x90\\x9c": "\ud83d\udc1c",
              "\\xf0\\x9f\\x92\\x9b": "\ud83d\udc9b", "\\xf0\\x9f\\x91\\x91": "\ud83d\udc51",
              "\\xf0\\x9f\\x8d\\x99": "\ud83c\udf59", "\\xe2\\x99\\x88": "\u2648", "\\xe2\\x9d\\x94": "\u2754",
              "\\xf0\\x9f\\x8c\\x88": "\ud83c\udf08", "\\xe2\\x86\\xaa": "\u21aa", "\\x32\\xe2\\x83\\xa3": "2\u20e3",
              "\\xf0\\x9f\\x88\\xb7": "\ud83c\ude37", "\\xf0\\x9f\\x8c\\x99": "\ud83c\udf19",
              "\\xf0\\x9f\\x8c\\xbb": "\ud83c\udf3b", "\\xf0\\x9f\\x92\\x94": "\ud83d\udc94",
              "\\xe2\\x9c\\xa8": "\u2728", "\\xf0\\x9f\\x91\\x9f": "\ud83d\udc5f",
              "\\xf0\\x9f\\x91\\xa4": "\ud83d\udc64", "\\xf0\\x9f\\x91\\xaf": "\ud83d\udc6f",
              "\\xf0\\x9f\\x98\\x9e": "\ud83d\ude1e", "\\xf0\\x9f\\x93\\xa9": "\ud83d\udce9",
              "\\xe2\\x9b\\x8e": "\u26ce", "\\xf0\\x9f\\x92\\x99": "\ud83d\udc99",
              "\\xf0\\x9f\\x98\\x95": "\ud83d\ude15", "\\xf0\\x9f\\x8e\\x85": "\ud83c\udf85",
              "\\xf0\\x9f\\x90\\x99": "\ud83d\udc19", "\\xf0\\x9f\\x92\\xa7": "\ud83d\udca7",
              "\\xf0\\x9f\\x8d\\xa1": "\ud83c\udf61", "\\xe2\\x99\\x92": "\u2652",
              "\\xf0\\x9f\\x98\\xa5": "\ud83d\ude25", "\\xf0\\x9f\\x92\\x8f": "\ud83d\udc8f",
              "\\xf0\\x9f\\x90\\xa3": "\ud83d\udc23", "\\xf0\\x9f\\x93\\xb3": "\ud83d\udcf3",
              "\\xf0\\x9f\\x92\\xaa": "\ud83d\udcaa", "\\xf0\\x9f\\x94\\x9d": "\ud83d\udd1d",
              "\\xf0\\x9f\\x8d\\xa5": "\ud83c\udf65", "\\xf0\\x9f\\x88\\xb8": "\ud83c\ude38",
              "\\xe2\\x9e\\x96": "\u2796", "\\xe2\\x9d\\x95": "\u2755", "\\xf0\\x9f\\x91\\x8a": "\ud83d\udc4a",
              "\\xf0\\x9f\\x92\\xab": "\ud83d\udcab", "\\xf0\\x9f\\x90\\xba": "\ud83d\udc3a",
              "\\xf0\\x9f\\x8c\\xba": "\ud83c\udf3a", "\\xf0\\x9f\\x93\\x9e": "\ud83d\udcde",
              "\\xf0\\x9f\\x92\\x80": "\ud83d\udc80", "\\xf0\\x9f\\x9a\\xb7": "\ud83d\udeb7",
              "\\xf0\\x9f\\x8f\\x88": "\ud83c\udfc8", "\\xe2\\x9a\\xaa": "\u26aa",
              "\\xf0\\x9f\\x87\\xaa\\xf0\\x9f\\x87\\xb8": "\ud83c\uddea\ud83c\uddf8", "\\xe2\\x8f\\xb0": "\u23f0",
              "\\xf0\\x9f\\x86\\x8e": "\ud83c\udd8e", "\\xf0\\x9f\\x8d\\x8e": "\ud83c\udf4e",
              "\\xf0\\x9f\\x8f\\xa3": "\ud83c\udfe3", "\\xf0\\x9f\\x9a\\xb3": "\ud83d\udeb3",
              "\\xf0\\x9f\\x93\\x95": "\ud83d\udcd5", "\\xf0\\x9f\\x91\\xb9": "\ud83d\udc79",
              "\\xe2\\xac\\x87": "\u2b07", "\\xf0\\x9f\\x92\\x9c": "\ud83d\udc9c",
              "\\xf0\\x9f\\x93\\xa8": "\ud83d\udce8", "\\xf0\\x9f\\x95\\xa1": "\ud83d\udd61",
              "\\xf0\\x9f\\x9b\\x85": "\ud83d\udec5", "\\xf0\\x9f\\x9a\\xba": "\ud83d\udeba",
              "\\xf0\\x9f\\x93\\x9c": "\ud83d\udcdc", "\\xf0\\x9f\\x94\\x87": "\ud83d\udd07",
              "\\xe2\\x97\\xbd": "\u25fd", "\\xe2\\x97\\xbb": "\u25fb", "\\xf0\\x9f\\x94\\xa2": "\ud83d\udd22",
              "\\xf0\\x9f\\x9a\\x9b": "\ud83d\ude9b", "\\xf0\\x9f\\x90\\x9b": "\ud83d\udc1b",
              "\\xf0\\x9f\\x8e\\xaf": "\ud83c\udfaf", "\\xe2\\x9b\\x85": "\u26c5", "\\xe2\\x99\\xa8": "\u2668",
              "\\xf0\\x9f\\x8d\\x83": "\ud83c\udf43", "\\xe2\\x9c\\x89": "\u2709",
              "\\xf0\\x9f\\x98\\x8f": "\ud83d\ude0f", "\\xf0\\x9f\\x91\\x9a": "\ud83d\udc5a",
              "\\xf0\\x9f\\x94\\xad": "\ud83d\udd2d", "\\xe2\\x9a\\xab": "\u26ab",
              "\\xf0\\x9f\\x92\\x92": "\ud83d\udc92", "\\xf0\\x9f\\x94\\xa6": "\ud83d\udd26",
              "\\xe2\\x84\\xa2": "\u2122", "\\xf0\\x9f\\x91\\x95": "\ud83d\udc55",
              "\\xf0\\x9f\\x8c\\x87": "\ud83c\udf07", "\\xf0\\x9f\\x8d\\x82": "\ud83c\udf42",
              "\\xf0\\x9f\\x98\\x81": "\ud83d\ude01",
              "\\xf0\\x9f\\x87\\xa8\\xf0\\x9f\\x87\\xb3": "\ud83c\udde8\ud83c\uddf3",
              "\\xf0\\x9f\\x94\\xa7": "\ud83d\udd27", "\\xf0\\x9f\\x90\\x9d": "\ud83d\udc1d",
              "\\xf0\\x9f\\x90\\xb9": "\ud83d\udc39", "\\xf0\\x9f\\x8d\\x87": "\ud83c\udf47",
              "\\xf0\\x9f\\x95\\x91": "\ud83d\udd51", "\\xf0\\x9f\\x98\\xa6": "\ud83d\ude26",
              "\\xf0\\x9f\\x94\\xbd": "\ud83d\udd3d", "\\xf0\\x9f\\x9a\\x8d": "\ud83d\ude8d",
              "\\xf0\\x9f\\x87\\xb7\\xf0\\x9f\\x87\\xba": "\ud83c\uddf7\ud83c\uddfa",
              "\\xf0\\x9f\\x98\\xb6": "\ud83d\ude36", "\\xf0\\x9f\\x8e\\x8e": "\ud83c\udf8e",
              "\\xf0\\x9f\\x92\\xbe": "\ud83d\udcbe", "\\xf0\\x9f\\x94\\x9a": "\ud83d\udd1a",
              "\\xf0\\x9f\\x87\\xaf\\xf0\\x9f\\x87\\xb5": "\ud83c\uddef\ud83c\uddf5",
              "\\xf0\\x9f\\x8c\\xb7": "\ud83c\udf37", "\\xf0\\x9f\\x9a\\x80": "\ud83d\ude80", "\\xc2\\xae": "\u00ae",
              "\\xf0\\x9f\\x94\\x9f": "\ud83d\udd1f", "\\xf0\\x9f\\x8d\\x84": "\ud83c\udf44",
              "\\xf0\\x9f\\x93\\xbc": "\ud83d\udcfc", "\\xf0\\x9f\\x92\\xb9": "\ud83d\udcb9",
              "\\xe2\\x96\\xb6": "\u25b6", "\\xf0\\x9f\\x95\\x9c": "\ud83d\udd5c",
              "\\xf0\\x9f\\x8e\\xad": "\ud83c\udfad", "\\xf0\\x9f\\x9a\\xa6": "\ud83d\udea6",
              "\\xe2\\x99\\xa0": "\u2660", "\\xf0\\x9f\\x93\\x8c": "\ud83d\udccc",
              "\\xf0\\x9f\\x94\\x96": "\ud83d\udd16", "\\xf0\\x9f\\x92\\xb3": "\ud83d\udcb3",
              "\\xf0\\x9f\\x8e\\xac": "\ud83c\udfac", "\\xf0\\x9f\\x93\\xa6": "\ud83d\udce6",
              "\\xf0\\x9f\\x8d\\xb5": "\ud83c\udf75", "\\xf0\\x9f\\x8f\\xa0": "\ud83c\udfe0",
              "\\xf0\\x9f\\x85\\xb1": "\ud83c\udd71", "\\xf0\\x9f\\x8c\\x91": "\ud83c\udf11",
              "\\xf0\\x9f\\x8e\\x86": "\ud83c\udf86", "\\xf0\\x9f\\x91\\xa1": "\ud83d\udc61",
              "\\xf0\\x9f\\x98\\x93": "\ud83d\ude13", "\\xf0\\x9f\\x94\\x8c": "\ud83d\udd0c",
              "\\xf0\\x9f\\x92\\xbb": "\ud83d\udcbb", "\\xe2\\x99\\x8f": "\u264f",
              "\\xf0\\x9f\\x8d\\x9d": "\ud83c\udf5d", "\\xf0\\x9f\\x8d\\xbc": "\ud83c\udf7c",
              "\\xe2\\x9d\\x8e": "\u274e", "\\xf0\\x9f\\x94\\x8f": "\ud83d\udd0f",
              "\\xf0\\x9f\\x98\\x9f": "\ud83d\ude1f",
              "\\xf0\\x9f\\x87\\xab\\xf0\\x9f\\x87\\xb7": "\ud83c\uddeb\ud83c\uddf7",
              "\\xf0\\x9f\\x8f\\xa9": "\ud83c\udfe9", "\\xf0\\x9f\\x9a\\x86": "\ud83d\ude86",
              "\\xf0\\x9f\\x8d\\xa4": "\ud83c\udf64", "\\xf0\\x9f\\x98\\x97": "\ud83d\ude17",
              "\\xf0\\x9f\\x9a\\xb8": "\ud83d\udeb8", "\\xf0\\x9f\\x94\\x8a": "\ud83d\udd0a",
              "\\xf0\\x9f\\x8c\\x8b": "\ud83c\udf0b", "\\x30\\xe2\\x83\\xa3": "0\u20e3",
              "\\xf0\\x9f\\x9a\\x89": "\ud83d\ude89", "\\xe2\\xad\\x95": "\u2b55",
              "\\xf0\\x9f\\x93\\xaf": "\ud83d\udcef", "\\xf0\\x9f\\x86\\x91": "\ud83c\udd91",
              "\\xe2\\x9d\\xa4": "\u2764", "\\xf0\\x9f\\x91\\x9e": "\ud83d\udc5e", "\\xe3\\x8a\\x97": "\u3297",
              "\\xe2\\x9d\\x84": "\u2744", "\\xf0\\x9f\\x8d\\xb6": "\ud83c\udf76",
              "\\xf0\\x9f\\x98\\x9b": "\ud83d\ude1b", "\\xf0\\x9f\\x93\\xa0": "\ud83d\udce0",
              "\\xf0\\x9f\\x9a\\xad": "\ud83d\udead", "\\xf0\\x9f\\x95\\xa0": "\ud83d\udd60",
              "\\xf0\\x9f\\x93\\x88": "\ud83d\udcc8", "\\xf0\\x9f\\x8f\\xa1": "\ud83c\udfe1",
              "\\xf0\\x9f\\x93\\xb6": "\ud83d\udcf6", "\\xf0\\x9f\\x91\\xbc": "\ud83d\udc7c",
              "\\xf0\\x9f\\x9a\\xb2": "\ud83d\udeb2", "\\xf0\\x9f\\x94\\xb8": "\ud83d\udd38",
              "\\xf0\\x9f\\x9a\\xb4": "\ud83d\udeb4", "\\xf0\\x9f\\x92\\x8a": "\ud83d\udc8a",
              "\\xf0\\x9f\\x90\\x97": "\ud83d\udc17", "\\xf0\\x9f\\x86\\x93": "\ud83c\udd93",
              "\\xe2\\x96\\xab": "\u25ab", "\\xf0\\x9f\\x90\\xa5": "\ud83d\udc25",
              "\\xf0\\x9f\\x93\\x86": "\ud83d\udcc6", "\\xe2\\x98\\x95": "\u2615",
              "\\xf0\\x9f\\x8d\\xb1": "\ud83c\udf71", "\\xf0\\x9f\\x8c\\xbe": "\ud83c\udf3e",
              "\\xf0\\x9f\\x91\\x9d": "\ud83d\udc5d", "\\xf0\\x9f\\x91\\x9c": "\ud83d\udc5c",
              "\\xe2\\x97\\xbc": "\u25fc", "\\xf0\\x9f\\x92\\x81": "\ud83d\udc81",
              "\\xf0\\x9f\\x98\\x85": "\ud83d\ude05", "\\xf0\\x9f\\x8d\\x95": "\ud83c\udf55",
              "\\xf0\\x9f\\x92\\xb8": "\ud83d\udcb8", "\\xf0\\x9f\\x86\\x94": "\ud83c\udd94",
              "\\xf0\\x9f\\x97\\xbe": "\ud83d\uddfe", "\\xf0\\x9f\\x9a\\x91": "\ud83d\ude91",
              "\\xf0\\x9f\\x9a\\xbc": "\ud83d\udebc", "\\xf0\\x9f\\x92\\xa9": "\ud83d\udca9",
              "\\xf0\\x9f\\x98\\xab": "\ud83d\ude2b", "\\xf0\\x9f\\x8d\\xac": "\ud83c\udf6c",
              "\\xf0\\x9f\\x91\\xb2": "\ud83d\udc72", "\\xe2\\x8c\\x9a": "\u231a",
              "\\xf0\\x9f\\x90\\xaa": "\ud83d\udc2a", "\\xf0\\x9f\\x92\\x86": "\ud83d\udc86",
              "\\xf0\\x9f\\x9a\\x9c": "\ud83d\ude9c", "\\xf0\\x9f\\x8d\\x98": "\ud83c\udf58",
              "\\xf0\\x9f\\x94\\x8d": "\ud83d\udd0d", "\\xf0\\x9f\\x9b\\x84": "\ud83d\udec4",
              "\\xf0\\x9f\\x90\\x8f": "\ud83d\udc0f", "\\xf0\\x9f\\x94\\xb6": "\ud83d\udd36",
              "\\xf0\\x9f\\x8c\\xb5": "\ud83c\udf35", "\\xf0\\x9f\\x91\\x96": "\ud83d\udc56",
              "\\xf0\\x9f\\x88\\xb4": "\ud83c\ude34", "\\xf0\\x9f\\x98\\x87": "\ud83d\ude07",
              "\\xf0\\x9f\\x8c\\xa0": "\ud83c\udf20", "\\xf0\\x9f\\x98\\xac": "\ud83d\ude2c",
              "\\xf0\\x9f\\x87\\xba\\xf0\\x9f\\x87\\xb8": "\ud83c\uddfa\ud83c\uddf8", "\\xe2\\x99\\xa5": "\u2665",
              "\\xf0\\x9f\\x8d\\x86": "\ud83c\udf46", "\\xf0\\x9f\\x91\\xbb": "\ud83d\udc7b",
              "\\xf0\\x9f\\x8e\\xb2": "\ud83c\udfb2", "\\xf0\\x9f\\x8e\\xa8": "\ud83c\udfa8",
              "\\xf0\\x9f\\x90\\xab": "\ud83d\udc2b", "\\xf0\\x9f\\x90\\x86": "\ud83d\udc06",
              "\\xf0\\x9f\\x8c\\x9d": "\ud83c\udf1d", "\\xf0\\x9f\\x8d\\x9e": "\ud83c\udf5e",
              "\\xf0\\x9f\\x8c\\xbd": "\ud83c\udf3d", "\\xf0\\x9f\\x8e\\xb3": "\ud83c\udfb3",
              "\\xf0\\x9f\\x95\\x9b": "\ud83d\udd5b", "\\xf0\\x9f\\x90\\x87": "\ud83d\udc07",
              "\\xf0\\x9f\\x98\\x89": "\ud83d\ude09", "\\xf0\\x9f\\x93\\x8d": "\ud83d\udccd",
              "\\xf0\\x9f\\x90\\xaf": "\ud83d\udc2f", "\\xf0\\x9f\\x8e\\x8d": "\ud83c\udf8d",
              "\\xf0\\x9f\\x92\\x93": "\ud83d\udc93", "\\xf0\\x9f\\x94\\xb0": "\ud83d\udd30",
              "\\xe2\\x9c\\xb4": "\u2734", "\\xf0\\x9f\\x91\\x82": "\ud83d\udc42",
              "\\xf0\\x9f\\x8e\\x8a": "\ud83c\udf8a", "\\xf0\\x9f\\x9a\\xaf": "\ud83d\udeaf",
              "\\xf0\\x9f\\x91\\x98": "\ud83d\udc58", "\\xf0\\x9f\\x95\\xa7": "\ud83d\udd67",
              "\\xf0\\x9f\\x97\\xbd": "\ud83d\uddfd", "\\xe2\\xa4\\xb4": "\u2934",
              "\\xf0\\x9f\\x8d\\x8f": "\ud83c\udf4f", "\\xf0\\x9f\\x8e\\x8b": "\ud83c\udf8b",
              "\\xf0\\x9f\\x93\\xb4": "\ud83d\udcf4", "\\xf0\\x9f\\x8d\\x85": "\ud83c\udf45",
              "\\xf0\\x9f\\x99\\x8b": "\ud83d\ude4b", "\\xf0\\x9f\\x90\\xb5": "\ud83d\udc35",
              "\\xf0\\x9f\\x8c\\x8a": "\ud83c\udf0a", "\\xf0\\x9f\\x8f\\xad": "\ud83c\udfed",
              "\\xf0\\x9f\\x93\\x82": "\ud83d\udcc2", "\\xf0\\x9f\\x90\\xb3": "\ud83d\udc33",
              "\\xf0\\x9f\\x8e\\xb4": "\ud83c\udfb4", "\\xf0\\x9f\\x94\\x89": "\ud83d\udd09",
              "\\xf0\\x9f\\x8e\\x93": "\ud83c\udf93", "\\xf0\\x9f\\x8d\\xb7": "\ud83c\udf77",
              "\\xf0\\x9f\\x8c\\x9c": "\ud83c\udf1c", "\\xe2\\x9a\\xa0": "\u26a0",
              "\\xf0\\x9f\\x94\\xb3": "\ud83d\udd33", "\\xf0\\x9f\\x8e\\xab": "\ud83c\udfab",
              "\\xf0\\x9f\\x92\\xb2": "\ud83d\udcb2", "\\xf0\\x9f\\x95\\x90": "\ud83d\udd50",
              "\\xe2\\x84\\xb9": "\u2139", "\\xf0\\x9f\\x90\\x82": "\ud83d\udc02",
              "\\xf0\\x9f\\x92\\xbc": "\ud83d\udcbc", "\\xf0\\x9f\\x8d\\x8b": "\ud83c\udf4b",
              "\\xf0\\x9f\\x93\\xa3": "\ud83d\udce3", "\\xf0\\x9f\\x8c\\xbf": "\ud83c\udf3f",
              "\\xf0\\x9f\\x88\\xba": "\ud83c\ude3a", "\\xf0\\x9f\\x93\\x8a": "\ud83d\udcca",
              "\\xf0\\x9f\\x91\\x97": "\ud83d\udc57", "\\xf0\\x9f\\x8e\\x90": "\ud83c\udf90",
              "\\xe2\\x98\\x8e": "\u260e", "\\xf0\\x9f\\x94\\x8e": "\ud83d\udd0e",
              "\\xf0\\x9f\\x8f\\xb0": "\ud83c\udff0",
              "\\xf0\\x9f\\x87\\xa9\\xf0\\x9f\\x87\\xaa": "\ud83c\udde9\ud83c\uddea",
              "\\xf0\\x9f\\x90\\xa9": "\ud83d\udc29", "\\xf0\\x9f\\x91\\x94": "\ud83d\udc54",
              "\\xf0\\x9f\\x92\\x90": "\ud83d\udc90", "\\xf0\\x9f\\x90\\xb0": "\ud83d\udc30",
              "\\xe2\\x8f\\xab": "\u23eb", "\\xf0\\x9f\\x8f\\xa4": "\ud83c\udfe4",
              "\\xf0\\x9f\\x8f\\xab": "\ud83c\udfeb", "\\xf0\\x9f\\x94\\x98": "\ud83d\udd18",
              "\\xf0\\x9f\\x9a\\x8f": "\ud83d\ude8f", "\\xf0\\x9f\\x8e\\x8c": "\ud83c\udf8c",
              "\\xf0\\x9f\\x94\\x91": "\ud83d\udd11", "\\xf0\\x9f\\x8f\\xaf": "\ud83c\udfef",
              "\\xf0\\x9f\\x98\\x92": "\ud83d\ude12", "\\xe2\\xac\\x9c": "\u2b1c",
              "\\xf0\\x9f\\x88\\xaf": "\ud83c\ude2f", "\\xf0\\x9f\\x90\\xa0": "\ud83d\udc20",
              "\\xf0\\x9f\\x8f\\x86": "\ud83c\udfc6", "\\x23\\xe2\\x83\\xa3": "#\u20e3",
              "\\xf0\\x9f\\x88\\x81": "\ud83c\ude01", "\\xf0\\x9f\\x93\\x90": "\ud83d\udcd0",
              "\\xf0\\x9f\\x86\\x97": "\ud83c\udd97", "\\xf0\\x9f\\x9a\\xa7": "\ud83d\udea7",
              "\\xf0\\x9f\\x9b\\x82": "\ud83d\udec2", "\\xf0\\x9f\\x8c\\xb4": "\ud83c\udf34",
              "\\xf0\\x9f\\x98\\xb0": "\ud83d\ude30", "\\xf0\\x9f\\x9a\\xac": "\ud83d\udeac",
              "\\xf0\\x9f\\x90\\xa6": "\ud83d\udc26", "\\xf0\\x9f\\x8c\\x8d": "\ud83c\udf0d",
              "\\xe2\\x9b\\xaa": "\u26ea", "\\xf0\\x9f\\x90\\x8e": "\ud83d\udc0e", "\\xe2\\x80\\xbc": "\u203c",
              "\\xf0\\x9f\\x93\\x97": "\ud83d\udcd7", "\\xe2\\x8c\\x9b": "\u231b",
              "\\xf0\\x9f\\x98\\xae": "\ud83d\ude2e", "\\xf0\\x9f\\x8e\\x91": "\ud83c\udf91",
              "\\xe2\\x9d\\x93": "\u2753", "\\xe2\\x81\\x89": "\u2049", "\\xf0\\x9f\\x93\\x9f": "\ud83d\udcdf",
              "\\xf0\\x9f\\x8c\\x82": "\ud83c\udf02", "\\xf0\\x9f\\x9a\\xae": "\ud83d\udeae",
              "\\xf0\\x9f\\x98\\xa3": "\ud83d\ude23", "\\xf0\\x9f\\x8d\\x96": "\ud83c\udf56",
              "\\xf0\\x9f\\x94\\x86": "\ud83d\udd06", "\\xf0\\x9f\\x9a\\xbf": "\ud83d\udebf",
              "\\xe2\\x9c\\x8f": "\u270f", "\\xe2\\x9c\\x85": "\u2705", "\\xf0\\x9f\\x92\\xac": "\ud83d\udcac",
              "\\xf0\\x9f\\x94\\xa8": "\ud83d\udd28", "\\xf0\\x9f\\x8c\\x9e": "\ud83c\udf1e",
              "\\xf0\\x9f\\x92\\x91": "\ud83d\udc91", "\\xf0\\x9f\\x95\\x94": "\ud83d\udd54",
              "\\xf0\\x9f\\x99\\x87": "\ud83d\ude47", "\\xe2\\x9c\\x8b": "\u270b", "\\xe2\\x97\\xbe": "\u25fe",
              "\\xe2\\x99\\xbf": "\u267f", "\\xf0\\x9f\\x98\\xa9": "\ud83d\ude29",
              "\\xf0\\x9f\\x9a\\x8e": "\ud83d\ude8e", "\\xf0\\x9f\\x8e\\x80": "\ud83c\udf80",
              "\\xf0\\x9f\\x93\\x93": "\ud83d\udcd3", "\\xf0\\x9f\\x94\\xa0": "\ud83d\udd20",
              "\\xf0\\x9f\\x9a\\xa1": "\ud83d\udea1", "\\xf0\\x9f\\x91\\xae": "\ud83d\udc6e",
              "\\xf0\\x9f\\x91\\x8f": "\ud83d\udc4f", "\\xf0\\x9f\\x90\\xb6": "\ud83d\udc36",
              "\\xf0\\x9f\\x92\\x82": "\ud83d\udc82", "\\xf0\\x9f\\x8d\\xad": "\ud83c\udf6d",
              "\\xf0\\x9f\\x98\\xa1": "\ud83d\ude21", "\\xf0\\x9f\\x92\\xa0": "\ud83d\udca0",
              "\\x35\\xe2\\x83\\xa3": "5\u20e3", "\\xf0\\x9f\\x8e\\x88": "\ud83c\udf88",
              "\\xf0\\x9f\\x90\\xac": "\ud83d\udc2c", "\\xf0\\x9f\\x95\\xa2": "\ud83d\udd62",
              "\\xf0\\x9f\\x9a\\x9f": "\ud83d\ude9f", "\\xf0\\x9f\\x8e\\x92": "\ud83c\udf92",
              "\\xf0\\x9f\\x90\\xa4": "\ud83d\udc24", "\\xe2\\x86\\x98": "\u2198",
              "\\xf0\\x9f\\x99\\x86": "\ud83d\ude46", "\\xf0\\x9f\\x91\\x88": "\ud83d\udc48",
              "\\xf0\\x9f\\x93\\xaa": "\ud83d\udcea", "\\xf0\\x9f\\x8e\\xa6": "\ud83c\udfa6",
              "\\xf0\\x9f\\x8c\\xb9": "\ud83c\udf39", "\\xf0\\x9f\\x8d\\x80": "\ud83c\udf40",
              "\\xe2\\x9b\\xb2": "\u26f2", "\\xf0\\x9f\\x90\\x8a": "\ud83d\udc0a",
              "\\xf0\\x9f\\x98\\xbd": "\ud83d\ude3d", "\\xf0\\x9f\\x8d\\xae": "\ud83c\udf6e",
              "\\xf0\\x9f\\x94\\x85": "\ud83d\udd05", "\\xf0\\x9f\\x94\\x92": "\ud83d\udd12",
              "\\xf0\\x9f\\x92\\xba": "\ud83d\udcba", "\\xf0\\x9f\\x91\\x99": "\ud83d\udc59",
              "\\xf0\\x9f\\x91\\xab": "\ud83d\udc6b", "\\xf0\\x9f\\x95\\xa5": "\ud83d\udd65",
              "\\xe2\\x9e\\x95": "\u2795", "\\xf0\\x9f\\x91\\xa8": "\ud83d\udc68", "\\xe2\\x98\\x9d": "\u261d",
              "\\xf0\\x9f\\x9a\\x90": "\ud83d\ude90", "\\xf0\\x9f\\x9a\\x8a": "\ud83d\ude8a",
              "\\xf0\\x9f\\x92\\x89": "\ud83d\udc89", "\\xf0\\x9f\\x8f\\x8a": "\ud83c\udfca",
              "\\xf0\\x9f\\x8e\\xb8": "\ud83c\udfb8", "\\xf0\\x9f\\x95\\xa3": "\ud83d\udd63",
              "\\xf0\\x9f\\x8e\\xbf": "\ud83c\udfbf", "\\xf0\\x9f\\x94\\xa3": "\ud83d\udd23",
              "\\xf0\\x9f\\x91\\xa6": "\ud83d\udc66", "\\xe2\\x8f\\xac": "\u23ec",
              "\\xf0\\x9f\\x93\\xac": "\ud83d\udcec", "\\xf0\\x9f\\x8d\\x9c": "\ud83c\udf5c",
              "\\xf0\\x9f\\x8e\\x8f": "\ud83c\udf8f", "\\xf0\\x9f\\x90\\x89": "\ud83d\udc09",
              "\\xf0\\x9f\\x93\\xad": "\ud83d\udced", "\\xf0\\x9f\\x92\\xad": "\ud83d\udcad",
              "\\xf0\\x9f\\x9a\\x85": "\ud83d\ude85", "\\xf0\\x9f\\x9a\\x9d": "\ud83d\ude9d",
              "\\xf0\\x9f\\x98\\xba": "\ud83d\ude3a", "\\xf0\\x9f\\x91\\xa2": "\ud83d\udc62",
              "\\xf0\\x9f\\x91\\x83": "\ud83d\udc43", "\\xf0\\x9f\\x9a\\xb9": "\ud83d\udeb9",
              "\\xf0\\x9f\\x98\\x99": "\ud83d\ude19", "\\xf0\\x9f\\x92\\x8d": "\ud83d\udc8d",
              "\\xe2\\x9e\\x97": "\u2797", "\\xf0\\x9f\\x91\\x8d": "\ud83d\udc4d", "\\xe2\\x86\\x94": "\u2194",
              "\\xf0\\x9f\\x9a\\xa4": "\ud83d\udea4", "\\x39\\xe2\\x83\\xa3": "9\u20e3",
              "\\xf0\\x9f\\x93\\x8f": "\ud83d\udccf", "\\xf0\\x9f\\x95\\xa4": "\ud83d\udd64",
              "\\xf0\\x9f\\x90\\x96": "\ud83d\udc16", "\\xf0\\x9f\\x9a\\x87": "\ud83d\ude87",
              "\\xf0\\x9f\\x94\\xa9": "\ud83d\udd29", "\\xf0\\x9f\\x97\\xbf": "\ud83d\uddff",
              "\\xf0\\x9f\\x9a\\x94": "\ud83d\ude94", "\\xf0\\x9f\\x92\\x9d": "\ud83d\udc9d",
              "\\xf0\\x9f\\x90\\xb2": "\ud83d\udc32", "\\xe2\\x98\\x81": "\u2601", "\\xe2\\xac\\x85": "\u2b05",
              "\\xf0\\x9f\\x8e\\xba": "\ud83c\udfba", "\\xf0\\x9f\\x8e\\xbe": "\ud83c\udfbe",
              "\\xf0\\x9f\\x85\\xbe": "\ud83c\udd7e", "\\xe2\\x9c\\x92": "\u2712", "\\xe2\\x9d\\x97": "\u2757",
              "\\xe2\\x97\\x80": "\u25c0", "\\xe2\\x86\\x97": "\u2197", "\\xf0\\x9f\\x8d\\x9a": "\ud83c\udf5a",
              "\\xf0\\x9f\\x92\\xa3": "\ud83d\udca3", "\\xf0\\x9f\\x98\\x86": "\ud83d\ude06",
              "\\xf0\\x9f\\x93\\x83": "\ud83d\udcc3", "\\xf0\\x9f\\x91\\xa3": "\ud83d\udc63",
              "\\xf0\\x9f\\x8f\\x82": "\ud83c\udfc2", "\\xf0\\x9f\\x90\\x9a": "\ud83d\udc1a",
              "\\xf0\\x9f\\x94\\x9c": "\ud83d\udd1c", "\\xf0\\x9f\\x91\\xbe": "\ud83d\udc7e",
              "\\xf0\\x9f\\x99\\x8a": "\ud83d\ude4a", "\\xf0\\x9f\\x90\\x81": "\ud83d\udc01",
              "\\xf0\\x9f\\x8e\\xb0": "\ud83c\udfb0", "\\xf0\\x9f\\x8e\\xb9": "\ud83c\udfb9",
              "\\xf0\\x9f\\x95\\x96": "\ud83d\udd56", "\\xf0\\x9f\\x8c\\x8c": "\ud83c\udf0c",
              "\\xf0\\x9f\\x95\\x9d": "\ud83d\udd5d", "\\xf0\\x9f\\x8d\\x92": "\ud83c\udf52",
              "\\xf0\\x9f\\x92\\xa5": "\ud83d\udca5", "\\xf0\\x9f\\x93\\xa2": "\ud83d\udce2",
              "\\xf0\\x9f\\x8c\\x9f": "\ud83c\udf1f", "\\xf0\\x9f\\x95\\x9a": "\ud83d\udd5a",
              "\\xf0\\x9f\\x8d\\xbb": "\ud83c\udf7b", "\\xf0\\x9f\\x8e\\x81": "\ud83c\udf81",
              "\\xf0\\x9f\\x9a\\x8c": "\ud83d\ude8c", "\\xf0\\x9f\\x8e\\xa7": "\ud83c\udfa7",
              "\\xf0\\x9f\\x92\\xbd": "\ud83d\udcbd", "\\xf0\\x9f\\x86\\x98": "\ud83c\udd98",
              "\\xf0\\x9f\\x98\\x9a": "\ud83d\ude1a", "\\xf0\\x9f\\x98\\x90": "\ud83d\ude10",
              "\\xe2\\x99\\x8e": "\u264e", "\\xf0\\x9f\\x8d\\xa6": "\ud83c\udf66",
              "\\xf0\\x9f\\x8c\\x93": "\ud83c\udf13", "\\xf0\\x9f\\x8c\\xb8": "\ud83c\udf38",
              "\\xf0\\x9f\\x98\\xb4": "\ud83d\ude34", "\\xf0\\x9f\\x8c\\xb1": "\ud83c\udf31",
              "\\xf0\\x9f\\x9a\\x99": "\ud83d\ude99", "\\xf0\\x9f\\x93\\x81": "\ud83d\udcc1",
              "\\xf0\\x9f\\x8e\\xa5": "\ud83c\udfa5", "\\xf0\\x9f\\x93\\xb5": "\ud83d\udcf5",
              "\\xe2\\x99\\x93": "\u2653", "\\xf0\\x9f\\x8c\\x9b": "\ud83c\udf1b",
              "\\xf0\\x9f\\x8e\\xa9": "\ud83c\udfa9", "\\xf0\\x9f\\x94\\x90": "\ud83d\udd10",
              "\\xf0\\x9f\\x8f\\xa8": "\ud83c\udfe8", "\\xf0\\x9f\\x91\\x84": "\ud83d\udc44",
              "\\xe2\\x9c\\x96": "\u2716", "\\xf0\\x9f\\x95\\x92": "\ud83d\udd52",
              "\\xf0\\x9f\\x88\\xb9": "\ud83c\ude39", "\\xf0\\x9f\\x8e\\x82": "\ud83c\udf82",
              "\\xf0\\x9f\\x8c\\x97": "\ud83c\udf17", "\\xf0\\x9f\\x92\\x96": "\ud83d\udc96",
              "\\xf0\\x9f\\x94\\x9b": "\ud83d\udd1b", "\\xf0\\x9f\\x93\\xbb": "\ud83d\udcfb",
              "\\xf0\\x9f\\x91\\xbf": "\ud83d\udc7f", "\\xf0\\x9f\\x8e\\xa2": "\ud83c\udfa2",
              "\\xf0\\x9f\\x95\\x98": "\ud83d\udd58", "\\xf0\\x9f\\x93\\x99": "\ud83d\udcd9",
              "\\xf0\\x9f\\x88\\x82": "\ud83c\ude02", "\\xe2\\x9e\\xb0": "\u27b0", "\\xe2\\x9b\\xba": "\u26fa",
              "\\xe2\\x9a\\xbd": "\u26bd", "\\xf0\\x9f\\x8c\\xb2": "\ud83c\udf32",
              "\\xf0\\x9f\\x8c\\x81": "\ud83c\udf01", "\\xf0\\x9f\\x88\\x9a": "\ud83c\ude1a",
              "\\xf0\\x9f\\x90\\x9f": "\ud83d\udc1f", "\\xf0\\x9f\\x86\\x92": "\ud83c\udd92",
              "\\xf0\\x9f\\x9b\\x83": "\ud83d\udec3", "\\xe3\\x80\\xb0": "\u3030",
              "\\xf0\\x9f\\x94\\x94": "\ud83d\udd14", "\\xf0\\x9f\\x8e\\xae": "\ud83c\udfae",
              "\\xf0\\x9f\\x8e\\xb6": "\ud83c\udfb6", "\\xf0\\x9f\\x91\\x85": "\ud83d\udc45",
              "\\xf0\\x9f\\x98\\xb9": "\ud83d\ude39", "\\xf0\\x9f\\x94\\xaa": "\ud83d\udd2a",
              "\\xf0\\x9f\\x91\\x8b": "\ud83d\udc4b", "\\xf0\\x9f\\x94\\xa5": "\ud83d\udd25",
              "\\xf0\\x9f\\x99\\x80": "\ud83d\ude40", "\\xf0\\x9f\\x90\\xb7": "\ud83d\udc37",
              "\\x34\\xe2\\x83\\xa3": "4\u20e3", "\\xf0\\x9f\\x8d\\x91": "\ud83c\udf51",
              "\\xf0\\x9f\\x90\\x95": "\ud83d\udc15", "\\xf0\\x9f\\x93\\xab": "\ud83d\udceb",
              "\\xe2\\xac\\x9b": "\u2b1b"}
EMOJIS_UNICODE = {"u+1f509": "\ud83d\udd09", "u+1f3e6": "\ud83c\udfe6", "u+2601": "\u2601", "u+1f391": "\ud83c\udf91",
                  "u+1f331": "\ud83c\udf31", "u+1f515": "\ud83d\udd15", "u+1f304": "\ud83c\udf04",
                  "u+1f539": "\ud83d\udd39", "u+1f645": "\ud83d\ude45", "u+1f4a6": "\ud83d\udca6", "u+3030": "\u3030",
                  "u+25fb": "\u25fb", "u+1f5ff": "\ud83d\uddff", "u+1f3c2": "\ud83c\udfc2", "u+1f196": "\ud83c\udd96",
                  "u+1f442": "\ud83d\udc42", "u+1f4e1": "\ud83d\udce1", "u+1f3bf": "\ud83c\udfbf",
                  "u+1f6ac": "\ud83d\udeac", "u+1f330": "\ud83c\udf30", "u+1f3a6": "\ud83c\udfa6",
                  "u+1f6b3": "\ud83d\udeb3", "u+1f61b": "\ud83d\ude1b", "u+1f369": "\ud83c\udf69",
                  "u+1f317": "\ud83c\udf17", "u+1f688": "\ud83d\ude88", "u+26ce": "\u26ce", "u+1f192": "\ud83c\udd92",
                  "u+2615": "\u2615", "u+1f6aa": "\ud83d\udeaa", "u+1f3e9": "\ud83c\udfe9", "u+1f455": "\ud83d\udc55",
                  "u+1f195": "\ud83c\udd95", "u+1f536": "\ud83d\udd36", "u+2705": "\u2705", "u+0035 u+20e3": "5\u20e3",
                  "u+1f385": "\ud83c\udf85", "u+1f382": "\ud83c\udf82", "u+1f1ec u+1f1e7": "\ud83c\uddec\ud83c\udde7",
                  "u+1f519": "\ud83d\udd19", "u+1f4eb": "\ud83d\udceb", "u+1f3ef": "\ud83c\udfef",
                  "u+1f62f": "\ud83d\ude2f", "u+1f49e": "\ud83d\udc9e", "u+1f687": "\ud83d\ude87",
                  "u+1f63f": "\ud83d\ude3f", "u+1f379": "\ud83c\udf79", "u+1f437": "\ud83d\udc37",
                  "u+1f497": "\ud83d\udc97", "u+2663": "\u2663", "u+1f4e8": "\ud83d\udce8", "u+1f30d": "\ud83c\udf0d",
                  "u+1f524": "\ud83d\udd24", "u+264e": "\u264e", "u+1f50c": "\ud83d\udd0c", "u+1f518": "\ud83d\udd18",
                  "u+1f639": "\ud83d\ude39", "u+1f347": "\ud83c\udf47", "u+1f44d": "\ud83d\udc4d",
                  "u+1f689": "\ud83d\ude89", "u+1f433": "\ud83d\udc33", "u+1f1e9 u+1f1ea": "\ud83c\udde9\ud83c\uddea",
                  "u+1f469": "\ud83d\udc69", "u+1f235": "\ud83c\ude35", "u+1f494": "\ud83d\udc94",
                  "u+1f4f5": "\ud83d\udcf5", "u+2195": "\u2195", "u+1f312": "\ud83c\udf12", "u+1f3b2": "\ud83c\udfb2",
                  "u+1f61d": "\ud83d\ude1d", "u+1f3e0": "\ud83c\udfe0", "u+1f43b": "\ud83d\udc3b",
                  "u+1f5fe": "\ud83d\uddfe", "u+1f68f": "\ud83d\ude8f", "u+1f376": "\ud83c\udf76",
                  "u+1f6b2": "\ud83d\udeb2", "u+1f42a": "\ud83d\udc2a", "u+1f693": "\ud83d\ude93",
                  "u+1f439": "\ud83d\udc39", "u+26be": "\u26be", "u+2194": "\u2194", "u+1f6c2": "\ud83d\udec2",
                  "u+1f479": "\ud83d\udc79", "u+1f18e": "\ud83c\udd8e", "u+1f34b": "\ud83c\udf4b",
                  "u+1f3e5": "\ud83c\udfe5", "u+2611": "\u2611", "u+1f446": "\ud83d\udc46", "u+1f45a": "\ud83d\udc5a",
                  "u+1f3e8": "\ud83c\udfe8", "u+1f47e": "\ud83d\udc7e", "u+1f616": "\ud83d\ude16", "u+2197": "\u2197",
                  "u+1f501": "\ud83d\udd01", "u+1f17e": "\ud83c\udd7e", "u+1f466": "\ud83d\udc66",
                  "u+1f35f": "\ud83c\udf5f", "u+1f55f": "\ud83d\udd5f", "u+1f525": "\ud83d\udd25",
                  "u+1f33a": "\ud83c\udf3a", "u+1f4ee": "\ud83d\udcee", "u+1f555": "\ud83d\udd55", "u+2653": "\u2653",
                  "u+1f6b5": "\ud83d\udeb5", "u+1f6a4": "\ud83d\udea4", "u+1f3bd": "\ud83c\udfbd",
                  "u+1f6b8": "\ud83d\udeb8", "u+1f47f": "\ud83d\udc7f", "u+1f627": "\ud83d\ude27",
                  "u+1f380": "\ud83c\udf80", "u+1f40f": "\ud83d\udc0f", "u+1f50f": "\ud83d\udd0f",
                  "u+1f3c4": "\ud83c\udfc4", "u+1f4ae": "\ud83d\udcae", "u+1f623": "\ud83d\ude23",
                  "u+1f4e2": "\ud83d\udce2", "u+1f53a": "\ud83d\udd3a", "u+2753": "\u2753", "u+1f30b": "\ud83c\udf0b",
                  "u+1f386": "\ud83c\udf86", "u+1f4af": "\ud83d\udcaf", "u+1f448": "\ud83d\udc48",
                  "u+1f48e": "\ud83d\udc8e", "u+1f171": "\ud83c\udd71", "u+2714": "\u2714", "u+1f4c7": "\ud83d\udcc7",
                  "u+1f487": "\ud83d\udc87", "u+1f6b1": "\ud83d\udeb1", "u+1f1ef u+1f1f5": "\ud83c\uddef\ud83c\uddf5",
                  "u+25ab": "\u25ab", "u+1f37a": "\ud83c\udf7a", "u+1f557": "\ud83d\udd57", "u+1f6bb": "\ud83d\udebb",
                  "u+21a9": "\u21a9", "u+1f523": "\ud83d\udd23", "u+1f562": "\ud83d\udd62", "u+1f4e4": "\ud83d\udce4",
                  "u+25b6": "\u25b6", "u+1f350": "\ud83c\udf50", "u+303d": "\u303d", "u+1f60c": "\ud83d\ude0c",
                  "u+1f63d": "\ud83d\ude3d", "u+1f564": "\ud83d\udd64", "u+27b0": "\u27b0", "u+1f567": "\ud83d\udd67",
                  "u+1f4cf": "\ud83d\udccf", "u+1f456": "\ud83d\udc56", "u+1f36c": "\ud83c\udf6c",
                  "u+1f566": "\ud83d\udd66", "u+1f604": "\ud83d\ude04", "u+1f4f9": "\ud83d\udcf9",
                  "u+1f485": "\ud83d\udc85", "u+1f521": "\ud83d\udd21", "u+1f40d": "\ud83d\udc0d",
                  "u+1f339": "\ud83c\udf39", "u+1f41d": "\ud83d\udc1d", "u+1f69b": "\ud83d\ude9b", "u+2198": "\u2198",
                  "u+1f4de": "\ud83d\udcde", "u+1f421": "\ud83d\udc21", "u+2668": "\u2668", "u+23ec": "\u23ec",
                  "u+2744": "\u2744", "u+25fd": "\u25fd", "u+1f359": "\ud83c\udf59", "u+1f457": "\ud83d\udc57",
                  "u+1f4d7": "\ud83d\udcd7", "u+1f41b": "\ud83d\udc1b", "u+1f4ce": "\ud83d\udcce",
                  "u+1f4fc": "\ud83d\udcfc", "u+1f343": "\ud83c\udf43", "u+1f649": "\ud83d\ude49",
                  "u+1f461": "\ud83d\udc61", "u+0030 u+20e3": "0\u20e3", "u+1f480": "\ud83d\udc80",
                  "u+1f373": "\ud83c\udf73", "u+1f4e5": "\ud83d\udce5", "u+1f31e": "\ud83c\udf1e",
                  "u+1f429": "\ud83d\udc29", "u+1f4be": "\ud83d\udcbe", "u+00a9": "\u00a9", "u+1f503": "\ud83d\udd03",
                  "u+2702": "\u2702", "u+1f4c3": "\ud83d\udcc3", "u+1f46e": "\ud83d\udc6e", "u+1f402": "\ud83d\udc02",
                  "u+1f510": "\ud83d\udd10", "u+1f69c": "\ud83d\ude9c", "u+1f478": "\ud83d\udc78",
                  "u+1f3ab": "\ud83c\udfab", "u+1f47b": "\ud83d\udc7b", "u+1f472": "\ud83d\udc72",
                  "u+1f21a": "\ud83c\ude1a", "u+263a": "\u263a", "u+1f318": "\ud83c\udf18", "u+1f626": "\ud83d\ude26",
                  "u+2665": "\u2665", "u+1f3c8": "\ud83c\udfc8", "u+1f342": "\ud83c\udf42", "u+26f2": "\u26f2",
                  "u+1f194": "\ud83c\udd94", "u+1f68d": "\ud83d\ude8d", "u+1f4ac": "\ud83d\udcac",
                  "u+1f68e": "\ud83d\ude8e", "u+1f356": "\ud83c\udf56", "u+1f52a": "\ud83d\udd2a",
                  "u+1f40e": "\ud83d\udc0e", "u+1f560": "\ud83d\udd60", "u+1f475": "\ud83d\udc75", "u+203c": "\u203c",
                  "u+1f422": "\ud83d\udc22", "u+1f55d": "\ud83d\udd5d", "u+1f40b": "\ud83d\udc0b",
                  "u+1f41f": "\ud83d\udc1f", "u+1f1ee u+1f1f9": "\ud83c\uddee\ud83c\uddf9", "u+1f432": "\ud83d\udc32",
                  "u+1f473": "\ud83d\udc73", "u+1f648": "\ud83d\ude48", "u+1f004": "\ud83c\udc04",
                  "u+1f1f0 u+1f1f7": "\ud83c\uddf0\ud83c\uddf7", "u+1f470": "\ud83d\udc70", "u+1f411": "\ud83d\udc11",
                  "u+1f400": "\ud83d\udc00", "u+1f558": "\ud83d\udd58", "u+1f4bb": "\ud83d\udcbb",
                  "u+1f43e": "\ud83d\udc3e", "u+1f514": "\ud83d\udd14", "u+1f618": "\ud83d\ude18",
                  "u+1f504": "\ud83d\udd04", "u+1f681": "\ud83d\ude81", "u+2648": "\u2648", "u+1f307": "\ud83c\udf07",
                  "u+1f38b": "\ud83c\udf8b", "u+25aa": "\u25aa", "u+1f6bc": "\ud83d\udebc", "u+1f477": "\ud83d\udc77",
                  "u+1f375": "\ud83c\udf75", "u+1f30c": "\ud83c\udf0c", "u+264a": "\u264a", "u+1f64c": "\ud83d\ude4c",
                  "u+1f4b1": "\ud83d\udcb1", "u+274c": "\u274c", "u+1f363": "\ud83c\udf63", "u+1f620": "\ud83d\ude20",
                  "u+1f6c5": "\ud83d\udec5", "u+1f516": "\ud83d\udd16", "u+1f4f2": "\ud83d\udcf2",
                  "u+1f634": "\ud83d\ude34", "u+2747": "\u2747", "u+1f364": "\ud83c\udf64", "u+1f608": "\ud83d\ude08",
                  "u+2049": "\u2049", "u+1f502": "\ud83d\udd02", "u+1f4c1": "\ud83d\udcc1", "u+2b1c": "\u2b1c",
                  "u+1f697": "\ud83d\ude97", "u+1f50e": "\ud83d\udd0e", "u+1f629": "\ud83d\ude29",
                  "u+1f417": "\ud83d\udc17", "u+267b": "\u267b", "u+1f42d": "\ud83d\udc2d", "u+1f3c6": "\ud83c\udfc6",
                  "u+1f496": "\ud83d\udc96", "u+1f462": "\ud83d\udc62", "u+23f0": "\u23f0", "u+1f6b4": "\ud83d\udeb4",
                  "u+1f361": "\ud83c\udf61", "u+1f602": "\ud83d\ude02", "u+2b06": "\u2b06", "u+1f529": "\ud83d\udd29",
                  "u+1f31d": "\ud83c\udf1d", "u+1f63b": "\ud83d\ude3b", "u+1f531": "\ud83d\udd31",
                  "u+1f31b": "\ud83c\udf1b", "u+1f47c": "\ud83d\udc7c", "u+1f423": "\ud83d\udc23", "u+270b": "\u270b",
                  "u+1f3e3": "\ud83c\udfe3", "u+1f1e8 u+1f1f3": "\ud83c\udde8\ud83c\uddf3", "u+1f4cb": "\ud83d\udccb",
                  "u+1f53c": "\ud83d\udd3c", "u+1f45b": "\ud83d\udc5b", "u+2708": "\u2708", "u+1f52c": "\ud83d\udd2c",
                  "u+1f37c": "\ud83c\udf7c", "u+1f3e2": "\ud83c\udfe2", "u+1f407": "\ud83d\udc07",
                  "u+1f55e": "\ud83d\udd5e", "u+1f528": "\ud83d\udd28", "u+1f383": "\ud83c\udf83",
                  "u+1f5fc": "\ud83d\uddfc", "u+1f619": "\ud83d\ude19", "u+26d4": "\u26d4", "u+1f40c": "\ud83d\udc0c",
                  "u+1f3b4": "\ud83c\udfb4", "u+1f362": "\ud83c\udf62", "u+1f3c9": "\ud83c\udfc9",
                  "u+1f699": "\ud83d\ude99", "u+2934": "\u2934", "u+1f3c0": "\ud83c\udfc0", "u+1f42c": "\ud83d\udc2c",
                  "u+1f311": "\ud83c\udf11", "u+1f6bf": "\ud83d\udebf", "u+1f3c7": "\ud83c\udfc7",
                  "u+1f611": "\ud83d\ude11", "u+1f4f7": "\ud83d\udcf7", "u+1f36a": "\ud83c\udf6a",
                  "u+1f4df": "\ud83d\udcdf", "u+1f4d6": "\ud83d\udcd6", "u+1f360": "\ud83c\udf60",
                  "u+1f43a": "\ud83d\udc3a", "u+1f533": "\ud83d\udd33", "u+1f3e4": "\ud83c\udfe4", "u+25fe": "\u25fe",
                  "u+1f6a3": "\ud83d\udea3", "u+1f50a": "\ud83d\udd0a", "u+1f41a": "\ud83d\udc1a",
                  "u+1f316": "\ud83c\udf16", "u+1f563": "\ud83d\udd63", "u+1f64a": "\ud83d\ude4a",
                  "u+1f334": "\ud83c\udf34", "u+1f62c": "\ud83d\ude2c", "u+1f537": "\ud83d\udd37", "u+26a0": "\u26a0",
                  "u+25c0": "\u25c0", "u+1f3ad": "\ud83c\udfad", "u+1f482": "\ud83d\udc82", "u+0036 u+20e3": "6\u20e3",
                  "u+1f691": "\ud83d\ude91", "u+1f3b5": "\ud83c\udfb5", "u+1f603": "\ud83d\ude03",
                  "u+1f42f": "\ud83d\udc2f", "u+1f6ab": "\ud83d\udeab", "u+1f4ba": "\ud83d\udcba",
                  "u+1f31a": "\ud83c\udf1a", "u+1f5fd": "\ud83d\uddfd", "u+1f607": "\ud83d\ude07",
                  "u+1f4e3": "\ud83d\udce3", "u+1f498": "\ud83d\udc98", "u+1f313": "\ud83c\udf13",
                  "u+1f4d4": "\ud83d\udcd4", "u+1f6b6": "\ud83d\udeb6", "u+1f43c": "\ud83d\udc3c",
                  "u+1f680": "\ud83d\ude80", "u+1f47a": "\ud83d\udc7a", "u+1f6ae": "\ud83d\udeae",
                  "u+1f62b": "\ud83d\ude2b", "u+2764": "\u2764", "u+1f55a": "\ud83d\udd5a",
                  "u+1f1fa u+1f1f8": "\ud83c\uddfa\ud83c\uddf8", "u+2b05": "\u2b05", "u+2734": "\u2734",
                  "u+1f197": "\ud83c\udd97", "u+1f36f": "\ud83c\udf6f", "u+1f6c4": "\ud83d\udec4", "u+231a": "\u231a",
                  "u+1f4e6": "\ud83d\udce6", "u+1f3ba": "\ud83c\udfba", "u+1f4c9": "\ud83d\udcc9",
                  "u+1f51a": "\ud83d\udd1a", "u+1f3a1": "\ud83c\udfa1", "u+1f53b": "\ud83d\udd3b",
                  "u+1f624": "\ud83d\ude24", "u+1f3e1": "\ud83c\udfe1", "u+1f3eb": "\ud83c\udfeb",
                  "u+1f61f": "\ud83d\ude1f", "u+2712": "\u2712", "u+24c2": "\u24c2", "u+1f354": "\ud83c\udf54",
                  "u+267f": "\u267f", "u+1f3af": "\ud83c\udfaf", "u+0037 u+20e3": "7\u20e3", "u+1f64d": "\ud83d\ude4d",
                  "u+1f401": "\ud83d\udc01", "u+1f345": "\ud83c\udf45", "u+1f3be": "\ud83c\udfbe",
                  "u+1f53d": "\ud83d\udd3d", "u+1f418": "\ud83d\udc18", "u+1f49f": "\ud83d\udc9f",
                  "u+1f393": "\ud83c\udf93", "u+1f481": "\ud83d\udc81", "u+1f682": "\ud83d\ude82",
                  "u+1f6ba": "\ud83d\udeba", "u+1f440": "\ud83d\udc40", "u+270a": "\u270a", "u+1f4f4": "\ud83d\udcf4",
                  "u+1f48a": "\ud83d\udc8a", "u+1f4a7": "\ud83d\udca7", "u+1f4d2": "\ud83d\udcd2",
                  "u+1f4c4": "\ud83d\udcc4", "u+1f500": "\ud83d\udd00", "u+1f420": "\ud83d\udc20",
                  "u+1f202": "\ud83c\ude02", "u+1f458": "\ud83d\udc58", "u+1f191": "\ud83c\udd91",
                  "u+1f6c0": "\ud83d\udec0", "u+1f42e": "\ud83d\udc2e", "u+1f34f": "\ud83c\udf4f",
                  "u+1f4ef": "\ud83d\udcef", "u+1f3b8": "\ud83c\udfb8", "u+274e": "\u274e", "u+1f486": "\ud83d\udc86",
                  "u+1f46a": "\ud83d\udc6a", "u+1f60a": "\ud83d\ude0a", "u+1f4d0": "\ud83d\udcd0",
                  "u+1f3aa": "\ud83c\udfaa", "u+2754": "\u2754", "u+1f460": "\ud83d\udc60", "u+1f335": "\ud83c\udf35",
                  "u+1f637": "\ud83d\ude37", "u+1f517": "\ud83d\udd17", "u+1f48d": "\ud83d\udc8d",
                  "u+1f41c": "\ud83d\udc1c", "u+1f351": "\ud83c\udf51", "u+1f341": "\ud83c\udf41",
                  "u+1f37b": "\ud83c\udf7b", "u+1f51d": "\ud83d\udd1d", "u+1f69d": "\ud83d\ude9d",
                  "u+1f34c": "\ud83c\udf4c", "u+1f694": "\ud83d\ude94", "u+1f476": "\ud83d\udc76", "u+26a1": "\u26a1",
                  "u+1f390": "\ud83c\udf90", "u+1f233": "\ud83c\ude33", "u+1f3e7": "\ud83c\udfe7",
                  "u+1f6a9": "\ud83d\udea9", "u+0032 u+20e3": "2\u20e3", "u+1f3b1": "\ud83c\udfb1",
                  "u+1f3a7": "\ud83c\udfa7", "u+1f449": "\ud83d\udc49", "u+1f69a": "\ud83d\ude9a",
                  "u+1f452": "\ud83d\udc52", "u+1f4b4": "\ud83d\udcb4", "u+1f44b": "\ud83d\udc4b",
                  "u+1f320": "\ud83c\udf20", "u+1f459": "\ud83d\udc59", "u+1f3ec": "\ud83c\udfec",
                  "u+1f695": "\ud83d\ude95", "u+1f33b": "\ud83c\udf3b", "u+2b50": "\u2b50", "u+1f559": "\ud83d\udd59",
                  "u+1f51e": "\ud83d\udd1e", "u+1f4a1": "\ud83d\udca1", "u+2666": "\u2666", "u+1f365": "\ud83c\udf65",
                  "u+1f633": "\ud83d\ude33", "u+1f30f": "\ud83c\udf0f", "u+1f48f": "\ud83d\udc8f",
                  "u+1f471": "\ud83d\udc71", "u+1f492": "\ud83d\udc92", "u+2935": "\u2935", "u+1f3b3": "\ud83c\udfb3",
                  "u+1f367": "\ud83c\udf67", "u+1f319": "\ud83c\udf19", "u+1f427": "\ud83d\udc27",
                  "u+1f33e": "\ud83c\udf3e", "u+1f4dc": "\ud83d\udcdc", "u+1f357": "\ud83c\udf57",
                  "u+1f4a0": "\ud83d\udca0", "u+1f378": "\ud83c\udf78", "u+1f601": "\ud83d\ude01",
                  "u+1f52b": "\ud83d\udd2b", "u+1f530": "\ud83d\udd30", "u+1f696": "\ud83d\ude96",
                  "u+0023 u+20e3": "#\u20e3", "u+1f426": "\ud83d\udc26", "u+1f6a8": "\ud83d\udea8", "u+2b55": "\u2b55",
                  "u+1f303": "\ud83c\udf03", "u+1f68a": "\ud83d\ude8a", "u+1f358": "\ud83c\udf58",
                  "u+1f424": "\ud83d\udc24", "u+1f4d9": "\ud83d\udcd9", "u+1f613": "\ud83d\ude13",
                  "u+1f4ab": "\ud83d\udcab", "u+1f372": "\ud83c\udf72", "u+1f3b0": "\ud83c\udfb0", "u+264f": "\u264f",
                  "u+1f556": "\ud83d\udd56", "u+1f377": "\ud83c\udf77", "u+1f46f": "\ud83d\udc6f",
                  "u+1f6be": "\ud83d\udebe", "u+1f405": "\ud83d\udc05", "u+1f30a": "\ud83c\udf0a",
                  "u+1f310": "\ud83c\udf10", "u+1f6af": "\ud83d\udeaf", "u+1f447": "\ud83d\udc47",
                  "u+1f4f1": "\ud83d\udcf1", "u+1f3a3": "\ud83c\udfa3", "u+1f491": "\ud83d\udc91",
                  "u+1f600": "\ud83d\ude00", "u+1f3ea": "\ud83c\udfea", "u+1f538": "\ud83d\udd38",
                  "u+1f366": "\ud83c\udf66", "u+1f3c3": "\ud83c\udfc3", "u+1f683": "\ud83d\ude83",
                  "u+1f234": "\ud83c\ude34", "u+1f1f7 u+1f1fa": "\ud83c\uddf7\ud83c\uddfa", "u+1f333": "\ud83c\udf33",
                  "u+1f431": "\ud83d\udc31", "u+1f488": "\ud83d\udc88", "u+1f4b7": "\ud83d\udcb7",
                  "u+1f238": "\ud83c\ude38", "u+26c5": "\u26c5", "u+1f468": "\ud83d\udc68", "u+1f387": "\ud83c\udf87",
                  "u+1f33c": "\ud83c\udf3c", "u+23f3": "\u23f3", "u+1f553": "\ud83d\udd53", "u+1f4c6": "\ud83d\udcc6",
                  "u+1f36d": "\ud83c\udf6d", "u+0034 u+20e3": "4\u20e3", "u+1f352": "\ud83c\udf52",
                  "u+1f34e": "\ud83c\udf4e", "u+1f49b": "\ud83d\udc9b", "u+1f52d": "\ud83d\udd2d",
                  "u+1f69e": "\ud83d\ude9e", "u+1f474": "\ud83d\udc74", "u+1f3f0": "\ud83c\udff0",
                  "u+1f201": "\ud83c\ude01", "u+1f193": "\ud83c\udd93", "u+1f51c": "\ud83d\udd1c",
                  "u+1f49c": "\ud83d\udc9c", "u+1f414": "\ud83d\udc14", "u+1f4a4": "\ud83d\udca4",
                  "u+1f1eb u+1f1f7": "\ud83c\uddeb\ud83c\uddf7", "u+1f55b": "\ud83d\udd5b", "u+1f22f": "\ud83c\ude2f",
                  "u+1f338": "\ud83c\udf38", "u+1f35e": "\ud83c\udf5e", "u+1f38f": "\ud83c\udf8f", "u+2614": "\u2614",
                  "u+1f4bc": "\ud83d\udcbc", "u+1f36b": "\ud83c\udf6b", "u+1f392": "\ud83c\udf92",
                  "u+1f4fa": "\ud83d\udcfa", "u+1f499": "\ud83d\udc99", "u+1f0cf": "\ud83c\udccf",
                  "u+0038 u+20e3": "8\u20e3", "u+1f493": "\ud83d\udc93", "u+261d": "\u261d", "u+23e9": "\u23e9",
                  "u+1f45c": "\ud83d\udc5c", "u+1f239": "\ud83c\ude39", "u+1f38d": "\ud83c\udf8d", "u+3297": "\u3297",
                  "u+1f61a": "\ud83d\ude1a", "u+1f506": "\ud83d\udd06", "u+1f353": "\ud83c\udf53",
                  "u+1f31f": "\ud83c\udf1f", "u+1f6c1": "\ud83d\udec1", "u+1f33f": "\ud83c\udf3f",
                  "u+1f3a0": "\ud83c\udfa0", "u+1f4e7": "\ud83d\udce7", "u+1f4c5": "\ud83d\udcc5",
                  "u+1f60b": "\ud83d\ude0b", "u+1f236": "\ud83c\ude36", "u+1f617": "\ud83d\ude17",
                  "u+1f315": "\ud83c\udf15", "u+1f60d": "\ud83d\ude0d", "u+1f3c1": "\ud83c\udfc1",
                  "u+1f438": "\ud83d\udc38", "u+1f465": "\ud83d\udc65", "u+1f690": "\ud83d\ude90",
                  "u+1f389": "\ud83c\udf89", "u+1f69f": "\ud83d\ude9f", "u+1f4b3": "\ud83d\udcb3",
                  "u+1f63c": "\ud83d\ude3c", "u+1f44c": "\ud83d\udc4c", "u+1f467": "\ud83d\udc67",
                  "u+1f445": "\ud83d\udc45", "u+1f4cd": "\ud83d\udccd", "u+1f374": "\ud83c\udf74",
                  "u+1f561": "\ud83d\udd61", "u+1f47d": "\ud83d\udc7d", "u+1f62e": "\ud83d\ude2e", "u+2b07": "\u2b07",
                  "u+1f4ea": "\ud83d\udcea", "u+1f403": "\ud83d\udc03", "u+23ea": "\u23ea", "u+1f3a2": "\ud83c\udfa2",
                  "u+1f526": "\ud83d\udd26", "u+1f4e9": "\ud83d\udce9", "u+1f4a8": "\ud83d\udca8", "u+26bd": "\u26bd",
                  "u+1f63e": "\ud83d\ude3e", "u+1f309": "\ud83c\udf09", "u+264b": "\u264b", "u+1f4dd": "\ud83d\udcdd",
                  "u+1f34a": "\ud83c\udf4a", "u+1f340": "\ud83c\udf40", "u+1f4aa": "\ud83d\udcaa",
                  "u+1f614": "\ud83d\ude14", "u+26c4": "\u26c4", "u+1f4b0": "\ud83d\udcb0", "u+270c": "\u270c",
                  "u+1f52f": "\ud83d\udd2f", "u+1f507": "\ud83d\udd07", "u+1f4a9": "\ud83d\udca9",
                  "u+1f435": "\ud83d\udc35", "u+1f451": "\ud83d\udc51", "u+1f4da": "\ud83d\udcda",
                  "u+1f409": "\ud83d\udc09", "u+1f6a6": "\ud83d\udea6", "u+1f412": "\ud83d\udc12",
                  "u+1f520": "\ud83d\udd20", "u+1f38c": "\ud83c\udf8c", "u+1f346": "\ud83c\udf46", "u+264c": "\u264c",
                  "u+1f638": "\ud83d\ude38", "u+1f4e0": "\ud83d\udce0", "u+1f425": "\ud83d\udc25",
                  "u+1f6b0": "\ud83d\udeb0", "u+21aa": "\u21aa", "u+1f513": "\ud83d\udd13", "u+1f48c": "\ud83d\udc8c",
                  "u+1f348": "\ud83c\udf48", "u+1f640": "\ud83d\ude40", "u+1f6b7": "\ud83d\udeb7",
                  "u+1f44a": "\ud83d\udc4a", "u+1f4b2": "\ud83d\udcb2", "u+1f527": "\ud83d\udd27",
                  "u+1f628": "\ud83d\ude28", "u+1f434": "\ud83d\udc34", "u+1f371": "\ud83c\udf71",
                  "u+1f44f": "\ud83d\udc4f", "u+1f436": "\ud83d\udc36", "u+1f6b9": "\ud83d\udeb9", "u+27a1": "\u27a1",
                  "u+1f61c": "\ud83d\ude1c", "u+1f52e": "\ud83d\udd2e", "u+2797": "\u2797", "u+1f337": "\ud83c\udf37",
                  "u+1f6a5": "\ud83d\udea5", "u+1f428": "\ud83d\udc28", "u+1f45d": "\ud83d\udc5d",
                  "u+1f46d": "\ud83d\udc6d", "u+1f698": "\ud83d\ude98", "u+1f51f": "\ud83d\udd1f",
                  "u+1f692": "\ud83d\ude92", "u+1f483": "\ud83d\udc83", "u+1f4cc": "\ud83d\udccc",
                  "u+1f4b6": "\ud83d\udcb6", "u+1f355": "\ud83c\udf55", "u+1f552": "\ud83d\udd52",
                  "u+1f170": "\ud83c\udd70", "u+1f23a": "\ud83c\ude3a", "u+1f4b9": "\ud83d\udcb9",
                  "u+1f62d": "\ud83d\ude2d", "u+1f381": "\ud83c\udf81", "u+1f6a0": "\ud83d\udea0",
                  "u+1f453": "\ud83d\udc53", "u+1f6c3": "\ud83d\udec3", "u+1f388": "\ud83c\udf88",
                  "u+1f532": "\ud83d\udd32", "u+1f35d": "\ud83c\udf5d", "u+1f609": "\ud83d\ude09",
                  "u+1f199": "\ud83c\udd99", "u+2651": "\u2651", "u+1f636": "\ud83d\ude36", "u+1f621": "\ud83d\ude21",
                  "u+2709": "\u2709", "u+1f60f": "\ud83d\ude0f", "u+1f647": "\ud83d\ude47", "u+1f4d3": "\ud83d\udcd3",
                  "u+1f4db": "\ud83d\udcdb", "u+1f6a1": "\ud83d\udea1", "u+1f489": "\ud83d\udc89",
                  "u+1f522": "\ud83d\udd22", "u+1f4fb": "\ud83d\udcfb", "u+1f6a2": "\ud83d\udea2",
                  "u+1f64b": "\ud83d\ude4b", "u+1f3a9": "\ud83c\udfa9", "u+1f55c": "\ud83d\udd5c",
                  "u+1f684": "\ud83d\ude84", "u+1f64e": "\ud83d\ude4e", "u+1f534": "\ud83d\udd34",
                  "u+1f4ca": "\ud83d\udcca", "u+1f565": "\ud83d\udd65", "u+264d": "\u264d", "u+1f62a": "\ud83d\ude2a",
                  "u+1f551": "\ud83d\udd51", "u+2122": "\u2122", "u+1f38e": "\ud83c\udf8e", "u+1f44e": "\ud83d\udc4e",
                  "u+1f42b": "\ud83d\udc2b", "u+1f3bb": "\ud83c\udfbb", "u+2755": "\u2755", "u+1f630": "\ud83d\ude30",
                  "u+2728": "\u2728", "u+2196": "\u2196", "u+270f": "\u270f", "u+1f413": "\ud83d\udc13",
                  "u+1f406": "\ud83d\udc06", "u+1f408": "\ud83d\udc08", "u+1f550": "\ud83d\udd50",
                  "u+1f344": "\ud83c\udf44", "u+1f237": "\ud83c\ude37", "u+1f490": "\ud83d\udc90",
                  "u+1f50d": "\ud83d\udd0d", "u+1f4f6": "\ud83d\udcf6", "u+1f49d": "\ud83d\udc9d",
                  "u+1f4c2": "\ud83d\udcc2", "u+1f300": "\ud83c\udf00", "u+1f4ec": "\ud83d\udcec",
                  "u+1f38a": "\ud83c\udf8a", "u+2795": "\u2795", "u+1f4bd": "\ud83d\udcbd", "u+1f3ca": "\ud83c\udfca",
                  "u+1f612": "\ud83d\ude12", "u+26ea": "\u26ea", "u+23eb": "\u23eb", "u+2733": "\u2733",
                  "u+1f3ed": "\ud83c\udfed", "u+1f444": "\ud83d\udc44", "u+1f306": "\ud83c\udf06",
                  "u+1f495": "\ud83d\udc95", "u+1f45f": "\ud83d\udc5f", "u+2693": "\u2693", "u+1f4f0": "\ud83d\udcf0",
                  "u+1f615": "\ud83d\ude15", "u+0039 u+20e3": "9\u20e3", "u+2b1b": "\u2b1b", "u+2650": "\u2650",
                  "u+1f198": "\ud83c\udd98", "u+1f3bc": "\ud83c\udfbc", "u+1f4a5": "\ud83d\udca5",
                  "u+1f511": "\ud83d\udd11", "u+2199": "\u2199", "u+1f349": "\ud83c\udf49", "u+1f454": "\ud83d\udc54",
                  "u+1f370": "\ud83c\udf70", "u+2757": "\u2757", "u+1f332": "\ud83c\udf32", "u+1f4a2": "\ud83d\udca2",
                  "u+1f535": "\ud83d\udd35", "u+1f368": "\ud83c\udf68", "u+1f512": "\ud83d\udd12",
                  "u+1f68c": "\ud83d\ude8c", "u+1f60e": "\ud83d\ude0e", "u+1f35b": "\ud83c\udf5b",
                  "u+1f622": "\ud83d\ude22", "u+1f4d5": "\ud83d\udcd5", "u+1f4b8": "\ud83d\udcb8", "u+2652": "\u2652",
                  "u+231b": "\u231b", "u+1f36e": "\ud83c\udf6e", "u+0031 u+20e3": "1\u20e3", "u+1f450": "\ud83d\udc50",
                  "u+1f64f": "\ud83d\ude4f", "u+1f30e": "\ud83c\udf0e", "u+1f305": "\ud83c\udf05", "u+260e": "\u260e",
                  "u+1f605": "\ud83d\ude05", "u+1f384": "\ud83c\udf84", "u+1f46c": "\ud83d\udc6c",
                  "u+1f635": "\ud83d\ude35", "u+1f4c8": "\ud83d\udcc8", "u+1f41e": "\ud83d\udc1e",
                  "u+1f3ee": "\ud83c\udfee", "u+1f610": "\ud83d\ude10", "u+1f4ed": "\ud83d\udced",
                  "u+1f43d": "\ud83d\udc3d", "u+1f685": "\ud83d\ude85", "u+26fa": "\u26fa", "u+2660": "\u2660",
                  "u+1f416": "\ud83d\udc16", "u+1f35c": "\ud83c\udf5c", "u+1f51b": "\ud83d\udd1b",
                  "u+1f61e": "\ud83d\ude1e", "u+1f646": "\ud83d\ude46", "u+1f625": "\ud83d\ude25",
                  "u+1f4bf": "\ud83d\udcbf", "u+1f3a8": "\ud83c\udfa8", "u+1f4d1": "\ud83d\udcd1",
                  "u+1f4d8": "\ud83d\udcd8", "u+1f4c0": "\ud83d\udcc0", "u+2649": "\u2649", "u+1f3ac": "\ud83c\udfac",
                  "u+26f5": "\u26f5", "u+1f40a": "\ud83d\udc0a", "u+1f6ad": "\ud83d\udead", "u+1f49a": "\ud83d\udc9a",
                  "u+1f404": "\ud83d\udc04", "u+1f45e": "\ud83d\udc5e", "u+1f250": "\ud83c\ude50", "u+26ab": "\u26ab",
                  "u+1f3a4": "\ud83c\udfa4", "u+1f3a5": "\ud83c\udfa5", "u+26fd": "\u26fd", "u+1f31c": "\ud83c\udf1c",
                  "u+1f3b7": "\ud83c\udfb7", "u+1f46b": "\ud83d\udc6b", "u+1f232": "\ud83c\ude32",
                  "u+1f4a3": "\ud83d\udca3", "u+1f3b6": "\ud83c\udfb6", "u+1f5fb": "\ud83d\uddfb", "u+00ae": "\u00ae",
                  "u+1f4ad": "\ud83d\udcad", "u+1f301": "\ud83c\udf01", "u+1f554": "\ud83d\udd54",
                  "u+1f415": "\ud83d\udc15", "u+1f505": "\ud83d\udd05", "u+1f463": "\ud83d\udc63", "u+3299": "\u3299",
                  "u+1f34d": "\ud83c\udf4d", "u+1f430": "\ud83d\udc30", "u+1f4b5": "\ud83d\udcb5", "u+26aa": "\u26aa",
                  "u+1f17f": "\ud83c\udd7f", "u+26f3": "\u26f3", "u+2796": "\u2796", "u+2716": "\u2716",
                  "u+2139": "\u2139", "u+1f314": "\ud83c\udf14", "u+1f63a": "\ud83d\ude3a", "u+1f632": "\ud83d\ude32",
                  "u+1f1ea u+1f1f8": "\ud83c\uddea\ud83c\uddf8", "u+1f251": "\ud83c\ude51", "u+25fc": "\u25fc",
                  "u+1f4f3": "\ud83d\udcf3", "u+1f631": "\ud83d\ude31", "u+1f19a": "\ud83c\udd9a",
                  "u+1f443": "\ud83d\udc43", "u+1f33d": "\ud83c\udf3d", "u+1f419": "\ud83d\udc19",
                  "u+1f302": "\ud83c\udf02", "u+1f3b9": "\ud83c\udfb9", "u+1f484": "\ud83d\udc84",
                  "u+1f48b": "\ud83d\udc8b", "u+1f686": "\ud83d\ude86", "u+0033 u+20e3": "3\u20e3",
                  "u+1f3ae": "\ud83c\udfae", "u+1f308": "\ud83c\udf08", "u+1f50b": "\ud83d\udd0b",
                  "u+1f6a7": "\ud83d\udea7", "u+1f606": "\ud83d\ude06", "u+1f464": "\ud83d\udc64",
                  "u+1f35a": "\ud83c\udf5a", "u+1f6bd": "\ud83d\udebd", "u+2600": "\u2600", "u+1f410": "\ud83d\udc10"}
LATIN_CHARS = {"\\xe2\\x80\\x9e": "\"", "\\xe2\\x80\\xb4": "'", "\\xe2\\x80\\x9d": "\"", "\\xe2\\x80\\x90": "-",
               "\\xe2\\x80\\x9f": "\"", "\\xe2\\x80\\xb7": "'", "\\xe2\\x80\\x9b": "'", "\\xe2\\x80\\xb3": "'",
               "\\xe2\\x81\\xbe": ")", "\\xe2\\x80\\x9c": "\"", "\\xe2\\x80\\x92": "-", "\\xe2\\x80\\x99": "'",
               "\\xe2\\x81\\xbc": "=", "\\xe2\\x81\\xba": "+", "\\xc3\\xa9": "e", "\\xe2\\x80\\x91": "-",
               "\\xe2\\x81\\xbb": "-", "\\xe2\\x80\\x94": "-", "\\xe2\\x80\\x98": "'", "\\xe2\\x80\\xb2": "'",
               "\\xe2\\x80\\xa6": "...", "\\xe2\\x80\\x93": "-", "\\xe2\\x81\\xbd": "(", "\\xe2\\x80\\xb6": "'",
               "\\xe2\\x80\\xb5": "'"}

EMOJIS_UTF_RE = re.compile(r"\\x", re.IGNORECASE)
EMOJIS_UNICODE_RE = re.compile(r"u\+", re.IGNORECASE)
EMOJIS_UTF_NOSPACE_RE = re.compile(r'(?<!x..)(\\x)', re.IGNORECASE)
EMOJIS_UNICODE_NOSPACE_RE = re.compile(r'(\D{2,})(U\+)', re.IGNORECASE)
LATIN_CHARS_RE = re.compile(r'\\xe2\\', re.IGNORECASE)
CHEMICALS = []

EMOJIS_UTF_PATS = {}
for key, value in EMOJIS_UTF.items():
    EMOJIS_UTF_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)
EMOJIS_UNICODE_PATS = {}
for key, value in EMOJIS_UNICODE.items():
    EMOJIS_UNICODE_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)
LATIN_CHARS_PATS = {}
for key, value in LATIN_CHARS.items():
    LATIN_CHARS_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)


def fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False,
                             print_png=True):
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(title=title, plot_bgcolor='rgb(255,255,255)',
                      yaxis=dict(
                          title=yaxis_title,
                          titlefont_size=18,
                          tickfont_size=18,
                          showgrid=showgrid,
                      ),
                      xaxis=dict(
                          title=xaxis_title,
                          titlefont_size=18,
                          tickfont_size=16,
                          showgrid=showgrid
                      ),
                      font=dict(
                          size=14
                      ),
                      showlegend=showlegend)
    if showlegend:
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.2,  # 0.01
            xanchor="right",  # "left", #  "right"
            x=1,    #0.01,  # 0.99
            bordercolor="Black",
            borderwidth=0.3,
            font=dict(
                size=14,
    )))

    pio.write_html(fig, savename, auto_open=False)
    pio.write_image(fig, savename.replace(".html", ".pdf"), engine="kaleido")
    if print_png:
        try:
            pio.write_image(fig, savename.replace("html", "png"), width=1024, height=571, scale=1)
            pio.write_image(fig, savename.replace("html", "eps"), width=1024, height=570, scale=1)
        except:
            logging.info("Cannot save plotly static image for timeseries - saved only HTML")

def plot_timeseries_with_confidence(timeseries, ts_to_plot, dates, xaxis_title, yaxis_title, title,
                                                      name="", savename="", lcolor=["blue"], dashed=["dash"],
                                    applyfn=None, smoothing_win=None, bounds=[], yrange=[], ci_window_smooth=1):
    """

    :param timeseries: timeseries wrt to which we will get the confidence intervals
    :param ts_to_plot: list of time series to be plotted together with the confidence intervals
    :param dates:
    :param xaxis_title:
    :param yaxis_title:
    :param title:
    :param name: list of names of the time series in ts_to_plot
    :param savename:
    :param lcolor:
    :return: plot confidence intervals that are the smoothed upper and lower bounds of the daily summary
    """
    fig = go.Figure()
    if lcolor != "":
        line_dict = dict(color=lcolor)
    else:
        line_dict = None

    smooth_n = ci_window_smooth
    # get daily CIs and then smooth them
    dataseries = pd.Series(timeseries)
    if applyfn is None:
        daily_summary_5p = dataseries.quantile(0.05, interpolation="lower")
        lower_confidence_band = daily_summary_5p.rolling(smooth_n*smoothing_win).mean().dropna()
        daily_summary_95p = dataseries.quantile(0.95, interpolation="lower")
        upper_confidence_band = daily_summary_95p.rolling(smooth_n*smoothing_win).mean().dropna()
    else:
        daily_summary_5p = dataseries.apply(lambda x: np.quantile(x, 0.05, interpolation="lower"))
        lower_confidence_band = daily_summary_5p.rolling(smooth_n*smoothing_win).mean().dropna()
        daily_summary_95p = dataseries.apply(lambda x: np.quantile(x, 0.95, interpolation="lower"))
        upper_confidence_band = daily_summary_95p.rolling(smooth_n*smoothing_win).mean().dropna()

    dates = dates[(smooth_n-1) * smoothing_win:]
    ts_to_plot[1] = ts_to_plot[1][(smooth_n-1) * smoothing_win:]
    ts_to_plot[0] = ts_to_plot[0][(smooth_n-1) * smoothing_win:]

    # color the rolling median
    if len(bounds) > 0:
        time = np.array(dates)
        k = 0
        curr_trace = []
        curr_time = [time[0]]
        plotk = ts_to_plot[1][k]
        if plotk > bounds[1]:
            curr_col = "Green"
        elif plotk < bounds[1]:
            curr_col = "Red"
        else:
            curr_col = "Blue"
        while k < len(time):
            plotk = ts_to_plot[1][k]
            if plotk > bounds[1]:
                new_col = "Green"
            elif plotk < bounds[1]:
                new_col = "Red"
            else:
                new_col = "Blue"
            if new_col == curr_col:
                curr_trace.append(plotk)
                curr_time.append(time[k])
                k = k + 1
            else:
                # plot for continuous visual even though in different color class
                # curr_trace.append(plotk)
                # curr_time.append(time[k])
                if curr_col != "Blue":
                    curr_trace.append(plotk)
                    curr_time.append(time[k])
                    fig.add_trace(go.Scatter(
                        x=curr_time,
                        y=curr_trace,
                        showlegend=False, mode="lines", line=dict(color=curr_col)
                    ))
                    plotk = ts_to_plot[1][k]
                    curr_trace = []
                    curr_time = []
                else:
                    fig.add_trace(go.Scatter(
                        x=curr_time[:-1],
                        y=curr_trace[:-1],
                        showlegend=False, mode="lines", line=dict(color=curr_col)
                    ))

                    plotk = ts_to_plot[1][k]
                    curr_trace = [ts_to_plot[1][k - 1]]
                    curr_time = [time[k - 1]]
                    curr_trace.append(plotk)
                    curr_time.append(time[k])
                if plotk > bounds[1]:
                    curr_col = "Green"
                elif plotk < bounds[1]:
                    curr_col = "Red"
                else:
                    curr_col = "Blue"
                k = k + 1

        fig.add_trace(go.Scatter(
            x=curr_time,
            y=curr_trace,
            showlegend=True, name=name[1], mode="lines", line=dict(color=curr_col)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=dates,
            y=ts_to_plot[1],
            name=name[1], mode="lines", line=line_dict
        ))

    # CI
    fig.add_trace(go.Scatter(
        name='',
        x=dates,
        y=upper_confidence_band.values,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        name='',
        x=dates,
        y=lower_confidence_band.values,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ))

    if line_dict != None:
        line_dict_mean = line_dict
    else:
        line_dict_mean = dict()
    line_dict_mean["dash"] = "dash"
    line_dict_mean["color"] = "black"
    fig.add_trace(go.Scatter(
        x=dates,
        y=ts_to_plot[0],
        name=name[0], mode="lines", line=line_dict_mean
    ))

    if len(yrange) > 0:
        fig.update_yaxes(range=yrange)

    if len(bounds) > 0:
        # lower bound
        xrng = dates
        fig.add_shape(type='line',
                      x0=xrng[0],
                      y0=bounds[0],
                      x1=xrng[-1],
                      y1=bounds[0],
                      line=dict(color='Red', dash='dot' ),
                      xref='x',
                      yref='y'
                      )
        # upper bound
        fig.add_shape(type='line',
                      x0=xrng[0],
                      y0=bounds[2],
                      x1=xrng[-1],
                      y1=bounds[2],
                      line=dict(color='Green', dash='dot'),
                      xref='x',
                      yref='y'
                      )

        # neutral line
        fig.add_shape(type='line',
                      x0=xrng[0],
                      y0=bounds[1],
                      x1=xrng[-1],
                      y1=bounds[1],
                      line=dict(color='Blue', dash='dot' ),
                      xref='x',
                      yref='y'
                      )

    fix_plot_layout_and_save(fig, savename, xaxis_title=xaxis_title, yaxis_title=yaxis_title, title=title,
                              showgrid=False, showlegend=True)

    return lower_confidence_band, upper_confidence_band

def save_sparse_oneoff(sparse_mat1, f):
    """

    :param sparse_mat:
    :param f: filename
    :return:
    """
    try:
        sparse_mat = sp.csr_matrix(sparse_mat1)
    except:
        print("Exception")
        sparse_mat = sp.csr_matrix(sparse_mat1.any().toarray().flatten())
    sparse_list = [sparse_mat.data, sparse_mat.indices, sparse_mat.indptr, sparse_mat.shape]
    with open(f, "wb") as ff:
        pickle.dump(sparse_list, ff, pickle.HIGHEST_PROTOCOL)

    return 0

def load_sparse_oneoff(f):
    """
    :return:
    """

    try:
        with open(f, "rb") as ff:
            sparse_list = pickle.load(ff)
        data = sparse_list[0]
        indices = sparse_list[1]
        indptr = sparse_list[2]
        shape = sparse_list[3]
        sparse_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
    except:
        sparse_matrix = None

    return sparse_matrix

def get_sentiment_counts(ngram, counts, sent_pos, sent_neg, sent_neutral):
    """
    If a token is not in any of the lists, then its count will be determined by 'diff'
    if diff is zero tokens go to neutral as one of the two happens:
        Either none of the tokens in the ngram are in any of the lists
        Or cnt_pos=cnt_neg and then if a token has a count of zero after weighting, it goes to neutral

        Note that we can have cnt_pos=cnt_neg AND token_count + 0 != 0. Then the ngram has to be considered
        neutral and not positive. This is captured by the if diff=0.. below, and the last condition.

    :param ngram:
    :param counts: list containing a sparse matrix with the ngram counts
    :param sent_pos:
    :param sent_neg:
    :param sent_neutral:
    :return:
    """

    cnt_pos = 0
    cnt_neg = 0

    for i in range(len(ngram)):
        if ngram[i].lower() in sent_pos:
            cnt_pos += 1
        if ngram[i].lower() in sent_neg:
            cnt_neg += 1
        # comment out following 3 lines when running with baseline vocab unless we have a baseline neutral dictionary
        if ngram[i].lower() in sent_neutral:
            cnt_pos += 1
            cnt_neg += 1

    diff = cnt_pos - cnt_neg
    data = counts.data.copy()
    data += diff
    new_data = []
    new_idx = []
    new_ptr = 0
    neutral_data = []
    neutral_idx = []
    neutral_ptr = 0
    pos_data = []
    pos_idx = []
    pos_ptr = 0
    neg_data = []
    neg_idx = []
    neg_ptr = 0
    for i in range(len(data)):
        if diff == 0:
            neutral_data.append(counts.data[i])
            neutral_idx.append(counts.indices[i])
            neutral_ptr += 1
        else:
            if data[i] != 0:
                new_data.append(data[i])
                new_idx.append(counts.indices[i])
                new_ptr += 1
                if data[i] > 0:
                    pos_data.append(data[i])
                    pos_idx.append(counts.indices[i])
                    pos_ptr += 1
                else:
                    neg_data.append(abs(data[i]))
                    neg_idx.append(counts.indices[i])
                    neg_ptr += 1
            else:
                # Use the original counts, since all weighted are zeros.
                # The distribution of the neutral words is expected to be uniform since all have a count of 'diff'
                neutral_data.append(counts.data[i])
                neutral_idx.append(counts.indices[i])
                neutral_ptr += 1
    new_counts = sp.csr_matrix((new_data, new_idx, np.array([0, new_ptr])), shape=counts.shape)
    """
    print("Before: {}".format(data))
    print("After: {}".format(new_data))
    print("Pos: {}".format(pos_data))
    print("Neg: {}".format(neg_data))
    print("Neutral: {}".format(neutral_data))
    """
    if len(pos_idx) > 0:
        pos_counts = sp.csr_matrix((pos_data, pos_idx, np.array([0, pos_ptr])), shape=counts.shape)
    else:
        pos_counts = []
    if len(neg_idx) > 0:
        neg_counts = sp.csr_matrix((neg_data, neg_idx, np.array([0, neg_ptr])), shape=counts.shape)
    else:
        neg_counts = []
    # neutral are those words that had non zero count before the weighting but have zero count after
    if len(neutral_idx) > 0:
        neutral_counts = sp.csr_matrix((neutral_data, neutral_idx, np.array([0, neutral_ptr])), shape=counts.shape)
    else:
        neutral_counts = []

    # return in same format as counts
    return new_counts, pos_counts, neg_counts, neutral_counts, diff

def load_config():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json',
                        help='configuration file')

    args = parser.parse_args()
    print(args)

    with open(args.config, 'rt') as config_file:
        cfg = json.load(config_file)

    return cfg

def retain_unchanged(stopwords_f, compounds_f, acronyms_f, latin_f, chemicals_f, metric_units_f):

    keep_untokenized = []

    compounds = []
    if compounds_f is not None:
        with open(compounds_f, 'r') as f:
            for i, line in enumerate(f):
                compounds.append(line.strip('\n'))
        keep_untokenized.extend(compounds)
    acronyms = []
    if acronyms_f is not None:
        with open(acronyms_f, 'r') as f:
            for i, line in enumerate(f):
                acronyms.append(line.strip('\n').lower())
        keep_untokenized.extend(acronyms)
    latin = []
    if latin_f is not None:
        with open(latin_f, 'r') as f:
            for i, line in enumerate(f):
                latin.append(line.strip('\n').lower())
        keep_untokenized.extend(latin)
    chemicals = []
    if chemicals_f is not None:
        with open(chemicals_f, 'r') as f:
            for i, line in enumerate(f):
                chemicals.append(line.strip('\n').lower())
        keep_untokenized.extend(chemicals)
    metricUn = []
    if metric_units_f is not None:
        with open(metric_units_f, 'r') as f:
            for i, line in enumerate(f):
                metricUn.append(line.strip('\n'))
        keep_untokenized.extend(metricUn)

    stopwords = []
    if stopwords_f is not None:
        with open(stopwords_f, 'r') as f:
            for i, line in enumerate(f):
                stopwords.append(line.strip('\n'))

    return keep_untokenized, stopwords

def run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    if char in string.punctuation:
        return True

    return False

# Tokenizer tuned to work with spaCy
class CustomSpacyTokenizerCounts(object):
    """
    Parameters:

    normalize: int or bool, optional
        If not False, perform normalization of repeated charachers
        ("awesoooooome" -> "awesooome"). The value of parameter
        determines the number of occurences to keep. Defaults to 1.

    ignore_stopwords: str, list, or boolean, optional
        Whether to ignore stopwords

        - str: language to get a list of stopwords from NLTK package
        - list: list of stopwords to remove
        - True: use built-in list of the english stop words
        - False: keep all tokens

        Defaults to True

    stem: {False, 'stem', 'lemm'}, optional
        Whether to perform word stemming

        - False: do not perform word stemming
        - 'stem': use PorterStemmer from NLTK package
        - 'lemm': use WordNetLemmatizer from NLTK package

        Defaults to False

    remove_punct: bool, optional
        If True, remove punctuation tokens. Defaults to True.

    decontract: bool, optional
        If True, attempt to expand certain contractions. Defaults to True
        Example: "'ll" -> " will"

    remove_nonunicode: boolean, optional
        If True, remove all non-unicode characters. Defaults to True.

    pos_emojis, neg_emojis, neutral_emojis: None, True, or list, optional
        Replace positive, negative, and neutral emojis with the special tokens

        - None: do not perform replacement
        - True: perform replacement of the default lists of emojis with ''
        - list: list of emojis to replace

        Defaults to True


    latin_chars_fix: bool, optional
        Try applying this fix if you have a lot of \\xe2\\x80\\x99-like
        or U+1F601-like strings in your data. Defaults to True.


    special_chars: List of characters that we would normally remove but want to keep as they are part
                    of a special word (word in keep_untokenized) we want to maintain. For example, the degree symbol in ' °C '

                    Defaults to the punctuation characters and the degree symbol

    valid_chars: List of characters that are considered valid.

                Defaults to the printable set: string.printable

    remove_oov: whether to remove out of vocabulary words. Defaults to False.

    spacy_model: model to lad from spaCy with vocabulary and model for dependency parsing.
                Defaults to "en_core_web_sm"

    parse: whether or not to perform dependency parsing with the default spaCy model. Defaults to False

    """

    def __init__(self, normalize=1,
                 ignore_stopwords=True, stem=False,
                 remove_punct=True, decontract=True,
                 remove_nonunicode=True,
                 pos_emojis=None, neg_emojis=None, neutral_emojis=None,
                 latin_chars_fix=True,
                 special_chars=None, valid_chars=None,
                 remove_oov=False, spacy_model="en_core_web_sm", disable_parse=False, _normalize_chars=None, \
                 word_basis=None):

        self.params = locals()
        self._stopwords = None
        self._special_chars = special_chars
        self._valid_chars = valid_chars
        self._normalize_chars = _normalize_chars
        self._word_basis = word_basis
        self._sents = []
        self._doc = []
        self.pos_emojis = pos_emojis
        self.neg_emojis = neg_emojis
        self.neutral_emojis = neutral_emojis
        self._disable_parse = disable_parse

        if isinstance(ignore_stopwords, list):
            self._stopwords = [word.lower() for word in ignore_stopwords]
        elif ignore_stopwords is not False:
            raise TypeError('Type {} is not supported by ignore_stopwords parameter or NLTK is not installed'.format(
                type(ignore_stopwords)))

        if disable_parse != True:
            raise Exception("Tokenizer without parsing for now")

        # self._nlp = spacy.load(spacy_model, disable=["ner", "parser", "tagger"])

    def _stopword_check(self, text):
        return bool(text.lower() in self._stopwords)

    @staticmethod
    def _decontract(text):
        # Expand contractions
        for contraction, decontraction in DECONTRACTIONS.items():
            text = re.sub(contraction, decontraction, text)
        return text

    def _remove_nonunicode(self, text, character_set=string.printable):
        # remove non-unicode characters (default) or any character not in character_set
        if self.params['remove_nonunicode']:
            try:
                # Ignore flag will remove unencodable character on failure
                text = text.encode('utf-8', errors="ignore").decode('unicode-escape')
                text = ''.join(
                    filter(lambda x: x in character_set, text)).strip()
            except UnicodeDecodeError:
                print(text)
                warnings.warn(
                    'UnicodeDecodeError while trying to remove non-unicode characters')
        return text

    def __fix_chemical(self, text, kw_processor):

        tmp_text = text
        # Replace numbers in subscripts with normal font
        sub = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        spans_no_split = kw_processor.extract_keywords(tmp_text, span_info=True)
        for span in spans_no_split:
            if span[0] in CHEMICALS:
                tmp_text[span[1]:span[2]] = span[0].translate(sub)
            else:
                continue

        return tmp_text

    def __normalize(self, text, normalize):
        # Normalize repeating symbols
        tmp_text = text
        match = NORMALIZE_RE.search(tmp_text)
        while match:
            text_part = match.group(0)
            tmp_part = NORMALIZE_RE.sub(r"\1" * normalize, text_part)
            tmp_text = tmp_text.replace(text_part, tmp_part)
            match = NORMALIZE_RE.search(tmp_text)

        return tmp_text

    def __remove_email(self, text):
        tmp_text = text
        match = EMAIL_RE.search(tmp_text)
        while match:
            text_part = match.group(0)
            tmp_text = tmp_text.replace(text_part, "")
            match = EMAIL_RE.search(tmp_text)
        return tmp_text

    def __remove_url(self, text):
        tmp_text = text
        match = URLS_RE.search(tmp_text)
        while match:
            text_part = match.group(0)
            tmp_text = tmp_text.replace(text_part, "")
            match = URLS_RE.search(tmp_text)
        return tmp_text

    def __hashtag_postprocess_text(self, text):
        # Process hashtags
        tmp_text = text
        match = HASHTAGS_RE.search(tmp_text)
        while match:
            text_part = match.group(0)
            tmp_text = tmp_text.replace(text_part, "")
            match = HASHTAGS_RE.search(tmp_text)

        return tmp_text

    def _preprocess_text(self, text):

        # 'normalize' unicode and accents/quotes
        text = run_strip_accents(text, normalize=self._normalize_chars)
        # reinstate @.@, @,@ and @-@ transformations performed by dataset providers
        text = re.sub(r"([0-9]+)( @.@ )([0-9]+)", r"\1.\3", text)
        text = re.sub(r"([0-9]+)( @,@ )([0-9]+)", r"\1,\3", text)
        text = re.sub(r"(\w)( @-@ )(\w)", r"\1-\3", text)

        # expand contractions
        if self.params['decontract']:
            text = self._decontract(text)

        # convert all named and numeric character references (e.g. &gt;, &#62;, &#x3e;) in the string text
        # to the corresponding Unicode characters.
        text = html.unescape(text)

        # Remove emojis and replace multibyte latin character bytes (in latin_chars.json) with unicode character
        if self.params['latin_chars_fix']:
            if self.pos_emojis:
                if not isinstance(self.pos_emojis, list):
                    pos_emojis = POS_EMOJIS
                for emoji in pos_emojis:
                    text = text.replace(emoji, '')

            if self.neg_emojis:
                if not isinstance(self.neg_emojis, list):
                    neg_emojis = NEG_EMOJIS
                for emoji in neg_emojis:
                    text = text.replace(emoji, '')

            if self.neutral_emojis:
                if not isinstance(self.neutral_emojis, list):
                    neutral_emojis = NEUTRAL_EMOJIS
                    for emoji in neutral_emojis:
                        text = text.replace(emoji, '')

            if EMOJIS_UTF_RE.findall(text):
                text = EMOJIS_UTF_NOSPACE_RE.sub(r' \1', text)
                for utf_code, emoji in EMOJIS_UTF.items():
                    # text = EMOJIS_UTF_PATS[utf_code].sub(emoji, text)
                    text = EMOJIS_UTF_PATS[utf_code].sub('', text)

            if EMOJIS_UNICODE_RE.findall(text):
                text = EMOJIS_UNICODE_NOSPACE_RE.sub(r'\1 \2', text)
                for utf_code, emoji in EMOJIS_UNICODE.items():
                    # text = EMOJIS_UNICODE_PATS[utf_code].sub(emoji, text)
                    text = EMOJIS_UNICODE_PATS[utf_code].sub('', text)

            if LATIN_CHARS_RE.findall(text):
                for _hex, _char in LATIN_CHARS.items():
                    text = LATIN_CHARS_PATS[_hex].sub(_char, text)

        # Add 1 whitespace in front of matches in regexp
        text = re.sub(r'([:.*;,!?\(\)\[\]])', r' \1 ', text)
        # Replace 2 or more occurences of whitespace characters with 1
        text = re.sub(r'\s{2,}', ' ', text)

        # Remove quotes
        text = text.replace("'", "")
        text = text.replace('"', '')
        text = text.replace("@", "")
        # Remove newlines/tabs/returns
        text = text.replace("\n", "")
        text = text.replace("\r", "")
        text = text.replace("\t", "")

        return text.strip()

    def __postprocess_single_token(self, tok):

        if self.params['ignore_stopwords'] is not False and self._stopword_check(tok.lower()):
            # Note that stopwords that are part of a compound word will remain, e.g. track and field
            # is a single token therefore stopword 'and' remains in this context
            return False
        elif len(tok) == 1 and is_punctuation(tok) and self.params['remove_punct']:
            # remove punctuation at this stage
            return False
        elif self.params["remove_oov"] and tok.lower() not in self._word_basis:
            # Remove OOV? only reason to keep them is to allow for changing dictionary later without having to re-postprocess
            # If we want to keep punctuation and remove OOV words, make sure that punctuation
            # characters are also included in the dictionary
            # Regardless of that, only print OOV words to count for the OOV rate
            if len(tok) > 1:
                logging.info("OOV: {}".format(tok))
            return False
        elif len(tok) == 1:
            # remove single character tokens that are not punctuation - we consider them noise
            return False
        else:
            return tok

    def __postprocess_tokens__(self, tokens):

        pp_tokens = []
        # the following is a list of punctuation symbols that are usually used to join tokens together
        # or suggest an alternative. Unless the token in the basis appears with that symbol, we separate the tokens in that
        # symbol
        separators = ["-", "/"]

        for tok in tokens:
            if tok != '' and tok != ' ':  # should always be true - check just in case
                # If we have a token without hyphen proceed - else if token is not included in dictionary hyphenated
                # then remove hyphen and check each individual token
                separate = np.any([True if s in tok else False for s in separators])
                if separate:
                    for s in separators:
                        # remove s unless token is in word basis
                        if s in tok:
                            if tok.lower() in self._word_basis:
                                t = self.__postprocess_single_token(tok)
                                if t is not False and isinstance(t, str):
                                    pp_tokens.append(t)
                            else:
                                hyph = tok.replace(s, " ").strip().split()
                                if len(hyph) == 0:
                                    # could come here for single char token '-' or "/"
                                    continue
                                elif len(hyph) > 1:
                                    for h in hyph:
                                        t = self.__postprocess_single_token(h)
                                        if t is not False and isinstance(t, str):
                                            pp_tokens.append(t)
                                else:
                                    t = self.__postprocess_single_token(hyph[0])
                                    if t is not False and isinstance(t, str):
                                        pp_tokens.append(t)
                        else:
                            continue
                else:
                    t = self.__postprocess_single_token(tok)
                    if t is not False and isinstance(t, str):
                        pp_tokens.append(t)

        return pp_tokens

    def __tokenize__(self, text):

        tokens = []
        # if we don't remove OOV words at postprocess stage, .lower() below is not necessary
        spans_no_split = self._word_basis.extract_keywords(text.lower(), span_info=True)
        init = 0
        for span in spans_no_split:
            start = init
            end = span[1]
            if start != end:
                # part of text not in word_basis trie - apply non-unicode cleaning here for easier postprocessing
                nonunicode_text = self._remove_nonunicode(text[start:end], character_set=self._valid_chars)
                w = nonunicode_text.strip()
                if w != ' ' and w != '':
                    if isinstance(w.split(), list):
                        tokens.extend([ws for ws in w.split()])
                    else:
                        tokens.append(w)
            # Keyword token
            w = text[span[1]:span[2]].strip()
            if w != ' ' and w != '':
                tokens.append(w)
            init = span[2]
        if init < len(text):
            part = text[init:]
            part = self._remove_nonunicode(part, character_set=self._valid_chars)
            w = part.strip()
            if isinstance(w.split(), list):
                tokens.extend([ws for ws in w.split()])
            else:
                tokens.append(w)

        return tokens

    def tokenize(self, text):
        """
        Tokenize document

        Parameters
        ----------
        text : str
            Document to tokenize

        Returns
        -------
        tokens
            List of tokens

        """
        if not isinstance(text, str):
            warnings.warn('Document {} is not a string'.format(text))
            return []

        # Do first as preprocess insert space around dots
        text = self.__hashtag_postprocess_text(text)
        text = self.__fix_chemical(text, self._word_basis)
        text = self.__remove_url(text)
        text = self.__remove_email(text)

        # Uses self._valid_chars as list of acceptable characters
        text = self._preprocess_text(text)
        text = self.__normalize(text, self.params['normalize'])
        tokens = self.__tokenize__(text)

        sentences = self.segment_sentences(" ".join(tokens), self._word_basis)
        pp_tokens = self.__postprocess_tokens__(tokens)
        tokens = pp_tokens

        self._doc = Doc(Vocab(), words=tokens)
        self._doc._.sentences = sentences
        self._doc._.tokens = tokens

        return tokens

    def segment_sentences(self, text, kw_processor):
        """
        Simple sentence segmenter. Split text at punctuation symbols included in sentencizer_default_punct_chars.
        Exceptions: 1) if the new 'sentence' is less than 3 chars long (e.g. of min. acceptable: I am) then concatenate it with previous sentence
                    2) if a full stop is found, don't create new sentence if it is part of a set of phrases that include a dot -
                        create new sentence if next token after dot (and the space after it) is uppercase
        :param text:
        :param kw_processor:
        :return:
        """

        punct = ['.', ':', ';', '?', '!']
        sentences = []
        sent = ""
        next_sent = False
        for i in range(len(text)):
            if text[i] not in punct:
                sent += text[i]
            else:
                # sent += text[i]
                if text[i] == ".":
                    # check if it is a special word that contains . Find 2 spaces since we have inserted space around the token
                    j = i - 1
                    p = 0
                    while p < 2:
                        if text[j] == ' ':
                            p += 1
                        j = j - 1
                        if j == 0:
                            break
                    w = text[j + 1:i + 1].replace(" ", "")
                    if len(w) > 1 and (
                            w.lower() in kw_processor or w.lower() in {'mr.', 'mrs.', 'ms.', 'tel.', 'ref.', 'etc.', 'et.',
                                                               'al.', 'jan.', 'feb.', 'mar.', 'apr.', 'aug.', 'sep.',
                                                               'oct.', 'nov.', 'dec.', 'approx.', 'dept.', 'apt.',
                                                               'appt.', 'est.', 'misc.', 'e.g.', 'u.s.', 'u.s.a.', 'u.k.'}):
                        # will have double space if 2 punctuation marks are back to back
                        sent += text[i]
                    elif i < len(text) - 2:
                        # i+2 as i+1 will be the space we have inserted
                        if text[i + 2].isupper():
                            sent += text[i]
                            next_sent = True
                else:
                    sent += text[i]
                    next_sent = True
            if next_sent:
                # for pu in punct:
                #    if pu in sent:
                #        sent = sent.replace(pu,"")
                if text != ".":
                    if len(sent) < 3:
                        if len(sentences) > 0:
                            sentences[-1] += sent.strip()
                        else:
                            sentences.append(sent.strip())
                    else:
                        sentences.append(sent.strip())
                else:
                    sentences.append(sent.strip())
                sent = ""
                next_sent = False
        if len(sent) < 3:
            if len(sentences) > 0:
                sentences[-1] += sent.strip()
            else:
                sentences.append(sent.strip())
        else:
            sentences.append(sent.strip())

        return sentences


    def __call__(self, text):
        """
        Calling the tokenizer by the name of the class will return a spaCy Doc object: basic sentence segmentation (using punctuation) is performed after
        a basic cleanup using _preprocess_text(text) and the merger and  matcher transformations as defined above (_nlp(text)).
        Finally we perform stopword and punctuation removal.

        self._doc.sents contains the obtained sentences and if parsing is performed, the corresponding Token attributes will be set.
        self._doc._.tokens contains the output of the tokenize() method and these are the tokens that should be used for token-based embeddings

        :param text: document to be processed
        :return: spaCy Doc tokenized using rules above, segmented into sentences and (optionally) parsed into a dependency graph.
                 Returned document will be as it entered the segmentation phase - final kept tokens are in doc._.tokens
        """

        tokens = self.tokenize(text)

        return self._doc

def preprocessor_builder(spacy_model="en_core_web_lg",
                         spacy_disable_list=["ner", "parser", "tagger"],
                         character_set=string.printable,
                         special_chars=string.punctuation + '°',
                         disable_parse=True, remove_oov=False,
                         acronyms_f=None,
                         latin_f=None,
                         metric_units_f=None,
                         chemicals_f=None,
                         stopwords_f="./special_stopwords.txt",
                         compounds_f=None,
                         topic_basis="./data_nlp/tokenizer_data/actuarial_dictionary_trie.pickle",
                         word_basis="./data_nlp/tokenizer_data/actuarial_vocabulary.pickle"):
    _, special_stopwords = retain_unchanged(stopwords_f=stopwords_f, compounds_f=compounds_f,
                                            acronyms_f=acronyms_f, latin_f=latin_f,
                                            chemicals_f=chemicals_f, metric_units_f=metric_units_f)

    # nlp_spacy = spacy.load(spacy_model, disable=spacy_disable_list)
    stopwords = special_stopwords + list(spacy_stopwords) + ["a", "o", "u", "e", "i",
                                                             "c"]  # to remove letters left after accent removal
    valid_chars = character_set

    if topic_basis == '':
        with open(word_basis, "rb") as f:
            actuarial = list(pickle.load(f))
        kw_basis = KeywordProcessor()
        kw_basis.add_keywords_from_list(actuarial)
        with open(word_basis.replace(".pickle", "_trie.pickle"), "wb") as f:
            pickle.dump(kw_basis, f, pickle.HIGHEST_PROTOCOL)
        print("Saved keyword processor trie actuarial dictionary...")
    else:
        with open(topic_basis, "rb") as f:
            kw_basis = pickle.load(f)

    tokenizer = CustomSpacyTokenizerCounts(normalize=1,
                                           ignore_stopwords=stopwords, stem=False,
                                           remove_punct=True, decontract=True,
                                           remove_nonunicode=True,
                                           pos_emojis=True, neg_emojis=True, neutral_emojis=True,
                                           latin_chars_fix=True,
                                           special_chars=special_chars,
                                           valid_chars=valid_chars,
                                           remove_oov=remove_oov, spacy_model=spacy_model,
                                           disable_parse=disable_parse, _normalize_chars="NFC", word_basis=kw_basis)

    return tokenizer
