#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

SITEURL = 'https://dionhaefner.github.io'
RELATIVE_URLS = False

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

DELETE_OUTPUT_DIRECTORY = True

# Following items are often useful when publishing

#ISSO_URL = "//vps.dionhaefner.de/isso"
#GOOGLE_ANALYTICS = ""
PIWIK_URL = "http://home.dionhaefner.de:60000/piwik"
PIWIK_SSL_URL = "https://home.dionhaefner.de:60000/piwik"
PIWIK_SITE_ID = "1"
