#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

from pelican_jupyter import markup as nb_markup

THEME = "maxwell"

AUTHOR = "Dion Häfner"
SITENAME = "dionhaefner.github.io"
SITEURL = ""
SITESUBTITLE = "Maximum entropy"

PATH = "content"

TIMEZONE = "Europe/Stockholm"

OUTPUT_SOURCES = True
OUTPUT_SOURCES_EXTENSION = ".source"

DEFAULT_LANG = "en"
LOCALE = ["en_US.utf8", "en_US", "usa"]

TYPOGRIFY = True
TYPOGRIFY_IGNORE_TAGS = ["style", "script", "title", "code", "pre"]

FORMATTED_FIELDS = ["summary", "_summary"]

ARTICLE_URL = "{date:%Y}/{date:%m}/{slug}/"
ARTICLE_SAVE_AS = "{date:%Y}/{date:%m}/{slug}/index.html"

DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = True

DEFAULT_PAGINATION = 20

# Remove unused templates
# DIRECT_TEMPLATES = ['index']

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None


#
# https://github.com/danielfrg/pelican-ipynb
#
MARKUP = ("md", "ipynb")
PLUGIN_PATHS = ["./plugins"]
PLUGINS = ["related_posts", nb_markup]

IGNORE_FILES = [".ipynb_checkpoints"]

IPYNB_EXPORT_TEMPLATE = "base"
IPYNB_SKIP_CSS = True

#
# Maxwell theme
#

import os
JINJA_FILTERS = {"basename": os.path.basename}
MAXWELL_ABOUT_PAGE = "introduction.md"
MAXWELL_ABOUT_IMAGE_BRIGHT = "images/logo-bright.png"
MAXWELL_ABOUT_IMAGE_DARK = "images/logo-dark.png"

MAXWELL_SINGLEAUTHOR = True

if MAXWELL_SINGLEAUTHOR:
    AUTHOR_SAVE_AS = ''

MAXWELL_COPYRIGHT = u"&copy; Dion Häfner 2016-2021"

# Metadata
MAXWELL_META_IMAGE = "images/logo-bright.png"
MAXWELL_TWITTER_CARD_TAG = "dionhaefner"

# Static files
STATIC_PATHS = ["images", "downloads", "extra/robots.txt", "extra/README"]
EXTRA_PATH_METADATA = {
    "extra/robots.txt": {"path": "robots.txt"},
    "extra/README.md": {"path": "README.md"}
}
