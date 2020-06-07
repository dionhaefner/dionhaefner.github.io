#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

from pelican_jupyter import markup as nb_markup

THEME = "nest-plus"

AUTHOR = "Dion Häfner"
SITENAME = "Dion Häfner"
SITEURL = ""

PATH = "content"

TIMEZONE = "Europe/Stockholm"

OUTPUT_SOURCES = True
OUTPUT_SOURCES_EXTENSION = ".source"

DEFAULT_LANG = "en"
LOCALE = ["en_US.utf8", "en_US", "usa"]

TYPOGRIFY = True
TYPOGRIFY_IGNORE_TAGS = ["style", "script", "title"]

FORMATTED_FIELDS = ["summary", "_summary"]
DIRECT_TEMPLATES = ["index", "tags", "archives", "onepage_simple"]

DISPLAY_PAGES_ON_MENU = True
ONEPAGE_SIMPLE_SAVE_AS = "index.html"
INDEX_SAVE_AS = "blog.html"
ARTICLE_URL = "{date:%Y}/{date:%m}/{slug}/"
ARTICLE_SAVE_AS = "{date:%Y}/{date:%m}/{slug}/index.html"
PAGE_URL = "index.html#{slug}"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
         ("Unsplash", "https://unsplash.com/"),
        )

# Social widget
SOCIAL = (("github fa-2x", "https://github.com/dionhaefner/"),
          ("linkedin fa-2x", "https://www.linkedin.com/in/dion-h%C3%A4fner-763821121/"),
          ("twitter fa-2x", "https://twitter.com/dionhaefner"))

DEFAULT_PAGINATION = 6

#
# SOCIAL SHARE
#
SOCIAL_SHARE = (("twitter", "twitter fa-2x", "Twitter"),
                ("facebook", "facebook fa-2x", "Facebook"),
                ("email", "send fa-2x", "E-Mail"))

#
# Meta data
#
NEST_META_IMAGE = "images/logo.png"
NEST_TWITTER_CARD_TAG = "dionhaefner"

#
# https://github.com/danielfrg/pelican-ipynb
#
MARKUP = ("md", "ipynb")
PLUGIN_PATHS = ["./plugins"]
PLUGINS = ["share_post", "summary", "related_posts", nb_markup]

IGNORE_FILES = [".ipynb_checkpoints"]
IPYNB_SKIP_CSS = True

#
# SUMMARY
#
SUMMARY_MAX_LENGTH = 100
SUMMARY_USE_FIRST_PARAGRAPH = True

#
# NEST THEME
#
SITESUBTITLE = u"Home"
NEST_SINGLEAUTHOR = True
# Add items to top menu before pages
MENUITEMS = [("Blog", "/blog.html")]
# Add header background image from content/images : "background.jpg"
NEST_HEADER_IMAGES = "headers/index.jpeg"
# Footer
NEST_SITEMAP_COLUMN_TITLE = u"Sitemap"
NEST_SITEMAP_MENU = [
    ("Archives", "/archives.html"),
    ("Tags", "/tags.html"),
]
NEST_SITEMAP_ATOM_LINK = u"Atom Feed"
NEST_SITEMAP_RSS_LINK = u"RSS Feed"
NEST_SOCIAL_COLUMN_TITLE = u"Visit Me"
NEST_LINKS_COLUMN_TITLE = u"Links"
NEST_COPYRIGHT = u"&copy; Dion Häfner 2016-2020"
# Footer optional
NEST_FOOTER_HTML = ""
# onepage.html
NEST_ONEPAGE_HEAD_TITLE = u"About Me"
NEST_ONEPAGE_HEAD_SUBTITLE = u"Dion Häfner"
NEST_ONEPAGE_HEADER_TITLE = u"About Me"
NEST_ONEPAGE_HEADER_SUBTITLE = u""
NEST_ONEPAGE_META_DESCRIPTION = "Learn more about me, my work, and my reasearch."
NEST_ONEPAGE_CONTENT_TITLE = u"Latest Content"
# archives.html
NEST_ARCHIVES_HEAD_TITLE = u"Archives"
NEST_ARCHIVES_HEAD_DESCRIPTION = u"Posts Archives"
NEST_ARCHIVES_HEADER_TITLE = u"Archives"
NEST_ARCHIVES_HEADER_SUBTITLE = u"Archives for all posts"
NEST_ARCHIVES_CONTENT_TITLE = u"Archives"
NEST_ARCHIVES_HEADER_IMAGE = u"headers/archive.jpeg"
# article.html
NEST_ARTICLE_HEADER_BY = u"By"
NEST_ARTICLE_HEADER_MODIFIED = u"modified"

if NEST_SINGLEAUTHOR:
    AUTHOR_SAVE_AS = False
else:
    # author.html
    NEST_AUTHOR_HEAD_TITLE = u"Posts by"
    NEST_AUTHOR_HEAD_DESCRIPTION = u"Posts by"
    NEST_AUTHOR_HEADER_SUBTITLE = u"Posts archives"
    NEST_AUTHOR_CONTENT_TITLE = u"Posts"

    # authors.html
    NEST_AUTHORS_HEAD_TITLE = u"Author list"
    NEST_AUTHORS_HEAD_DESCRIPTION = u"Author list"
    NEST_AUTHORS_HEADER_TITLE = u"Author list"
    NEST_AUTHORS_HEADER_SUBTITLE = u"Archives listed by author"

# index.html
NEST_BLOG_HEADER_IMAGE = u"headers/plant.jpeg"
NEST_BLOG_DEFAULT_HEADER_IMAGE = u""
NEST_BLOG_HEAD_TITLE = u"The Terrarium"
NEST_BLOG_HEAD_DESCRIPTION = u"The Terrarium"
NEST_BLOG_HEADER_TITLE = u"The Terrarium"
NEST_BLOG_HEADER_SUBTITLE = u"A Blog in wild growth."
NEST_BLOG_LATESTPOSTS_TITLE = u"Latest Posts"
# pagination.html
NEST_PAGINATION_PREVIOUS = u"Previous"
NEST_PAGINATION_NEXT = u"Next"
# period_archives.html
NEST_PERIOD_ARCHIVES_HEAD_TITLE = u"Archives for"
NEST_PERIOD_ARCHIVES_HEAD_DESCRIPTION = u"Archives for"
NEST_PERIOD_ARCHIVES_HEADER_TITLE = u"Archives"
NEST_PERIOD_ARCHIVES_HEADER_SUBTITLE = u"Archives for"
NEST_PERIOD_ARCHIVES_CONTENT_TITLE = u"Archives for"
# tag.html
NEST_TAG_HEAD_TITLE = u"Tag archives"
NEST_TAG_HEAD_DESCRIPTION = u"Tag archives"
NEST_TAG_HEADER_TITLE = u"Tag"
NEST_TAG_HEADER_SUBTITLE = u"Tag archives"
# tags.html
NEST_TAGS_HEAD_TITLE = u"Tags"
NEST_TAGS_HEAD_DESCRIPTION = u"Tags List"
NEST_TAGS_HEADER_TITLE = u"Tags"
NEST_TAGS_HEADER_SUBTITLE = u"Tags List"
NEST_TAGS_CONTENT_TITLE = u"Tags List"
NEST_TAGS_CONTENT_LIST = u"tagged"
# Static files
STATIC_PATHS = ["images", "downloads", "extra/robots.txt", "extra/README"]
EXTRA_PATH_METADATA = {
    "extra/robots.txt": {"path": "robots.txt"},
    "extra/README": {"path": "README.md"}
}
