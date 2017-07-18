#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

THEME = "nest-plus"

AUTHOR = 'Dion Häfner'
SITENAME = "Dion Häfner"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Berlin'

OUTPUT_SOURCES = True
OUTPUT_SOURCES_EXTENSION = ".source"

DEFAULT_LANG = 'en'

TYPOGRIFY = True
TYPOGRIFY_IGNORE_TAGS = ["style", "script", "title"]

FORMATTED_FIELDS = ['summary','_summary']
DIRECT_TEMPLATES = ['index', 'tags', 'categories', 'archives', 'home']

DISPLAY_PAGES_ON_MENU = True
INDEX_SAVE_AS = "blog.html"
HOME_SAVE_AS = "index.html"
ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{slug}/index.html'
PAGE_URL = 'index.html#{slug}'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
         ('Unsplash', 'https://unsplash.com/'),
        )

# Social widget
SOCIAL = (('github fa-3x', 'https://github.com/dionhaefner/'),)

DEFAULT_PAGINATION = 10

#
# SOCIAL SHARE
#
SOCIAL_SHARE = (("twitter", "twitter fa-2x", "Twitter"),
                ("facebook", "facebook fa-2x", "Facebook"),
                ("email", "send fa-2x", "E-Mail"))

#
# https://github.com/danielfrg/pelican-ipynb
#
MARKUP = ('md', 'ipynb')
PLUGIN_PATH = './plugins'
PLUGINS = ['ipynb.markup','share_post','summary','related_posts']
IPYNB_USE_META_SUMMARY = True
IPYNB_STRIP_CSS = True

#
# SUMMARY
#
SUMMARY_MAX_LENGTH = 100
SUMMARY_USE_FIRST_PARAGRAPH = True

#
# NEST THEME
#
SITESUBTITLE = u'Home'
NEST_INTRODUCTION_PAGE = u'introduction'
# Add items to top menu before pages
MENUITEMS = [('Blog', '/blog.html')]
# Add header background image from content/images : 'background.jpg'
NEST_HEADER_IMAGES = 'headers/index.jpeg'
NEST_HEADER_LOGO = ''
# Footer
NEST_SITEMAP_COLUMN_TITLE = u'Sitemap'
NEST_SITEMAP_MENU = [('Archives', '/archives.html'),('Categories', '/categories.html'), ('Tags','/tags.html')]
NEST_SITEMAP_ATOM_LINK = u'Atom Feed'
NEST_SITEMAP_RSS_LINK = u'RSS Feed'
NEST_SOCIAL_COLUMN_TITLE = u'Visit Me'
NEST_LINKS_COLUMN_TITLE = u'Links'
NEST_COPYRIGHT = u'&copy; Dion Häfner 2017'
# Footer optional
NEST_FOOTER_HTML = ''
# index.html
NEST_INDEX_HEAD_TITLE = u'Home'
NEST_INDEX_HEAD_SUBTITLE = u'Dion Häfner'
NEST_INDEX_HEADER_TITLE = u'Welcome to my personal homepage!'
NEST_INDEX_HEADER_SUBTITLE = u""
NEST_INDEX_CONTENT_TITLE = u'Latest Content'
# archives.html
NEST_ARCHIVES_HEAD_TITLE = u'Archives'
NEST_ARCHIVES_HEAD_DESCRIPTION = u'Posts Archives'
NEST_ARCHIVES_HEADER_TITLE = u'Archives'
NEST_ARCHIVES_HEADER_SUBTITLE = u'Archives for all posts'
NEST_ARCHIVES_CONTENT_TITLE = u'Archives'
NEST_ARCHIVES_HEADER_IMAGE = u'headers/archive.jpeg'
# article.html
NEST_ARTICLE_HEADER_BY = u'By'
NEST_ARTICLE_HEADER_MODIFIED = u'modified'
NEST_ARTICLE_HEADER_IN = u'in category'
NEST_ARTICLE_DEFAULT_HEADER_IMAGE = "headers/blog.jpeg"
# author.html
NEST_AUTHOR_HEAD_TITLE = u'Posts by'
NEST_AUTHOR_HEAD_DESCRIPTION = u'Posts by'
NEST_AUTHOR_HEADER_SUBTITLE = u'Posts archives'
NEST_AUTHOR_CONTENT_TITLE = u'Posts'
# authors.html
NEST_AUTHORS_HEAD_TITLE = u'Author list'
NEST_AUTHORS_HEAD_DESCRIPTION = u'Author list'
NEST_AUTHORS_HEADER_TITLE = u'Author list'
NEST_AUTHORS_HEADER_SUBTITLE = u'Archives listed by author'
# blog.html
NEST_BLOG_HEADER_IMAGE = u'headers/blog.jpeg'
NEST_BLOG_HEAD_TITLE = u'Blog'
NEST_BLOG_HEAD_DESCRIPTION = u'Blog'
NEST_BLOG_HEADER_TITLE = u'Blog'
NEST_BLOG_HEADER_SUBTITLE = u''
NEST_BLOG_LATESTPOSTS_TITLE = u'Latest Posts'
# categories.html
NEST_CATEGORIES_HEAD_TITLE = u'Categories'
NEST_CATEGORIES_HEAD_DESCRIPTION = u'Archives listed by category'
NEST_CATEGORIES_HEADER_TITLE = u'Categories'
NEST_CATEGORIES_HEADER_SUBTITLE = u'Archives listed by category'
# category.html
NEST_CATEGORY_HEAD_TITLE = u'Category Archive'
NEST_CATEGORY_HEAD_DESCRIPTION = u'Category Archive'
NEST_CATEGORY_HEADER_TITLE = u'Category'
NEST_CATEGORY_HEADER_SUBTITLE = u'Category Archive'
# pagination.html
NEST_PAGINATION_PREVIOUS = u'Previous'
NEST_PAGINATION_NEXT = u'Next'
# period_archives.html
NEST_PERIOD_ARCHIVES_HEAD_TITLE = u'Archives for'
NEST_PERIOD_ARCHIVES_HEAD_DESCRIPTION = u'Archives for'
NEST_PERIOD_ARCHIVES_HEADER_TITLE = u'Archives'
NEST_PERIOD_ARCHIVES_HEADER_SUBTITLE = u'Archives for'
NEST_PERIOD_ARCHIVES_CONTENT_TITLE = u'Archives for'
# tag.html
NEST_TAG_HEAD_TITLE = u'Tag archives'
NEST_TAG_HEAD_DESCRIPTION = u'Tag archives'
NEST_TAG_HEADER_TITLE = u'Tag'
NEST_TAG_HEADER_SUBTITLE = u'Tag archives'
# tags.html
NEST_TAGS_HEAD_TITLE = u'Tags'
NEST_TAGS_HEAD_DESCRIPTION = u'Tags List'
NEST_TAGS_HEADER_TITLE = u'Tags'
NEST_TAGS_HEADER_SUBTITLE = u'Tags List'
NEST_TAGS_CONTENT_TITLE = u'Tags List'
NEST_TAGS_CONTENT_LIST = u'tagged'
# Static files
STATIC_PATHS = ['images', 'downloads', 'extra/robots.txt', 'extra/README', 'extra/CNAME']
EXTRA_PATH_METADATA = {
    'extra/robots.txt': {'path': 'robots.txt'},
    'extra/README': {'path': 'README.md'},
    'extra/CNAME': {'path': 'CNAME'}
}
