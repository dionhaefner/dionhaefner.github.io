# Nest-Plus

This is a theme for [Pelican](http://getpelican.com) 3.5+, a static site generator written in Python.

I have created this theme for [my personal website](http://dionhaefner.github.io). It is heavily based on Matthieu Olivier's [Nest](https://github.com/molivier/nest) theme, although I have changed quite a lot to provide the advanced functionality I wanted.

## New features

- All pages are bundled into one large landing page with a responsive one-page layout
- Dark variant of the theme, used for the landing page
- Custom CSS for some widgets such as:
    * A tiled portfolio view
    * Static navigation bar
    * FontAwesome icons
- Improved responsiveness (also scales font sizes, collapsing navigation)
- Some JS/JQuery sugar (smooth scrolling)
- "Download source" link at the base of each article

## Third-party assets

The theme uses external softwares, scripts, libraries and artworks:

* [Bootstrap](http://getbootstrap.com/) 3.x.x (CSS and JS)
* [Open Sans Font](http://www.google.com/fonts/specimen/Open+Sans)
* [JQuery](https://jquery.com/)
* [JQuery smooth scroll](https://github.com/kswedberg/jquery-smooth-scroll)
* [Pygments](http://pygments.org/)
* [FontAwesome](https://fortawesome.github.io/Font-Awesome/)

## Settings you should use

In order to make the one-page layout work, you should be using the following additional settings in your ``pelicanconf.py`` file:

```python
OUTPUT_SOURCES = True
OUTPUT_SOURCES_EXTENSION = ".source"

DIRECT_TEMPLATES = ['index', 'tags', 'categories', 'archives', 'home']
INDEX_SAVE_AS = "blog/index.html"
HOME_SAVE_AS = "index.html"
PAGE_URL = 'index.html#{slug}'
MENUITEMS = [('Home', '/index.html#'), ('Blog', '/blog/')]
```
