# Configuration file for the Sphinx documentation builder.

project = 'MLA-SoSe-26'
author = 'Niklas (MLA Team)'
release = '0.1.0'

extensions = []

templates_path = ['_templates']
# project/ wird komplett per include in project.rst gezogen -- die Teildateien duerfen
# nicht zusaetzlich als eigene Seiten gebaut werden, sonst gibt es jede Ueberschrift
# doppelt und Sphinx meckert, dass sie in keinem toctree stehen.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'project/**']

language = 'de'

html_theme = 'sphinx_rtd_theme'
# Die Projektseite ist eine einzige lange Seite -- ohne das zeigt die Sidebar nur
# die Hauptteile und klappt beim Navigieren alles andere zu.
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['toc-collapse.js']
