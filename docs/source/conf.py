# 文件作用：配置 Sphinx 文档生成，并让 .rst 模板可以读取 PlanetAlign 的导出内容。
# 核心依赖：setup() 注册 source-read 事件，rst_jinja_render() 通过构建器的模板引擎渲染。
# 结果去向：供 docs/Makefile 的 Sphinx 构建使用，不属于训练运行流程。

# Configuration file for the Sphinx documentation builder.

import os
import sys
import datetime

# Add the parent directory (where PlanetAlign is) to sys.path
sys.path.insert(0, os.path.abspath("../.."))

import PlanetAlign

# -- Project information
project = 'PlanetAlign'
copyright = f'{datetime.datetime.now().year}, PLANETALIGN'
author = 'PLANETALIGN'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

autosummary_generate = True
add_module_names = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# typehints_use_rtype = True
# typehints_defaults = 'comma'

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

print("==== Debugging PlanetAlign ====")
print("PlanetAlign found:", PlanetAlign)
print("Available attributes:", dir(PlanetAlign))


# 输入来源：Sphinx 读取 .rst 时触发 source-read，传入应用对象和源码列表。
# 功能：将 PlanetAlign 放入模板上下文，渲染后写回 source[0]。
def rst_jinja_render(app, _, source):
    """Enable Jinja templating in .rst files."""
    if hasattr(app.builder, "templates"):
        rst_context = {"PlanetAlign": PlanetAlign}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


# 功能：将 rst_jinja_render() 注册为文档源码读取事件的回调。
def setup(app):
    """Setup Sphinx to process Jinja templates."""
    app.connect("source-read", rst_jinja_render)
