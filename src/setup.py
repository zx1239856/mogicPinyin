from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
        Extension("ime",["ime.pyx"]),
        Extension("builder",["builder.pyx"]),
        Extension("polyphone",["polyphone.pyx"]),
]

setup(
        name = "mogicPinyin",
        version = "1.0",
        description = "Pinyin input method based on 2-gram and 3-gram models",
        author = "zx1239856",
        author_email = "zx1239856@gmail.com",
        ext_modules = cythonize(ext_modules, build_dir="build"),
)
