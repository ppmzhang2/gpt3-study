
# -*- coding: utf-8 -*-
from setuptools import setup

import codecs

with codecs.open('README.md', encoding="utf-8") as fp:
    long_description = fp.read()
INSTALL_REQUIRES = [
    'openai>=0.16.0',
    'transformers>=4.17.0',
    'torch>=1.11.0',
]
EXTRAS_REQUIRE = {
    'ipy': [
        'jupyter>=1.0.0',
    ],
}
ENTRY_POINTS = {
    'console_scripts': [
        'study = study.cli:cli',
    ],
}

setup_kwargs = {
    'name': 'gpt3-study',
    'version': '0',
    'description': 'notes of GPT-3',
    'long_description': long_description,
    'license': 'MIT',
    'author': '',
    'author_email': 'ZHANG Meng <ztz2000@gmail.com>',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/ppmzhang2/gpt3-study',
    'packages': [
        'study.models',
        'study.serv',
        'study.datasets',
        'study',
    ],
    'package_dir': {'': 'src'},
    'package_data': {'': ['*']},
    'long_description_content_type': 'text/markdown',
    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Build Tools',
    ],
    'install_requires': INSTALL_REQUIRES,
    'extras_require': EXTRAS_REQUIRE,
    'python_requires': '>=3.9,<3.11',
    'entry_points': ENTRY_POINTS,

}


setup(**setup_kwargs)
