#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# load __version__, __author__, __email__, etc variables
exec(open('imsciutils/version_info.py').read())


with open('requirements.txt','r') as f:
    requirements = f.readlines()

long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup(name='imsciutils',
      version=__version__,
      description=__description__,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=__author__,
      author_email=__email__,
      license=__license__,
      url=__url__,
      download_url=__download_url__,
      maintainer=__maintainer__,
      maintainer_email=__maintainer_email__,
      keywords=__keywords__,
      python_requires=__python_requires__,
      platforms=__platforms__,
      classifiers=__classifiers__,
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements,
      )
