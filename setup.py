import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '0.1.0'
DESCRIPTION = 'a python package for openocr, which is used to help developers quickly deploy OCR algorithms implemented in the openocr framework.'

# Setting up
setup(
    author="OpenOCR",
    author_email="18300180089@fudan.edu.cn",
    name="openocr-python",
    url='https://github.com/Topdu/OpenOCR',
    version=VERSION,
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    project_urls={"Source Code": "https://github.com/Topdu/OpenOCR",},
    install_requires=['numpy','opencv-python<=4.6.0.66','pyclipper'],
    keywords=['python','OCR','STR','OpenOCR','openocr'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ]
)

