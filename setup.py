import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    README = fh.read()

import lingd

VERSION = lingd.__version__

setuptools.setup(
    name='lingd',
    version=VERSION,
    description='LiNG discovery algorithm',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scikit-learn',
        'lingam',
    ],
    url='https://github.com/cdt15/lingd',
    packages=setuptools.find_packages(exclude=['tests', 'examples']),
    package_data={
        'lingd': ['*.r'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
