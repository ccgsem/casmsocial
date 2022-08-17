#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'numpy>=1.19',
    'pandas>=1.2',
    'repast4py>=1.0.b1',
]

test_requirements = ['pytest>=3', ]

setup(
    author="Jon C. Cline",
    author_email='jon.c.cline@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="communitysim is a Python framework for implementing agent-based models that simulate the dynamics of a synthetic population",
    entry_points={
        'console_scripts': [
            'communitysimpy=communitysim.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='communitysim',
    name='communitysim',
    #packages=find_packages(include=['communitysim', 'communitysim.*']),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/clinejc/communitysimpy',
    version='0.2.0',
    zip_safe=False,
)
