#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='model-converters',
    version='1.2.1',
    description='Tools for converting Keras models for use with other ML frameworks.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[
        'bin/keras_to_tensorflow',
    ],
    install_requires=[
        'tensorflow >= 0.12.0',
        'Keras >= 2.2.0',
        'h5py'
    ]
)
