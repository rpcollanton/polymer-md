from setuptools import setup, find_packages


setup(
   name = 'polymerMD',
   packages = find_packages(),
   install_requires = ['numpy','matplotlib','hoomd'],
   version = '0.0.1'
)