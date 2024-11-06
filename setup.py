from setuptools import setup, find_packages

setup(
   name='wcd_project_template',
   version='1.0',
   description='A Computer Vision Project Template for WeCloudData',
   author='Jaskirat Singh Bhatia',
   author_email='jaskirat.bhatia.work@gmail.com',
   packages=find_packages(),
   install_requires=['albumentations'], #external packages as dependencies
)