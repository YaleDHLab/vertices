from setuptools import setup

setup (
  name='vertices',
  version='0.0.1',
  packages=['vertices'],
  keywords = ['geometry', '2d', '3d', 'mesh', 'obj', 'wavefront', 'd3', 'webgl'],
  description='Convert an obj with n vertices into one with p vertices',
  url='https://github.com/yaledhlab/vertices',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'matplotlib>=2.0.0',
    'numpy>=1.16.4',
    'PyWavefront>=1.0.5',
    'scikit-image>=0.15.0',
    'scikit-learn>=0.21.3',
    'scipy==1.1.0',
  ],
)