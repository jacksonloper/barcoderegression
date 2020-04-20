from setuptools import setup

setup(
    name='barcoderegression',
    author='Jackson Loper',
    version='0.0.1',
    description='Tools for deconvolution and demixing',
    packages=['barcoderegression'],
    package_data={
        'barcoderegression': ['*.pkl']
    }
)
