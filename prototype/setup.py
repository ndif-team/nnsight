from setuptools import find_packages, setup

setup(
    name='engine',
    version='1.0',
    author='...',
    description='...',
    long_description='...',
    url='https://github.com/JadenFiotto-Kaufman/ndif',
    python_requires='>=3.7',
    packages=find_packages(include=['engine', 'engine.*']),
    install_requires=[
        "baukit @ git+https://github.com/davidbau/baukit.git@e14a18a6ad6cf9e0d6a5dc7a97e671e393c01682",
        "torch",
        "torchvision",
        "transformers"
    ])