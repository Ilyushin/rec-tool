# pylint: disable=line-too-long, invalid-name, missing-docstring

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cf_experiments_loop",
    version="1.0.0",
    author="Eugene Ilyushin",
    author_email="eugene.ilyushin@gmail.com",
    description="The package helps to perform experiments with collaborative filtering models. It contains accessible datasets, metrics, and, moreover can save results in a database for future analysis.",
    long_description="The package helps to perform experiments with collaborative filtering models. It contains accessible datasets, metrics, and, moreover can save results in a database for future analysis.",
    long_description_content_type="text/markdown",
    url="https://github.com/Ilyushin/cf-experiments-loop",
    packages=setuptools.find_packages(),
    package_dir={
        'cf_experiments_loop': 'src',
    },
    entry_points={
        'console_scripts': [
            'cf_experiments_loop=src.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow',
        'pylint',
        'signal-transformation'
    ],
)
