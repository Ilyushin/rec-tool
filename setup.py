# pylint: disable=line-too-long, invalid-name, missing-docstring

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rec-tool",
    version="1.0.0",
    author="Eugene Ilyushin",
    author_email="eugene.ilyushin@gmail.com",
    description="The package helps to perform experiments with collaborative filtering models. It contains accessible datasets, metrics, and, moreover can save results in a database for future analysis.",
    long_description="The package helps to perform experiments with collaborative filtering models. It contains accessible datasets, metrics, and, moreover can save results in a database for future analysis.",
    long_description_content_type="text/markdown",
    url="https://github.com/Ilyushin/cf-experiments-loop",
    packages=setuptools.find_packages(),
    package_dir={
        'rec-tool': 'rec-tool',
        'rec-tool.common': 'rec-tool/common',
        'rec-tool.models': 'rec-tool/models',
        'rec-tool.transformations': 'rec-tool/transformations',
        'rec-tool.ml_flow': 'rec-tool/ml_flow',
    },
    entry_points={
        'console_scripts': [
            'rec-tool=rec-tool.main:main',
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
        'signal-transformation',
        'pyyaml',
        'pandas',
        'mlflow',
        'tqdm'
    ],
)
