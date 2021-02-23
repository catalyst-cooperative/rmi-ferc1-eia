"""Setup script to make rmi_ferc1_eia directly installable with pip."""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text()

setup(
    name='pudl_rmi',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/catalyst-cooperative/rmi-ferc1-eia/',
    license="MIT",
    version='0.0.1',
    install_requires=[
        "catalystcoop.pudl>0.3.2",
        "fuzzywuzzy~=0.18.0",
        "recordlinkage~=0.14.0",
        "python-levenshtein~=0.12.2",
    ],
    python_requires=">=3.8,<3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    author="Catalyst Cooperative",
    author_email="pudl@catalyst.coop",
    maintainer="Christina Gosnell",
    maintainer_email="cgosnell@catalyst.coop",
    keywords=['depreciation', 'ferc1', 'eia', 'rmi']
)
