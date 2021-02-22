"""Setup script to make rmi_ferc1_eia directly installable with pip."""

from setuptools import setup, find_packages
from pathlib import Path

install_requires = [
    "addfips~=0.3.0",
    "catalystcoop.dbfread~=3.0",
    "coloredlogs~=15.0",
    "contextily~=1.0",
    "datapackage~=1.11",
    "geopandas~=0.8.1",
    "goodtables-pandas-py~=0.2.0",
    "google-cloud-storage~=1.35.0",
    "matplotlib~=3.0",
    "networkx~=2.2",
    "numpy~=1.19",
    "pandas~=1.2",
    "prefect[viz, gcp]~=0.14.2",
    "pyarrow~=2.0",
    "pyyaml~=5.0",
    "scikit-learn~=0.24",
    "scipy~=1.6",
    "seaborn~=0.11.1",
    "sqlalchemy~=1.3",
    "tableschema~=1.12",
    "tableschema-sql~=1.3",
    "timezonefinder~=5.0",
    "tqdm~=4.0",
    "xlsxwriter~=1.3",
    "recordlinkage",
    "fuzzywuzzy",
]

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text()

setup(
    name='rmi_pudl',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/catalyst-cooperative/rmi-ferc1-eia/',
    license="MIT",
    version='0.0.1',
    install_requires=install_requires,
    python_requires=">=3.8",
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
