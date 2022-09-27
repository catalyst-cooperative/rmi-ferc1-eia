"""Setup script to make rmi_ferc1_eia directly installable with pip."""

from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text()

setup(
    name="pudl_rmi",
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/catalyst-cooperative/rmi-ferc1-eia/",
    license="MIT",
    version="0.1.0",
    install_requires=[
        "catalystcoop.pudl",
        "fuzzywuzzy>=0.18,<0.19",
        "python-levenshtein>=0.12,<0.21",
        "recordlinkage>=0.14,<0.16",
    ],
    extras_require={
        "test": [
            "black>=22,<23",
            "coverage>=5.3,<7.0",
            "flake8>=4.0,<5",
            "flake8-builtins~=1.5",
            "flake8-colors~=0.1",
            "flake8-docstrings~=1.5",
            "flake8-rst-docstrings~=0.2",
            "flake8-use-fstring~=1.0",
            "isort>=5.0,<6",
            "memory_profiler>=0.60.0",
            "pep8-naming~=0.12",
            "pre-commit>=2.9,<3",
            "pytest>=6.2,<8.0",
            "pytest-cov>=2.10,<4",
            "tox>=3.20,<4",
        ]
    },
    python_requires=">=3.10,<3.11",
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
    keywords=["depreciation", "ferc1", "eia", "rmi"],
)
