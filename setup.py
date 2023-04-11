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
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    url="https://github.com/catalyst-cooperative/rmi-ferc1-eia/",
    license="MIT",
    version="0.1.0",
    install_requires=[
        "catalystcoop-pudl @ git+https://github.com/catalyst-cooperative/pudl@dev",
        "fuzzywuzzy>=0.18,<0.19",
        "python-levenshtein>=0.12,<0.13",
        "recordlinkage>=0.14,<0.16",
    ],
    extras_require={
        "test": [
            "black>=22.0,<23.4",
            "coverage>=5.3,<7.3",
            "flake8>=4.0,<6.1",
            "flake8-builtins>=1.5,<2.2",
            "flake8-colors>=0.1,<0.2",
            "flake8-docstrings>=1.5,<1.8",
            "flake8-rst-docstrings>=0.2,<0.4",
            "flake8-use-fstring>=1.0,<1.5",
            "isort>=5.0,<5.13",
            # "memory_profiler>=0.60.0",
            "pep8-naming>=0.12,<0.14",
            "pre-commit>=2.9,<3.3",
            "pytest>=6.2,<7.4",
            "pytest-cov>=2.10,<4.1",
            "rstcheck>=5.0,<6.2",
            "tox>=4.0,<4.5",
        ]
    },
    python_requires=">=3.11,<3.12",
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
