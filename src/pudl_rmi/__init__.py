"""Connecting depreciation data, FERC1 and EIA for RMI."""

import os
from pathlib import Path

import pkg_resources

import pudl_rmi.connect_deprish_to_eia
import pudl_rmi.connect_deprish_to_ferc1
import pudl_rmi.connect_ferc1_to_eia
import pudl_rmi.coordinate
import pudl_rmi.deprish
import pudl_rmi.formatter_optimus
import pudl_rmi.make_plant_parts_eia
import pudl_rmi.validate  # noqa: F401

REPO_DIR = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = Path(os.environ.get("PUDL_RMI_INPUTS_DIR", REPO_DIR / "inputs"))
"""
Directory of input files that are used in generating the RMI outputs.

These files are required inputs and should be checked into the repo. Some of
these files (even the excel files) are updated during some of the processing.
If these files don't exist, then many of the outputs will fail.
"""
OUTPUTS_DIR = Path(os.environ.get("PUDL_RMI_OUTPUTS_DIR", REPO_DIR / "outputs"))
"""
Directory of output files that are generated from the RMI processes.

Nothing needs to be checked into this directory for any of the RMI processes
to run. If using ``pudl_rmi.coordinate.Output()`` these files will be generated
and stored as pickled dataframes. If these files do exist,
``pudl_rmi.coordinate.Output()`` will either grab them or clobber them.
"""

TRAIN_FERC1_EIA_CSV: Path = INPUTS_DIR / "train_ferc1_eia.csv"
"""Path to training data for FERC1 plants and EIA plant-part list."""
DEPRISH_RAW_XLSX: Path = INPUTS_DIR / "deprish_raw.xlsx"
"""Path to the raw depreciation data."""
DEPRISH_COMMON_LABELS_XLSX: Path = INPUTS_DIR / "deprish_common_labels.xlsx"
"""Path to mannual label of common records in depreciation studies."""
FERC_ACCT_NAMES_CSV: Path = INPUTS_DIR / "ferc_acct_names.csv"
NULL_FERC1_EIA_CSV: Path = INPUTS_DIR / "null_ferc1_eia.csv"
"""Path to list of record_id_ferc1 values with no EIA match."""

PLANT_PARTS_EIA_PKL: Path = OUTPUTS_DIR / "plant_parts_eia.pkl.gz"
"""Path to EIA plant-part list."""
DISTINCT_PLANT_PARTS_EIA_PKL: Path = OUTPUTS_DIR / "plant_parts_eia_distinct.pkl.gz"
"""Path to EIA plant-parts list with only true granularity and ownership records."""
DEPRISH_PKL: Path = OUTPUTS_DIR / "deprish.pkl.gz"
"""Path to processed depreciation data."""
DEPRISH_EIA_PKL: Path = OUTPUTS_DIR / "deprish_eia.pkl.gz"
"""Path to connection between depreciation data and the EIA plant-part list."""
FERC1_EIA_PKL: Path = OUTPUTS_DIR / "ferc1_eia.pkl.gz"
"""Path to connection between FERC1 plants and the EIA plant-part list."""
DEPRISH_FERC1_PKL: Path = OUTPUTS_DIR / "deprish_ferc1.pkl.gz"
"""Path to connection between depreciation data and FERC1 plants."""


EQR_DATA_DIR: Path = INPUTS_DIR / "eqr_data"
"""Path to raw EQR zipfiles."""
EQR_DB_PATH: Path = OUTPUTS_DIR / "eqr.db"
"""Path to procesesed EQR sqlite database."""

__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "cgosnell@catalyst.coop"
__version__ = pkg_resources.get_distribution("pudl_rmi").version
__docformat__ = "restructuredtext en"
__description__ = "This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level."
