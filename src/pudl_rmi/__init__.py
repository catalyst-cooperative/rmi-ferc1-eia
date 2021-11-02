"""Connecting depreciation data, FERC1 and EIA for RMI."""

import pkg_resources
from pathlib import Path

import pudl_rmi.make_plant_parts_eia
import pudl_rmi.deprish
import pudl_rmi.coordinate
import pudl_rmi.connect_ferc1_to_eia
import pudl_rmi.connect_deprish_to_ferc1
import pudl_rmi.connect_deprish_to_eia

REPO_DIR = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_DIR / 'inputs'
"""
Directory of input files that are used in generating the RMI outputs.

These files are required inputs and should be checked into the repo. Some of
these files (even the excel files) are updated during some of the processing.
If these files don't exist, then many of the outputs will fail.
"""
OUTPUTS_DIR = REPO_DIR / 'outputs'
"""
Directory of output files that are generated from the RMI processes.

Nothing needs to be checked into this directory for any of the RMI processes
to run. If using ``pudl_rmi.coordinate.Output()`` these files will be generated
and stored as pickled dataframes. If these files do exist,
``pudl_rmi.coordinate.Output()`` will either grab them or clobber them.
"""

FILE_PATH_TRAIN_FERC_EIA = INPUTS_DIR / 'train_ferc1_eia.csv'
FILE_PATH_DEPRISH_RAW = INPUTS_DIR / 'deprish_raw.xlsx'
FILE_PATH_DEPRISH_COMMON_LABELS = INPUTS_DIR / 'deprish_common_labels.xlsx'

FILE_PATH_PLANT_PARTS_EIA = OUTPUTS_DIR / 'plant_parts_eia.pkl.gz'
FILE_PATH_DEPRISH = OUTPUTS_DIR / 'deprish.pkl.gz'
FILE_PATH_DEPRISH_EIA = OUTPUTS_DIR / 'deprish_eia.pkl.gz'
FILE_PATH_FERC1_EIA = OUTPUTS_DIR / 'ferc1_eia.pkl.gz'
FILE_PATH_DEPRISH_FERC1 = OUTPUTS_DIR / 'deprish_ferc1.pkl.gz'


__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "cgosnell@catalyst.coop"
__version__ = pkg_resources.get_distribution('pudl_rmi').version
__docformat__ = "restructuredtext en"
__description__ = "This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level."
