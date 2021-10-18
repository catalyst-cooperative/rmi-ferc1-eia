"""Connecting depreciation data, FERC1 and EIA for RMI."""

import pkg_resources
from pathlib import Path

import pudl_rmi.make_plant_parts_eia
import pudl_rmi.deprish
import pudl_rmi.coordinate
import pudl_rmi.connect_ferc1_to_eia
import pudl_rmi.connect_deprish_to_ferc1
import pudl_rmi.connect_deprish_to_eia

repo_dir = Path(__file__).resolve().parent.parent.parent
inputs_dir = repo_dir / 'inputs'
outputs_dir = repo_dir / 'outputs'

FILE_PATH_TRAINING = inputs_dir / 'train_ferc1_to_eia.csv'
FILE_PATH_PLANT_PARTS_EIA = outputs_dir / 'master_unit_list.pkl.gz'
FILE_PATH_DEPRISH_RAW = inputs_dir / 'depreciation_rmi.xlsx'
FILE_PATH_DEPRISH = outputs_dir / 'depreciation.pkl.gz'
FILE_PATH_DEPRISH_EIA = outputs_dir / 'deprish_to_eia.pkl.gz'
FILE_PATH_FERC1_EIA = outputs_dir / 'ferc1_to_eia.pkl.gz'
FILE_PATH_DEPRISH_FERC1 = outputs_dir / 'deprish_ferc1.pkl.gz'


__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "cgosnell@catalyst.coop"
__version__ = pkg_resources.get_distribution('pudl_rmi').version
__docformat__ = "restructuredtext en"
__description__ = "This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level."
