"""Connecting depreciation data, FERC1 and EIA for RMI."""

import pkg_resources

import pudl_rmi.connect_deprish_to_eia
import pudl_rmi.connect_deprish_to_ferc1
import pudl_rmi.connect_ferc1_to_eia
import pudl_rmi.coordinate
import pudl_rmi.deprish
import pudl_rmi.make_plant_parts_eia

__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "cgosnell@catalyst.coop"
__version__ = pkg_resources.get_distribution('pudl_rmi').version
__docformat__ = "restructuredtext en"
__description__ = "This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level."
