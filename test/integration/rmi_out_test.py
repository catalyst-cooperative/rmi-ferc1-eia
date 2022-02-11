"""
Test whether all of the FERC1/EIA/Depreciation outputs can be generated.

This can take up to an hour to run.
"""

import logging

logger = logging.getLogger(__name__)


def test_ppl_out(rmi_out, request):
    """Test generation of the EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-plant-parts-eia")
    ppl = rmi_out.grab_plant_part_list(clobber=clobber)
    assert not ppl.empty


def test_deprish_out(rmi_out, request):
    """Test compilation of depreciation data."""
    clobber = not request.config.getoption("--cached-deprish")
    deprish = rmi_out.grab_deprish(clobber=clobber)
    assert not deprish.empty


def test_deprish_to_eia_out(rmi_out, request):
    """Test fuzzy matching of depreciation data to EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-deprish-eia")
    deprish_to_eia = rmi_out.grab_deprish_to_eia(clobber=clobber)
    assert not deprish_to_eia.empty


def test_ferc1_to_eia(rmi_out, request):
    """Test linkage of FERC 1 data to EIA PPL using record linkage."""
    clobber = not request.config.getoption("--cached-ferc1-eia")
    ferc1_to_eia = rmi_out.grab_ferc1_to_eia(clobber=clobber)
    assert not ferc1_to_eia.empty


def test_deprish_to_ferc1(rmi_out):
    """Test linkage of Depriciation data to FERC 1 data."""
    deprish_to_ferc1 = rmi_out.grab_deprish_to_ferc1(clobber=True)
    assert not deprish_to_ferc1.empty