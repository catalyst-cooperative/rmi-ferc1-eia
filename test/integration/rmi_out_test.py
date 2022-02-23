"""
Test whether all of the FERC1/EIA/Depreciation outputs can be generated.

This can take up to an hour to run.
"""

import logging

import pytest
import sqlalchemy as sa

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "table_name", [
        "fuel_ferc1",
        "ownership_eia860",
        "plants_entity_eia",
        "fuel_receipts_costs_eia923",
        "utilities_pudl",
    ]
)
def test_pudl_engine(pudl_engine, table_name):
    """Test that the PUDL DB is actually available."""
    insp = sa.inspect(pudl_engine)
    assert table_name in insp.get_table_names()


def test_ppl_out(rmi_out, request):
    """Test generation of the EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-plant-parts-eia")
    ppl = rmi_out.grab_plant_part_list(clobber=clobber)
    assert not ppl.empty
    del ppl
    for ppl_df in ["plant_parts_eia", "gens_mega_eia", "true_grans_eia"]:
        if ppl_df in rmi_out.pudl_out._dfs:
            del rmi_out.pudl_out._dfs[ppl_df]


def test_deprish_out(rmi_out, request):
    """Test compilation of depreciation data."""
    clobber = not request.config.getoption("--cached-deprish")
    deprish = rmi_out.grab_deprish(clobber=clobber)
    assert not deprish.empty
    del deprish


def test_deprish_to_eia_out(rmi_out, request):
    """Test fuzzy matching of depreciation data to EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-deprish-eia")
    deprish_to_eia = rmi_out.grab_deprish_to_eia(clobber=clobber)
    assert not deprish_to_eia.empty
    del deprish_to_eia


def test_ferc1_to_eia(rmi_out, request):
    """Test linkage of FERC 1 data to EIA PPL using record linkage."""
    clobber = not request.config.getoption("--cached-ferc1-eia")
    ferc1_to_eia = rmi_out.grab_ferc1_to_eia(clobber=clobber)
    assert not ferc1_to_eia.empty
    del ferc1_to_eia


def test_deprish_to_ferc1(rmi_out):
    """Test linkage of Depriciation data to FERC 1 data."""
    deprish_to_ferc1 = rmi_out.grab_deprish_to_ferc1(clobber=True)
    assert not deprish_to_ferc1.empty
    del deprish_to_ferc1
