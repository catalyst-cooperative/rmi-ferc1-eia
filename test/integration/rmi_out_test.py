"""
Test whether all of the FERC1/EIA/Depreciation outputs can be generated.

This can take up to an hour to run.
"""

import logging
from typing import List, Literal

import pandas as pd
import pytest
import sqlalchemy as sa

import pudl_rmi

logger = logging.getLogger(__name__)


PK_UTILS = ["report_year", "data_source", "utility_id_pudl"]
PK_PLANTS = ["report_year", "data_source", "utility_id_pudl", "plant_id_eia"]
EXPECTED_ERRORS_DIR = pudl_rmi.INPUTS_DIR / "expected_errors"


@pytest.mark.parametrize(
    "table_name",
    [
        "fuel_ferc1",
        "ownership_eia860",
        "plants_entity_eia",
        "fuel_receipts_costs_eia923",
        "utilities_pudl",
    ],
)
def test_pudl_engine(pudl_engine, table_name):
    """Test that the PUDL DB is actually available."""
    insp = sa.inspect(pudl_engine)
    assert table_name in insp.get_table_names()


def test_ppl_out(rmi_out, request):
    """Test generation of the EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-plant-parts-eia")
    ppl = rmi_out.plant_parts_eia(clobber=clobber)
    assert not ppl.empty
    for ppl_df in ["plant_parts_eia", "gens_mega_eia"]:
        if ppl_df in rmi_out.pudl_out._dfs:
            del rmi_out.pudl_out._dfs[ppl_df]


def test_deprish_out(rmi_out, request):
    """Test compilation of depreciation data."""
    clobber = not request.config.getoption("--cached-deprish")
    deprish = rmi_out.deprish(clobber=clobber)
    assert not deprish.empty


def test_deprish_to_eia_out(rmi_out, request):
    """Test fuzzy matching of depreciation data to EIA Plant Parts List."""
    clobber = not request.config.getoption("--cached-deprish-eia")
    deprish_to_eia = rmi_out.deprish_to_eia(clobber=clobber)
    assert not deprish_to_eia.empty


def test_ferc1_to_eia(rmi_out, request):
    """Test linkage of FERC 1 data to EIA PPL using record linkage."""
    rmi_out.pudl_out._dfs.clear()
    clobber = not request.config.getoption("--cached-ferc1-eia")
    ferc1_to_eia = rmi_out.ferc1_to_eia(clobber=clobber)
    assert not ferc1_to_eia.empty


def test_deprish_to_ferc1(rmi_out):
    """Test linkage of Depriciation data to FERC 1 data."""
    deprish_to_ferc1 = rmi_out.deprish_to_ferc1(clobber=True)
    assert not deprish_to_ferc1.empty


##################
# Validation Tests
##################


@pytest.mark.parametrize(
    "df1_name,df2_name,data_cols,by_name",
    [
        (
            "deprish",
            "deprish_to_eia",
            ["plant_balance_w_common"],
            "utilities",
        ),
        (
            "deprish",
            "deprish_to_eia",
            ["plant_balance_w_common"],
            "plants",
        ),
        (
            "deprish_to_eia",
            "deprish_to_ferc1",
            ["plant_balance_w_common"],
            "utilities",
        ),
        (
            "deprish_to_eia",
            "deprish_to_ferc1",
            ["plant_balance_w_common"],
            "plants",
        ),
        (
            "deprish",
            "deprish_to_ferc1",
            ["plant_balance_w_common"],
            "utilities",
        ),
        (
            "deprish",
            "deprish_to_ferc1",
            ["plant_balance_w_common"],
            "plants",
        ),
        (
            "ferc1_to_eia",
            "deprish_to_ferc1",
            ["capex_total"],
            "utilities",
        ),
        (
            "ferc1_to_eia",
            "deprish_to_ferc1",
            ["capex_total"],
            "plants",
        ),
    ],
)
def test_consistency_of_data_stages(
    rmi_out: pudl_rmi.coordinate.Output,
    df1_name: str,
    df2_name: str,
    data_cols: List[str],
    by_name: Literal["plants", "utilities"],
):
    """
    Test the consistency of a data column at two stages.

    The data that is processed in this repo goes along multiple stages of its
    journey. There are some quantities that are expected to be invariant when
    aggregated at the plant level, and the utility level -- that quantity might
    slosh around between generators in a plant, or between plants within a
    utility, but that when aggregated to the plant or utility level, it should
    be the same at every step in the processing.

    This test enables iteration over any two stages with a specific ``data_col``
    to test by aggregating both stages' df by either `plants` or `utilities` and
    checking if those plant or utility sums are exactly the same in each of the
    two stages.

    There are a relatively small set of known cases of inconsistencies between
    these stages. We have saved these known inconsistencies and hope to decrease
    them over time. When a code change causes either more errors or unexpected
    fixes to the consistency of these stages this test will fail.

    Nulls and connections... There are a lot of depreciation and FERC1 records
    that are not connected to the EIA plant-parts. These un-connected records
    are often not able to be propagated through the rest of the processing
    stages. Because of this, we often have more data in early stages than we do
    in later stages. This test is only testing the records which have non-null
    values in the aggregated ``data_col`` for both stages. We are avoiding the
    nulls because there are many of them and they would drown out the truly
    wrong/inconsistent data. (In truth, it is probable that many of the known
    inconsistencies are also caused by missing connections to EIA.)

    Note: this will not be effective on any of the rate columns like
    `depreciation_annual_rate` or `avg_num_employees`. The columns which this
    would work on are roughly synonymous with the columns which recieve a
    `scale` treatement type in :mod:``pudl_rmi.connect_deprish_to_ferc1.py``
    """
    if by_name == "plants":
        by = PK_PLANTS
    elif by_name == "utilities":
        by = PK_UTILS
    else:
        raise AssertionError(
            "This test only takes `plants` or `utilities` as an argument for `by_name`"
        )

    agg_test = pudl_rmi.validate.agg_test_data(
        df1=rmi_out.__getattribute__(df1_name)(),
        df2=rmi_out.__getattribute__(df2_name)(),
        data_cols=data_cols,
        by=by,
    )
    for data_col in data_cols:
        # there are a ton of unconnected data in these tables (eg. depreciation
        # records that don't have a EIA connection) these records will not get
        # effectively propegated through each of these stages that require a match
        # to EIA. the `data_ratio` column being null is an indication that either df1
        # or df2 has a null data_col.
        actual_aggregation_errors = agg_test[
            ~agg_test[f"{data_col}_isclose"] & agg_test[f"{data_col}_ratio"].notnull()
        ]

        logger.info(
            f"Failures for {data_col} by {by_name} btwn {df1_name} "
            f"and {df2_name}: {len(actual_aggregation_errors)}"
        )
        expected_errors_path = (
            EXPECTED_ERRORS_DIR
            / f"expected_errors_{data_col}_{by_name}_{df1_name}_vs_{df2_name}.csv"
        )
        expected_aggregation_errors = pd.read_csv(expected_errors_path).set_index(by)
        # the commented out lines here are here to help
        # try:
        pd.testing.assert_index_equal(
            actual_aggregation_errors.index,
            expected_aggregation_errors.index,
            exact="equiv",
            check_order=False,
        )
        # except AssertionError:
        #     actual_aggregation_errors.reset_index()[by].to_csv(
        #         expected_errors_path, index=False
        #     )
