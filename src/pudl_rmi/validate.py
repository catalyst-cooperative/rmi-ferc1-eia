"""A module for data validation and QA/QC of the RMI outputs."""

import logging
from typing import List

import numpy as np
import pandas as pd
import sqlalchemy as sa

logger = logging.getLogger(__name__)


RMI_TO_PUDL_COLS = {
    "parent_name": "utility_name_parent",
    "utility_name": "utility_name_ferc1",
    "respondent_id": "utility_id_ferc1",
    "year": "report_year",
    "FERC_class": "ferc_acct_name",
    "original_cost": "plant_balance_w_common",
    "accum_depr": "book_reserve_w_common",
    "net_plant_balance": "unaccrued_balance_w_common",
}

RMI_TO_PUDL_FERC_ACCT_NAME = {
    "other_fossil": "Other",
    "hydro": "Hydrauli1c",
    "regional_transmission_and_market_operation": "Transmission",
}


def download_and_clean_net_plant_balance(pudl_engine: sa.engine.Engine) -> pd.DataFrame:
    """
    Download the Utility Transition Hub's cleaned FERC1 Net Plant Balance Table.

    This function grabs the RMI Hub Team's cleaned version of the FERC1 Net
    Plant Balance table. This function converts column names to PUDL column
    names, standardizes FERC Account Names and merges in the ``utility_id_pudl``
    through the ``utilities_ferc1`` PUDL table.

    Note: In the fullness of time, this table should really come directly from
    PUDL or the FERC1 database.

    Args:
        pudl_engine: A connection engine for the PUDL DB.
    """
    utils_f1 = pd.read_sql("utilities_ferc1", pudl_engine)
    npb = pd.read_csv(
        "https://utilitytransitionhub.rmi.org/static/data_download/net_plant_balance.csv"
    )
    npb = (
        npb.convert_dtypes(convert_floating=False)
        .rename(columns=RMI_TO_PUDL_COLS)
        .replace({"ferc_acct_name": RMI_TO_PUDL_FERC_ACCT_NAME})
        .merge(  # we need the pudl id!!
            utils_f1[["utility_id_ferc1", "utility_id_pudl"]],
            on=["utility_id_ferc1"],
            how="left",
            validate="m:1",
        )
    )
    npb.loc[:, "ferc_acct_name"] = npb.ferc_acct_name.str.title()
    return npb


def add_data_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a data_source column with 'FERC' and 'PUC'.

    The depreciation data includes a `data_source` column as a part of the
    primary key of that data set. This is because there are sometimes we have
    depreciation study from both `FERC` and `PUC` for the same utility and
    plants. Because of this, in order to compare the FERC plant input tables to
    the FERC-EIA-Depreciation tables we need to add *both* versions of the
    depreciation `data_source`.
    """
    if "data_source" not in df:
        df = pd.concat(
            [
                df.assign(data_source="PUC"),
                df.assign(data_source="FERC"),
            ]
        )
    return df


def test_df_vs_net_plant_balance(
    ferc_deprish_eia: pd.DataFrame,
    pudl_engine: sa.engine.Engine,
    data_cols: List = [
        "plant_balance_w_common",
        "book_reserve_w_common",
        "unaccrued_balance_w_common",
    ],
    rtol: float = 5e-02,
    atol: float = 5e-02,
) -> pd.DataFrame:
    """
    Compare FERC-Deprish-EIA output to FERC1 Net Plant Balance totals.

    The FERC1 Net Plant Balance table includes utility-FERC account level totals
    of several key financial data points. This function compares the utility-FERC
    account level ``data_cols`` from the Net Plant Balance table and the
    FERC-Deprish-EIA outputs. To do this, we aggregate the more granular
    FERC-Deprish-EIA output to the utility-FERC account level. Then we compare
    the ``data_cols`` from each of  the two inputs using ``np.isclose``. We also
    calculate a ratio of the FERC-Deprish-EIA column divided by the Net Plant
    Balance column.

    Question: Originally I passed in npb instead of generating it within this
    function and just had a little wrapper function but that also felt bad. Any
    Suggestions here?

    Args:
        ferc_deprish_eia: table of FERC-Deprish-EIA output.
        pudl_engine: A connection engine for the PUDL DB.
        data_cols: list of columns to compare.
        rtol: The relative tolerance parameter from ``np.isclose``. Default is 5e-02.
        atol: The absolute tolerance parameter from ``np.isclose``. Default is 5e-02.
    """
    pk_utils_acct = ["report_year", "data_source", "utility_id_pudl", "ferc_acct_name"]
    npb = download_and_clean_net_plant_balance(pudl_engine)
    test = pd.merge(
        (ferc_deprish_eia.groupby(pk_utils_acct)[data_cols].sum(min_count=1)),
        add_data_source(npb).set_index(pk_utils_acct)[data_cols],
        suffixes=("_fde", "_npb"),
        left_index=True,
        right_index=True,
        how="left",
    ).astype("float64")

    for data_col in data_cols:
        test.loc[:, f"{data_col}_isclose"] = np.isclose(
            test[f"{data_col}_fde"],
            test[f"{data_col}_npb"],
            equal_nan=True,
            rtol=5e-02,
            atol=5e-02,
        )
        test.loc[:, f"{data_col}_ratio"] = (
            test[f"{data_col}_fde"] / test[f"{data_col}_npb"]
        )
        logger.info(
            f"{data_col} close records: {len(test[test[f'{data_col}_isclose']])/len(test):.02%}"
        )
    return test
