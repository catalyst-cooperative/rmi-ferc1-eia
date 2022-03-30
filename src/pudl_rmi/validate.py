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
    "hydro": "Hydraulic",
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
        .assign(report_date=lambda x: pd.to_datetime(x.report_year, format="%Y"))
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


def group_sum_cols(df, data_cols: List[str], by: List[str]) -> pd.DataFrame:
    """Groupby sum a specific table's data cols."""
    # convert date to year bc many of the og depish studies are EOY
    summed_out = (
        df.assign(report_year=lambda x: x.report_date.dt.year)
        .astype({"report_year": pd.Int64Dtype()})
        .groupby(by=by, dropna=True)[data_cols]
        .sum(min_count=1)
    )
    return summed_out


def agg_test_data(
    df1: pd.DataFrame, df2: pd.DataFrame, data_cols: List[str], by: List[str], **kwarg
) -> pd.DataFrame:
    """
    Merge two grouped input tables to determine if summed data column are equal.

    Args:
        df1: One dataframe to sum and check consistency with ``df2``.
        df2: Other dataframe to sum and check consistency against ``df1``.
        data_cols: data columns to check. Columns must be in both ``df1`` and
            ``df2``.
    """
    test = pd.merge(
        group_sum_cols(df1.pipe(add_data_source), data_cols=data_cols, by=by),
        group_sum_cols(df2.pipe(add_data_source), data_cols=data_cols, by=by),
        right_index=True,
        left_index=True,
        suffixes=("_1", "_2"),
        how="outer",
    ).astype("float64")
    for data_col in data_cols:
        test.loc[:, f"{data_col}_isclose"] = np.isclose(
            test[f"{data_col}_1"],
            test[f"{data_col}_2"],
            equal_nan=True,
            **kwarg,
        )
        test.loc[:, f"{data_col}_ratio"] = test[f"{data_col}_1"] / test[f"{data_col}_2"]
        logger.info(
            f"{data_col} close records: {len(test[test[f'{data_col}_isclose']])/len(test):.02%}"
        )
    return test.sort_index()


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
    test = agg_test_data(
        df1=ferc_deprish_eia,
        df2=npb,
        data_cols=data_cols,
        by=pk_utils_acct,
        rtol=rtol,
        atol=atol,
    )

    return test
