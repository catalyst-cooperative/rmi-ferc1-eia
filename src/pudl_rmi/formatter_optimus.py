"""Convert RMI outputs into model output format."""

import logging
from typing import Dict, List, Literal

import pandas as pd
import pudl
import sqlalchemy as sa

from pudl_rmi import validate
from pudl_rmi.connect_deprish_to_ferc1 import allocate_cols

logger = logging.getLogger(__name__)

RENAME_COLS: Dict = {
    "record_id_eia": "Unique_ID",
    "faked_1": "Scenario_Short",
    "utility_name_eia": "Utility",
    "faked_2": "Scenario",
    "operational_status": "Asset Status",
    "plant_name_eia": "Plants",
    "generator_id": "Unit",
    "resource_type": "Resource Type",
    "state": "State (Physical Location)",  # will be merged
    "fraction_owned": "Ownership Percentage (%)",
    "capacity_mw": "Net Capacity (MW)",
    "capacity_factor": "Capacity Factor (%)",
    "net_generation_mwh": "Net Generation (MWh)",
    "faked_3": "CO2 Emission (tons)",
    "total_fuel_cost": "Fuel Cost ($)",
    "faked_4": "Non-Fuel Variable O&M Costs ($)",
    "faked_5": "Fixed O&M Costs ($)",
    "opex_total_nonfuel": "Total O&M Cost",
    "faked_6": "Total Production Costs ($)",
    "installation_year": "Commission Year",
    "remaining_life_avg": "Current Remaining Accounting Life (Yrs)",
    "plant_balance_w_common": "Gross Plant Balance/Original Cost ($)",
    "book_reserve_w_common": "Book Reserve/Accumulated Depreciation ($)",
    "net_plant_balance_w_common": "Current Net Plant Balance ($)",
    "depreciation_annual_epxns_w_common": "Annual Depreciation Expense ($)",
    "depreciation_annual_rate": "Depreciation Rate (%)",
    "net_salvage_w_common": "Decommissioning Cost ($)",
    "net_salvage_rate": "Decommissioning Rate (%)",
    "capex_annual_addition": "Annual Capital Additions ($)",
    "faked_7": "Accumulated Deferred Income Tax (ADIT) ($)",
    "faked_8": "Accumulated Deferred Income Tax (ASC740) ($)",
    "faked_9": "Protected Excess Deferred Income Tax (EDIT) ($)",
    "faked_10": "Does this plant have an Flue-gas desulfurization (FGD) device? (Y/N)",
    "faked_11": "Ownership",
    "faked_12": "Accounting Retirement Year",
    "faked_13": "Maintenance CAPEX_2020",
    "faked_14": "Maintenance CAPEX_2021",
    "faked_15": "Maintenance CAPEX_2022",
    "faked_16": "Maintenance CAPEX_2023",
    "faked_17": "Maintenance CAPEX_2024",
    "faked_18": "Maintenance CAPEX_2025",
    "faked_19": "Maintenance CAPEX_2026",
    "faked_20": "Maintenance CAPEX_2027",
    "faked_21": "Maintenance CAPEX_2028",
    "faked_22": "Maintenance CAPEX_2029",
    "faked_23": "Maintenance CAPEX_2030",
    "faked_24": "Maintenance CAPEX_2031",
    "faked_25": "Maintenance CAPEX_2032",
    "faked_26": "Maintenance CAPEX_2033",
    "faked_27": "Maintenance CAPEX_2034",
    "faked_28": "Maintenance CAPEX_2035",
    "faked_29": "Maintenance CAPEX_2036",
    "faked_30": "Maintenance CAPEX_2037",
    "faked_31": "Maintenance CAPEX_2038",
    "faked_32": "Maintenance CAPEX_2039",
    "faked_33": "Maintenance CAPEX_2040",
    "faked_34": "Maintenance CAPEX_2041",
    "faked_35": "Maintenance CAPEX_2042",
    "faked_36": "Maintenance CAPEX_2043",
    "faked_37": "Maintenance CAPEX_2044",
    "faked_38": "Maintenance CAPEX_2045",
    "faked_39": "Maintenance CAPEX_2046",
    "faked_40": "Maintenance CAPEX_2047",
    "faked_41": "Maintenance CAPEX_2048",
    "faked_42": "Maintenance CAPEX_2049",
    "faked_43": "Maintenance CAPEX_2050",
    "faked_44": "CWIP_2020",
    "faked_45": "CWIP_2021",
    "faked_46": "CWIP_2022",
    "faked_47": "CWIP_2023",
    "faked_48": "CWIP_2024",
    "faked_49": "CWIP_2025",
    "faked_50": "CWIP_2026",
    "faked_51": "CWIP_2027",
    "faked_52": "CWIP_2028",
    "faked_53": "CWIP_2029",
    "faked_54": "CWIP_2030",
    "faked_55": "CWIP_2031",
    "faked_56": "CWIP_2032",
    "faked_57": "CWIP_2033",
    "faked_58": "CWIP_2034",
    "faked_59": "CWIP_2035",
    "faked_60": "CWIP_2036",
    "faked_61": "CWIP_2037",
    "faked_62": "CWIP_2038",
    "faked_63": "CWIP_2039",
    "faked_64": "CWIP_2040",
    "faked_65": "CWIP_2041",
    "faked_66": "CWIP_2042",
    "faked_67": "CWIP_2043",
    "faked_68": "CWIP_2044",
    "faked_69": "CWIP_2045",
    "faked_70": "CWIP_2046",
    "faked_71": "CWIP_2047",
    "faked_72": "CWIP_2048",
    "faked_73": "CWIP_2049",
    "faked_74": "CWIP_2050",
    "line_id": "Record ID Depreciation",
    "record_id_ferc1": "Record ID FERC 1",
    "plant_id_eia": "Plant ID EIA",
    "report_year": "Report Year",
    "data_source": "Data Source of Depreciation Study",
    "ferc_acct_name": "FERC account",
    "ferc_account_name_rmi": "technology",
    "unaccrued_balance_w_common": "Unaccrued Balance",
}

TECHNOLOGY_DESCRIPTION_TO_RESOURCE_TYPE: Dict = {
    "Conventional Steam Coal": "Coal",
    "Natural Gas Fired Combined Cycle": "NaturalGasCC",
    "Natural Gas Fired Combustion Turbine": "NaturalGasCT",
    "Natural Gas Steam Turbine": "NaturalGasCT",  # MAYBE?!?
    "Geothermal": "Geothermal",
    "Onshore Wind Turbine": "LandbasedWind",
    "Conventional Hydroelectric": "Hydropower",
    # pd.NA: 'Transmission',
    # pd.NA: 'Distribution',
    "Solar Photovoltaic": "UtilityPV",
    "Nuclear": "Nuclear",
    "Offshore Wind Turbine": "OffshoreWind",  # THIS ONE IS A GUESS
    "Solar Thermal with Energy Storage": "SolarPlusBattery",
    # pd.NA: 'EE',
    # pd.NA: 'DR',
    # pd.NA: 'Battery'
}

UTILITY_RENAME: Dict = {
    "Carolina Power & Light Co": "Duke Energy Progress",
    "Duke Energy Corp": "Duke Energy Carolinas",
}
"""
EIA utility names (keys) to Optimus-preferred names (values).
"""


def execute(
    deprish_ferc1_eia: pd.DataFrame,
    plants_eia860: pd.DataFrame,
    utils_eia860: pd.DataFrame,
    balancing_account: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Convert RMI outputs into optimus model inputs.

    This function converts the Deprish-FERC-EIA output to the RMI Optimus format.
    This conversion mostly consists of renaming columns and content. It also
    adds two data columns from PUDL tables: ``state`` from the ``plants_entity_eia``
    table and ``utility_id_pudl`` from ``utilities_eia``.

    Args:
        deprish_ferc1_eia: table of depreciation & FERC 1 & EIA data.
        plants_eia860: ``pudl_out.plants_eia860()``
        utils_eia860: ``pudl_out.utils_eia860()``
        kwargs: Additional kwargs to be passed into :func:``select_from_deprish_ferc1``
            if you want to select specific utilities/years/data sources.

    """
    # if there are kwargs, this means you want to select from deprish_ferc1_eia
    if kwargs:
        deprish_ferc1_eia = select_from_deprish_ferc1(deprish_ferc1_eia, **kwargs)
    # add plant state and the EIA utility name
    if balancing_account:
        pudl_engine = sa.create_engine(pudl.workspace.setup.get_defaults()["pudl_db"])
        net_plant_balance = validate.download_and_clean_net_plant_balance(pudl_engine)
        deprish_ferc1_eia = add_balancing_account(
            deprish_ferc1_eia, net_plant_balance
        ).pipe(allocate_balancing_account_to_assets)
    model_input = (
        deprish_ferc1_eia.reset_index()  # reset index b4 merge bc it's prob 'record_id_eia'
        .merge(
            plants_eia860[["plant_id_eia", "report_date", "state"]].drop_duplicates(),
            on=["plant_id_eia", "report_date"],
            validate="m:1",
            how="left",
        )
        .merge(  # sorting by report date to get the most recent utility name
            # because they can change over time and rmi prefers the up-to-date names
            utils_eia860.sort_values(["report_date"], ascending=False)[
                ["utility_id_pudl", "utility_name_eia"]
            ].drop_duplicates(subset=["utility_id_pudl"]),
            on=["utility_id_pudl"],
            validate="m:1",
            how="left",
        )
        .replace({"utility_name_eia": UTILITY_RENAME})
        .dropna(subset=["record_id_eia"])  # this is the pk. it must be non-null
        .round({k: 1 for (k, v) in RENAME_COLS.items() if "($)" in v})
    )

    model_input.loc[:, "resource_type"] = model_input.technology_description.replace(
        TECHNOLOGY_DESCRIPTION_TO_RESOURCE_TYPE
    )
    model_input.loc[:, "operational_status"] = (
        model_input.operational_status.str.title()
    )

    # Create a bunch of "faked"  columns
    for faked_col in [x for x in RENAME_COLS.keys() if "faked_" in x]:
        model_input.loc[:, faked_col] = pd.NA

    model_input = model_input.rename(columns=RENAME_COLS)[list(RENAME_COLS.values())]

    return model_input


def select_from_deprish_ferc1(
    deprish_ferc1_eia: pd.DataFrame,
    util_id_pudls: List[int],
    years: List[int],
    priority_data_source: Literal["PUC", "FERC"] = "PUC",
    include_non_priority_data_source: bool = True,
) -> pd.DataFrame:
    """
    Select subset of Deprish-FERC1-EIA output.

    Usually, we want to look at a particular utility or utilities from a
    particular year or years from a particular depreciation data source.

    The depreciation data source arw special arguments here because sometimes
    the depreciation studies from one data source excludes plants or even whole
    technology types (ex: a PUC study does not include hydro plants). This
    function enables us to grab just one data source OR to prioritize a data
    source and get any plants that are missing from the priority data source
    from the other data source.

    Args:
        deprish_ferc1_eia: table of depreciation & FERC 1 & EIA data.
        util_id_pudls: list of desired ``utility_id_pudl``'s to select for.
        years: list of desired years to select for.
        priority_data_source: depreciation data source to prioritize.
        include_non_priority_data_source: If True, include records from
            depreciation data sources other than the ``priority_data_source``
            if there are plants that are not in the ``priority_data_source``.

    """
    util_all = deprish_ferc1_eia[
        (
            (deprish_ferc1_eia.report_year.isin(years))
            & (deprish_ferc1_eia.utility_id_pudl.isin(util_id_pudls))
        )
    ]

    util_source1 = util_all[util_all.data_source.isin([priority_data_source, pd.NA])]

    if include_non_priority_data_source:
        util_source2 = util_all[
            ~util_all.plant_id_eia.isin(util_source1.plant_id_eia.unique())
            & ~util_all.plant_id_eia.isnull()
        ]
        util_out = pd.concat([util_source1, util_source2])
    else:
        util_out = util_source1
    return util_out.sort_index()


def add_balancing_account(
    ferc_deprish_eia: pd.DataFrame, net_plant_balance: pd.DataFrame
) -> pd.DataFrame:
    """
    Add a balancing account record for every utility-ferc account.

    The FERC1 net plant balance table includes utility-ferc account level
    financial data points including: net plant balance (all the captial that has
    been pour into an asset), book reserve (asset value that has been
    depreciated), and unaccrued balance (asset value that has not been
    depreciated).

    This function generates new records for each utility-ferc account that
    balances the depreciation study values with the FERC1 net plant balance
    table. The assumption here is that the utility-level data should be more
    acurate than the asset-level data points.

    Implementation: this function takes an asset-level data table with
    depreciation data, sums it by the utility-ferc account, merges the FERC1
    data, calculates the difference for each of these data points, and finallly
    uses those differences to generates new records that get added into the
    original depreciation data.

    Args:
        ferc_deprish_eia: FERC1, depreciation and EIA data table. Result of
            :meth:`pudl_rmi.coordinate.Output.deprish_to_ferc1()`
        net_plant_balance: result of :func:`pudl_rmi.validate.download_and_clean_net_plant_balance`
    """
    compare_npb = validate.compare_df_vs_net_plant_balance(
        df=ferc_deprish_eia,
        net_plant_balance=net_plant_balance,
        data_cols=[
            "plant_balance_w_common",
            "book_reserve_w_common",
            "unaccrued_balance_w_common",
        ],
    ).sort_index()

    pk_utils_acct = ["report_year", "data_source", "utility_id_pudl", "ferc_acct_name"]
    compare_npb.loc[:, "depreciation_annual_epxns_w_common"] = ferc_deprish_eia.groupby(
        pk_utils_acct
    )["depreciation_annual_epxns_w_common"].mean()

    balancing_records = (
        compare_npb[
            [
                "plant_balance_w_common_diff",
                "book_reserve_w_common_diff",
                "unaccrued_balance_w_common_diff",
                "depreciation_annual_epxns_w_common",
            ]
        ]
        .dropna(how="all")
        .rename(
            columns={
                "plant_balance_w_common_diff": "plant_balance_w_common",
                "book_reserve_w_common_diff": "book_reserve_w_common",
                "unaccrued_balance_w_common_diff": "unaccrued_balance_w_common",
            }
        )
        .reset_index()  # compare_npb's index is the utility-ferc accout pk's
        .assign(
            plant_name_eia="balancing_account",
            fraction_ownership=1,
            depreciation_annual_epxns=lambda x: x.depreciation_annual_epxns_w_common
            * x.plant_balance_w_common,
            remaining_life_avg=lambda x: x.unaccrued_balance_w_common
            / x.depreciation_annual_epxns,
            record_id_eia=lambda x: (
                x.plant_name_eia
                + "_"
                + x.report_year.astype(str)
                + "_"
                + x.data_source
                + "_"
                + x.ferc_acct_name
                + "_"
                + x.utility_id_pudl.astype(str)
            ),
        )
        .set_index(["record_id_eia"])
    )
    logger.info(f"adding {len(balancing_records)} balancing account records.")
    ferc_deprish_eia = pd.concat([ferc_deprish_eia, balancing_records], join="outer")
    return ferc_deprish_eia


def allocate_balancing_account_to_assets(
    ferc_deprish_eia_w_ba: pd.DataFrame,
    idk_ba: List[str] = [
        "utility_name_eia",
        "operational_status",
        "ferc_acct_name",
        "data_source",
        "report_year",
    ],
    data_and_allocator_cols: dict = {
        "plant_balance_w_common": ["plant_balance_w_common"],
        "book_reserve_w_common": ["book_reserve_w_common"],
        "depreciation_annual_epxns_w_common": ["depreciation_annual_epxns_w_common"],
    },
):
    """Allocate the balancing account records across related assets.

    We generate balancing account records to true up the depreciation studies
    with the utility/FERC account level values reported to FERC1 via
    :func:`add_balancing_account`. This function takes those balancing account
    records and allocates them across assets. It utilizes
    :func:`pudl_rmi.connect_deprish_to_ferc1.allocate_cols`

    Args:
        ferc_deprish_eia_w_ba
        idk_ba: list of ID columns to allocate across.
        data_and_allocator_cols: dictionary of data columns to allocate (keys)
            and lists of column(s) to allocate based on.

    """
    # split records
    mask_ba = ferc_deprish_eia_w_ba.plant_name_eia == "Balancing_Account"
    balancing_accounts = ferc_deprish_eia_w_ba.loc[mask_ba]
    assets = ferc_deprish_eia_w_ba.loc[~mask_ba]
    data_cols = list(data_and_allocator_cols.keys())
    # merge the ba's onto the assets
    assets_w_ba = pd.merge(
        assets,
        balancing_accounts[idk_ba + data_cols],
        on=idk_ba,
        suffixes=("", "_ba"),
        how="left",
        validate="m:1",
    )
    # allocate each col
    assets_w_ba = allocate_cols(
        to_allocate=assets_w_ba,
        by=idk_ba,
        data_and_allocator_cols=data_and_allocator_cols,
    )
    if not (
        check_sums := all(
            ferc_deprish_eia_w_ba[data_cols].sum().round()
            == assets_w_ba[data_cols].sum().round()
        )
    ):
        raise AssertionError(
            "Allocating the balancing accounts changed the sum of the data cols"
            f"{check_sums}"
        )
    return assets_w_ba
