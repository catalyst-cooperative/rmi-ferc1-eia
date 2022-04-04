"""Convert RMI outputs into model output format."""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

RENAME_COLS = {
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
    "opex_nonfuel": "Total O&M Cost",
    "faked_6": "Total Production Costs ($)",
    "installation_year": "Commission Year",
    "remaining_life_avg": "Current Remaining Accounting Life (Yrs)",
    "plant_balance_w_common": "Gross Plant Balance/Original Cost ($)",
    "book_reserve_w_common": "Book Reserve/Accumulated Depreciation ($)",
    "unaccrued_balance_w_common": "Current Net Plant Balance ($)",
    "depreciation_annual_epxns_w_common": "Annual Depreciation Expense ($)",
    "depreciation_annual_rate": "Depreciation Rate (%)",
    "net_salvage_w_common": "Decommissioning Cost ($)",
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
    "ferc_acct_name": "FERC Acct",
}

TECHNOLOGY_DESCRIPTION_TO_RESOURCE_TYPE = {
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
    """
    # add plant state
    model_input = (
        deprish_ferc1_eia.merge(
            plants_eia860[["plant_id_eia", "report_date", "state"]].drop_duplicates(),
            on=["plant_id_eia", "report_date"],
            validate="m:1",
            how="left",
        )
        .merge(  # merge in the EIA utility name
            utils_eia860[["utility_id_pudl", "utility_name_eia"]].drop_duplicates(),
            on=["utility_id_pudl"],
            validate="m:1",
            how="left",
        )
        .replace({"utility_name_eia": UTILITY_RENAME})
        .reset_index()
        .dropna(subset=["record_id_eia"])
        .round({k: 1 for (k, v) in RENAME_COLS.items() if "($)" in v})
    )

    model_input.loc[:, "resource_type"] = model_input.technology_description.replace(
        TECHNOLOGY_DESCRIPTION_TO_RESOURCE_TYPE
    )
    model_input.loc[
        :, "operational_status"
    ] = model_input.operational_status.str.title()

    # Create a bunch of "faked"  columns
    for faked_col in [x for x in RENAME_COLS.keys() if "faked_" in x]:
        model_input.loc[:, faked_col] = pd.NA

    model_input = model_input.rename(columns=RENAME_COLS)[list(RENAME_COLS.values())]

    return model_input
