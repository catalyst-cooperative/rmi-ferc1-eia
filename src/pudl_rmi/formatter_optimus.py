"""Convert RMI outputs into model output format."""

import pandas as pd
import numpy as np
import logging

import pudl_rmi

logger = logging.getLogger(__name__)

col_to_rename = {
    "record_id_eia": "Unique_ID",
    "scenario_short": "Scenario_Short",  # generate
    "utility_name_eia": "Utility",
    "scenario": "Scenario",
    "operational_status": "Asset Status",
    "plant_name_eia": "Plants",
    "generator_id": "Unit",
    "resource_type": "Resource Type",  # generate
    "state": "State (Physical Location)",  # will be merged
    "fraction_owned": "Ownership Percentage (%)",
    "capacity_mw": "Net Capacity (MW)",
    "capacity_factor": "Capacity Factor (%)",
    "net_generation_mwh": "Net Generation (MWh)",
    "co2_mass_tons": "CO2 Emission (tons)",  # generate
    "total_fuel_cost": "Fuel Cost ($)",
    "variable_om": "Non-Fuel Variable O&M Costs ($)",  # generate
    "fixed_om": "Fixed O&M Costs ($)",  # generate
    "opex_nonfuel": "Total O&M Cost",
    "production_cost_total": "Total Production Costs ($)",  # generate
    "installation_year": "Commission Year",
    "remaining_life_avg": "Current Remaining Accounting Life (Yrs)",
    "plant_balance_w_common": "Gross Plant Balance/Original Cost ($)",
    "book_reserve_w_common": "Book Reserve/Accumulated Depreciation ($)",
    "unaccrued_balance_w_common": "Current Net Plant Balance ($)",
    "depreciation_annual_epxns_w_common": "Annual Depreciation Expense ($)",
    "depreciation_annual_rate": "Depreciation Rate (%)",
    "net_salvage_w_common": "Decommissioning Cost ($)",
    "arc_by_plant": "Special Asset Retirment Cost ($)",
    "capex_annual_addition": "Annual Capital Additions ($)",
    "faked_1": "Accumulated Deferred Income Tax (ADIT) ($)",
    "faked_2": "Accumulated Deferred Income Tax (ASC740) ($)",
    "faked_3": "Protected Excess Deferred Income Tax (EDIT) ($)",
    "faked_4": "Does this plant have an Flue-gas desulfurization (FGD) device? (Y/N)",
    "faked_5": "Ownership",
    "faked_6": "Accounting Retirement Year",
    "faked_7": "Maintenance CAPEX_2020",
    "faked_8": "Maintenance CAPEX_2021",
    "faked_9": "Maintenance CAPEX_2022",
    "faked_10": "Maintenance CAPEX_2023",
    "faked_11": "Maintenance CAPEX_2024",
    "faked_12": "Maintenance CAPEX_2025",
    "faked_13": "Maintenance CAPEX_2026",
    "faked_14": "Maintenance CAPEX_2027",
    "faked_15": "Maintenance CAPEX_2028",
    "faked_16": "Maintenance CAPEX_2029",
    "faked_17": "Maintenance CAPEX_2030",
    "faked_18": "Maintenance CAPEX_2031",
    "faked_19": "Maintenance CAPEX_2032",
    "faked_20": "Maintenance CAPEX_2033",
    "faked_21": "Maintenance CAPEX_2034",
    "faked_22": "Maintenance CAPEX_2035",
    "faked_23": "Maintenance CAPEX_2036",
    "faked_24": "Maintenance CAPEX_2037",
    "faked_25": "Maintenance CAPEX_2038",
    "faked_26": "Maintenance CAPEX_2039",
    "faked_27": "Maintenance CAPEX_2040",
    "faked_28": "Maintenance CAPEX_2041",
    "faked_29": "Maintenance CAPEX_2042",
    "faked_30": "Maintenance CAPEX_2043",
    "faked_31": "Maintenance CAPEX_2044",
    "faked_32": "Maintenance CAPEX_2045",
    "faked_33": "Maintenance CAPEX_2046",
    "faked_34": "Maintenance CAPEX_2047",
    "faked_35": "Maintenance CAPEX_2048",
    "faked_36": "Maintenance CAPEX_2049",
    "faked_37": "Maintenance CAPEX_2050",
    "faked_38": "CWIP_2020",
    "faked_39": "CWIP_2021",
    "faked_40": "CWIP_2022",
    "faked_41": "CWIP_2023",
    "faked_42": "CWIP_2024",
    "faked_43": "CWIP_2025",
    "faked_44": "CWIP_2026",
    "faked_45": "CWIP_2027",
    "faked_46": "CWIP_2028",
    "faked_47": "CWIP_2029",
    "faked_48": "CWIP_2030",
    "faked_49": "CWIP_2031",
    "faked_50": "CWIP_2032",
    "faked_51": "CWIP_2033",
    "faked_52": "CWIP_2034",
    "faked_53": "CWIP_2035",
    "faked_54": "CWIP_2036",
    "faked_55": "CWIP_2037",
    "faked_56": "CWIP_2038",
    "faked_57": "CWIP_2039",
    "faked_58": "CWIP_2040",
    "faked_59": "CWIP_2041",
    "faked_60": "CWIP_2042",
    "faked_61": "CWIP_2043",
    "faked_62": "CWIP_2044",
    "faked_63": "CWIP_2045",
    "faked_64": "CWIP_2046",
    "faked_65": "CWIP_2047",
    "faked_66": "CWIP_2048",
    "faked_67": "CWIP_2049",
    "faked_68": "CWIP_2050",
    "line_id": "Record ID Depreciation",
    "record_id_ferc1": "Record ID FERC 1",
    "plant_id_eia": "Plant ID EIA",
    "report_year": "Report Year",
    "data_source": "Data Source of Depreciation Study",
    "ferc_acct_name": "FERC Acct",
}

tech_descrpt_to_resource_type = {
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


def convert_to_model_format(scaled_df, pudl_out, util_ids_pudl, years):
    """Convert RMI outputs into optimus model inputs."""
    # add plant state
    scaled_df = scaled_df.merge(
        pudl_out.plants_eia860()[
            [
                "plant_id_eia",
                "report_date",
                "state",
            ]
        ].drop_duplicates(),
        on=[
            "plant_id_eia",
            "report_date",
        ],
        validate="m:1",
        how="left",
    )
    # merge in the EIA utility name
    scaled_df = scaled_df.merge(
        pudl_out.utils_eia860()[
            ["utility_id_pudl", "utility_name_eia"]
        ].drop_duplicates(subset=["utility_id_pudl"]),
        on=["utility_id_pudl"],
        validate="m:1",
        how="left",
    )

    # maybe we should combine these two methodologies...
    empty_cols = [
        "scenario_short",
        "scenario",
        "variable_om",
        "fixed_om",
        "co2_mass_tons",
        "production_cost_total",
    ]
    for col in empty_cols:
        scaled_df.loc[:, col] = pd.NA
    for faked_col in [x for x in col_to_rename.keys() if "faked_" in x]:
        scaled_df.loc[:, faked_col] = pd.NA

    scaled_df.loc[:, "resource_type"] = scaled_df.technology_description.replace(
        tech_descrpt_to_resource_type
    )
    scaled_df.loc[:, "operational_status"] = scaled_df.loc[
        :, "operational_status"
    ].str.title()
    if util_ids_pudl:
        scaled_df = scaled_df[
            scaled_df.utility_id_pudl.isin(util_ids_pudl)
            & scaled_df.report_year.isin(years)
        ]

    model_input = (
        scaled_df.reset_index()
        .dropna(subset=["record_id_eia"])
        .rename(
            columns=col_to_rename,
        )[list(col_to_rename.values())]
        .round({k: 1 for k in [c for c in scaled_df.columns if ("($)" in c)]})
    )

    model_input = model_input.replace(
        {
            "Utility": {
                "Carolina Power & Light Co": "Duke Energy Progress",
                "Duke Energy Corp": "Duke Energy Carolinas",
            },
        }
    )

    return model_input


#################################
# Temp functions for Duke outputs
#################################


def fake_duke_deprish_eia_for_mod(df_to_fake, ppe):
    """Temp function to fake Duke's deprish records for modernization."""
    logger.info("Adding fake years of Duke data....")
    # og_index = df_to_fake.index.names
    # df_to_fake = df_to_fake.reset_index()
    fake_year_dfs = []
    to_fake_2018 = df_to_fake[
        df_to_fake.utility_id_pudl.isin([90, 97])
        & (df_to_fake.report_date.dt.year == 2018)
    ].reset_index()[
        [
            c
            for c in df_to_fake.reset_index()
            if c
            in ["record_id_eia", "line_id"]  # core IDs
            + pudl_rmi.deprish.IDX_COLS_DEPRISH  # deprish IDS
        ]
    ]
    for fake_year in [2019, 2020]:
        fake_new_year = (
            to_fake_2018.copy()
            .assign(
                report_year=fake_year,
                report_date=pd.to_datetime(fake_year, format="%Y"),
            )
            .replace(
                {
                    "record_id_eia": "_2018_",
                    "line_id": "2018_",
                },
                {"record_id_eia": f"_{fake_year}_", "line_id": f"{fake_year}_"},
                regex=True,
            )
        )
        fake_new_year = fake_new_year[
            ~(
                (fake_new_year.report_date.dt.year == 2020)
                & (df_to_fake.ferc_acct_name.str.lower() == "nuclear")
            )
        ]
        fake_year_dfs.append(fake_new_year)

    # concat the fake years & merge back in the ppe columns
    fake_years_squish = pd.concat(fake_year_dfs)
    if fake_years_squish.index.name != "record_id_eia":
        fake_years_squish = fake_years_squish.set_index(["record_id_eia"])

    fake_new_years = fake_years_squish.merge(
        ppe[
            ppe.utility_id_pudl.isin([90, 97])
            & ppe.report_date.dt.year.isin([2019, 2020])
        ][  # mask to make this quicker
            [c for c in ppe if c not in fake_new_year]
        ],
        left_index=True,
        right_index=True,
        how="left",
        validate="m:1",
    ).reset_index()

    de_faked = pd.concat(
        [df_to_fake, fake_new_years], join="outer"  # .set_index(og_index),
    )
    assert ~de_faked[de_faked.report_date.dt.year == 2020].empty
    return de_faked


def append_non_plant_deprish_records(d, ferc_deprish_eia, ppe):
    """Add the T&D records into the output with faked record_id_eia."""
    scaled_append = ferc_deprish_eia.reset_index()
    # ensure the depreciation data does not have stray columns that aren't in
    # the deprish/EIA combo
    d_non_plant = (
        d[
            ~d.line_id.isin(
                scaled_append["line_id"].str.split("; ", expand=True).stack().unique()
            )
            & (~d.common | d.common.isnull())
            & (d.plant_id_eia.isnull())
        ]
        .assign(report_year=lambda x: x.report_date.dt.year)
        .convert_dtypes(convert_floating=False)
    )
    d_non_plant.loc[:, "faked_id"] = (
        d_non_plant.ferc_acct_name
        + "_"
        + d_non_plant.plant_part_name
        + "_"
        + d_non_plant.report_year.astype(str)
        + "_"
        + d_non_plant.utility_id_pudl.astype(str)
        + "_"
        + d_non_plant.data_source
    )

    if "record_id_eia" not in d_non_plant:
        d_non_plant.loc[:, "record_id_eia"] = pd.NA
    d_non_plant.loc[:, "record_id_eia"] = d_non_plant.record_id_eia.fillna(
        d_non_plant.faked_id
    )

    d_non_plant = fake_duke_deprish_eia_for_mod(d_non_plant, ppe)
    # make up a fake "record_id_eia" for just the T&D records
    de_w_td = (
        pd.concat([scaled_append, d_non_plant])
        .assign(
            operational_status=lambda x: np.where(
                (
                    x.record_id_eia.isnull()
                    & x.plant_part_name.notnull()
                    & x.ferc_acct_name.str.lower().isin(
                        ["distribution", "general", "transmission", "intangible"]
                    )
                ),
                "existing",
                x.operational_status,
            ),
        )
        .convert_dtypes(convert_floating=False)
    )

    de_w_td = (
        de_w_td.drop(columns=["plant_part_name"])
        .set_index("record_id_eia")
        .sort_index()
    )
    return de_w_td
