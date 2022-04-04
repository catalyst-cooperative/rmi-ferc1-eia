"""
WIP!!! Modernize the depreciation studies.

This is UNDER CONSTRUCTION. Right now, this only supports generating the
modernizable records that we've been using to mock up the modernization process
for Duke.
"""

import pandas as pd
import numpy as np
import logging

import pudl_rmi

logger = logging.getLogger(__name__)

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
