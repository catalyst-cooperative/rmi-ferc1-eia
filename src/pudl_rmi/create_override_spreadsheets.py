"""
Create output spreadsheets used for overriding and checking FERC-EIA record matches.

The connect_ferc1_to_eia.py module uses machine learning to link records from FERC
with records from EIA. While this process is waaay better more efficient and logical
than a human, it requires a set of hand-compiled training data in order to do it's job.
On top of that, it's not always capable of mapping some of the trickier records that
humans might be able to make sense of based on past or surrounding records.

This module creates an output spreadsheet, based on a certain utility, that makes the
matching and machine-matched human validation process much easier. It also contains
functions that will read those new/updated/validated matches from the spreadsheet and
incorporate them into the existing training data.

"""
import logging
import os
import pathlib

import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

relevant_cols_ferc_eia = [
    "record_id_ferc1",
    "record_id_eia",
    "true_gran",
    "report_year",
    "match_type",
    "plant_part",
    "ownership",
    "utility_id_eia",
    "utility_id_pudl",
    "utility_name_ferc1",
    "plant_id_pudl",
    "unit_id_pudl",
    "generator_id",
    "plant_name_ferc1",
    "plant_name_new",
    "fuel_type_code_pudl_ferc1",
    "fuel_type_code_pudl_eia",
    "net_generation_mwh_ferc1",
    "net_generation_mwh_eia",
    "capacity_mw_ferc1",
    "capacity_mw_eia",
    "capacity_factor_ferc1",
    "capacity_factor_eia",
    "total_fuel_cost_ferc1",
    "total_fuel_cost_eia",
    "total_mmbtu_ferc1",
    "total_mmbtu_eia",
    "fuel_cost_per_mmbtu_ferc1",
    "fuel_cost_per_mmbtu_eia",
    "installation_year_ferc1",
    "installation_year_eia",
]

relevant_cols_mul = [
    "record_id_eia",
    "report_year",
    "utility_id_pudl",
    "utility_id_eia",
    "utility_name_eia",  # I add this in from the utils_eia860() table
    "operational_status_pudl",
    "true_gran",
    "plant_part",
    "ownership_dupe",
    "fraction_owned",
    "plant_id_eia",
    "plant_id_pudl",
    "plant_name_new",
    "generator_id",
    "capacity_mw",
    "capacity_factor",
    "net_generation_mwh",
    "installation_year",
    "fuel_type_code_pudl",
    "total_fuel_cost",
    "total_mmbtu",
    "fuel_cost_per_mmbtu",
    "heat_rate_mmbtu_mwh",
]

output_path = pathlib.Path().cwd().parent / "outputs"
fixed_overrides_path = pathlib.Path().cwd().parent / "add_to_training"
training_path = pathlib.Path().cwd().parent / "inputs" / "train_ferc1_to_eia_copy.csv"
training_data = pd.read_csv(training_path)

# --------------------------------------------------------------------------------------
# Generate Override Tools
# --------------------------------------------------------------------------------------


def _pct_diff(df, col):
    """Calculate percent difference between EIA and FERC versions of a column."""
    df.loc[
        (df[f"{col}_eia"] > 0) & (df[f"{col}_ferc1"] > 0), f"{col}_pct_diff"
    ] = round(((df[f"{col}_ferc1"] - df[f"{col}_eia"]) / df[f"{col}_ferc1"] * 100), 2)


def is_best_match(df):
    """Return a string indicating whether FERC-EIA match is immediately passable.

    The process of manually checking all of the FERC-EIA matches made by the machine
    learning algorithm is tedius. This function makes it easier to speed through the
    obviously good matches and pay more attention to those that are more questionable.

    Currently, a "best match" is comprised of a FERC-EIA match with a capacity percent
    difference of less than 6%, a net generation percent difference of less than 6%,
    and an installation year difference of less than 3 years.

    """
    message = []
    if abs(df.capacity_mw_pct_diff) < 6:
        message.append("cap")
    if abs(df.net_generation_mwh_pct_diff) < 6:
        message.append("net-gen")
    if abs(df.installation_year_diff) < 3:
        message.append("inst-y")

    return "_".join(message)


def _prep_ferc_eia(ferc1_eia, pudl_out):
    """Prep FERC-EIA for use in override output sheet pre-utility subgroups."""
    logger.debug("Prepping FERC-EIA table")
    check_connections = ferc1_eia[relevant_cols_ferc_eia].copy()

    # Add a column to tell whether it's a good match, who verified / made the match,
    # and any notes about weirdness.
    check_connections.insert(0, "verified", np.nan)
    check_connections.insert(1, "used_match_record", np.nan)
    check_connections.insert(2, "signature_1", np.nan)
    check_connections.insert(3, "signature_2", np.nan)
    check_connections.insert(4, "notes", np.nan)
    check_connections.insert(6, "record_id_eia_override_1", np.nan)
    check_connections.insert(7, "record_id_eia_override_2", np.nan)
    check_connections.insert(8, "record_id_eia_override_3", np.nan)
    check_connections.insert(9, "best_match", np.nan)

    # put these in the right order to be filled in by pct_diff
    check_connections.insert(26, "fuel_type_code_pudl_diff", np.nan)
    check_connections.insert(29, "net_generation_mwh_pct_diff", np.nan)
    check_connections.insert(32, "capacity_mw_pct_diff", np.nan)
    check_connections.insert(35, "capacity_factor_pct_diff", np.nan)
    check_connections.insert(38, "total_fuel_cost_pct_diff", np.nan)
    check_connections.insert(41, "total_mmbtu_pct_diff", np.nan)
    check_connections.insert(44, "fuel_cost_per_mmbtu_pct_diff", np.nan)
    check_connections.insert(47, "installation_year_diff", np.nan)

    # Fix some column names
    check_connections.rename(
        columns={
            "utility_id_pudl_ferc1": "utility_id_pudl",
            "plant_id_pudl_ferc1": "plant_id_pudl",
            "plant_name_new": "plant_name_eia",
        },
        inplace=True,
    )

    # Add pct diff columns
    for col in [
        "net_generation_mwh",
        "capacity_mw",
        "capacity_factor",
        "total_fuel_cost",
        "total_mmbtu",
        "fuel_cost_per_mmbtu",
    ]:
        _pct_diff(check_connections, col)

    # Add best match col
    # check_connections["best_match"] = _is_best_match(check_connections)

    # Add qualitative similarity columns (fuel_type_code_pudl)
    check_connections.loc[
        (check_connections.fuel_type_code_pudl_eia.notna())
        & (check_connections.fuel_type_code_pudl_ferc1.notna()),
        "fuel_type_code_pudl_diff",
    ] = check_connections.fuel_type_code_pudl_eia == (
        check_connections.fuel_type_code_pudl_ferc1
    )

    # Add quantitative similarity columns (installation year)
    check_connections.loc[
        :, "installation_year_ferc1"
    ] = check_connections.installation_year_ferc1.astype("Int64")
    check_connections.loc[
        (check_connections.installation_year_eia.notna())
        & (check_connections.installation_year_ferc1.notna()),
        "installation_year_diff",
    ] = (
        check_connections.installation_year_eia
        - check_connections.installation_year_ferc1
    )

    # Move record_id_ferc1
    record_id_ferc1 = check_connections.pop("record_id_ferc1")
    check_connections.insert(9, "record_id_ferc1", record_id_ferc1)

    # Add utility name eia
    utils = pudl_out.utils_eia860().assign(report_year=lambda x: x.report_date.dt.year)[
        ["utility_id_eia", "utility_name_eia", "report_year"]
    ]
    check_connections = pd.merge(
        check_connections,
        utils,
        on=["utility_id_eia", "report_year"],
        how="left",
        validate="m:1",
    ).rename(columns={"utility_name_eia": "utility_name"})

    check_connections.insert(19, "utility_name_eia", check_connections.utility_name)
    check_connections = check_connections.drop(columns=["utility_name"])

    # Add column with excel formula to check if the override record id is the same
    # as the AI assigend id.
    check_connections = check_connections.reset_index().assign(
        used_match_record=lambda x: (  # can this be moved to prep?
            "=(F"
            + (x.index + 2).astype("str")
            + "=K"
            + (x.index + 2).astype("str")
            + ")"
        )
    )

    return check_connections


def _prep_ppl(ppl, pudl_out):
    """Prep MUL for use in override output sheet pre-utility subgroups."""
    logger.debug("Prepping Plant Parts Table")

    # Add utilty name eia and only take relevant columns
    ppl_out = (
        ppl.reset_index()
        .merge(
            pudl_out.utils_eia860()[
                ["utility_id_eia", "utility_name_eia", "report_date"]
            ].copy(),
            on=["utility_id_eia", "report_date"],
            how="left",
            validate="m:1",
        )[relevant_cols_mul]
        .copy()
    )

    return ppl_out


def _prep_deprish(deprish, pudl_out):
    """Prep depreciation data for use in override output sheet pre-utility subgroups."""
    logger.debug("Prepping Deprish Data")

    # Get utility_id_eia from EIA
    util_df = pudl_out.utils_eia860()
    id_dict = dict(zip(util_df["utility_id_pudl"], util_df["utility_id_eia"]))

    deprish_out = deprish.assign(
        report_year=lambda x: x.report_date.dt.year.astype("Int64"),
        utility_id_eia=lambda x: x.utility_id_pudl.map(id_dict).astype("Int64"),
    )

    return deprish_out


def generate_input_dfs(pudl_out, rmi_out):
    """Load and prep and all the input tables."""
    logger.debug("Generating inputs")
    inputs_dict = {
        "ferc_eia": rmi_out.grab_ferc1_to_eia().pipe(_prep_ferc_eia, pudl_out),
        "ppl": rmi_out.grab_plant_part_list().pipe(_prep_ppl, pudl_out),
        "deprish": rmi_out.grab_deprish().pipe(_prep_deprish, pudl_out),
    }
    logger.info("")

    return inputs_dict


def _get_util_year_subsets(inputs_dict, util_id_eia_list, years):
    """Get utility and year subsets for each of the input dfs."""
    util_year_subset_dict = {}
    for df_name, df in inputs_dict.items():
        logger.debug(f"Getting utility-year subset for {df_name}")
        in_year_range = df["report_year"].isin(years)
        in_eia_util_list = df["utility_id_eia"].isin(util_id_eia_list)
        subset_df = df[in_year_range & in_eia_util_list].copy()
        # Make sure dfs aren't too big...
        assert (
            len(subset_df) < 500000
        ), "Your subset is more than 500,000 rows...this \
            is going to make excel reaaalllllyyy slow. Try entering a smaller utility \
            or year subset"
        util_year_subset_dict[f"{df_name}_util_year_subset"] = subset_df

    return util_year_subset_dict


def _output_override_sheet(util_year_subset_dict, util_name):
    """Put three tables from in the inputs_dict into excel tabs and output."""
    # Enable unique file names and put all files in directory called overrides
    new_output_path = (
        output_path / "overrides" / f"{util_name}_fix_FERC-EIA_overrides.xlsx"
    )
    # Output file to a folder called overrides
    logger.info("Outputing table subsets to tabs\n")
    writer = pd.ExcelWriter(new_output_path, engine="xlsxwriter")
    for df_name, df in util_year_subset_dict.items():
        df.to_excel(writer, sheet_name=df_name, index=False)
    writer.save()


def generate_override_tools(pudl_out, rmi_out, util_dict, years):
    """Generate inputs and generate override sheets."""
    # Generate full input tables
    inputs_dict = _generate_input_dfs(pudl_out, rmi_out)

    # Make sure overrides dir exists
    if not os.path.isdir(output_path / "overrides"):
        os.mkdir(output_path / "overrides")

    # For each utility, make an override sheet with the correct input table slices
    for util_name, util_id_eia_list in util_dict.items():
        logger.info(f"Developing outputs for {util_name}")
        util_year_subset_dict = _get_util_year_subsets(
            inputs_dict, util_id_eia_list, years
        )
        _output_override_sheet(util_year_subset_dict, util_name)


# Removed deprish for now b/c not working
def output_override_tools(ferc_eia, ppl, deprish, util_ids, util_name, years, pudl_out):
    """Create output spreadsheets based on specified utilities and years.

    This is the function that brings all the steps together. It loads the input tables,
    preps them, segements them by year and utility, and outputs them into spreadsheets
    in a folder called overrides.

    """
    logger.info(f"Making override file for {util_name.upper()}")

    # Load, prep and get subsets for each table: FERC-EIA, PPL, Deprish
    logger.info("Loading ferc-eia subset")
    ferc_eia_subset = (
        ferc_eia.pipe(_prep_ferc_eia, pudl_out)
        .pipe(_get_util_year_subsets, util_ids, years)
        .reset_index(drop=True)
        # Create a column that produces an excel function telling the user whether their
        # selected match matches that used by the record linkage ML or not.
        .assign(
            used_match_record=lambda x: (  # can this be moved to prep?
                "=(F"
                + (x.index + 2).astype("str")
                + "=K"
                + (x.index + 2).astype("str")
                + ")"
            )
        )
    )

    logger.info("Loading plant part list subset")
    ppl_subset = ppl.pipe(_prep_ppl, pudl_out).pipe(
        _get_util_year_subsets, util_ids, years
    )

    logger.info("Loading depreciation subset")
    deprish_subset = _get_util_year_subsets(deprish, util_ids, years, deprish=True)

    # Create a dict of each df and the tab name you want to give it in the output sheet
    tool_dict = {
        "ferc_eia_util_subset": ferc_eia_subset,
        "mul_util_subset": ppl_subset,
        "deprish_util_subset": deprish_subset,
    }

    # Make sure overrides dir exists
    if not os.path.isdir(output_path / "overrides"):
        os.mkdir(output_path / "overrides")

    # Enable unique file names and put all files in directory called overrides
    new_output_path = (
        output_path / "overrides" / f"{util_name}_fix_FERC-EIA_overrides.xlsx"
    )

    # Output file to a folder called overrides
    logger.info("Outputing table subsets to tabs\n")
    writer = pd.ExcelWriter(new_output_path, engine="xlsxwriter")
    for tab, df in tool_dict.items():
        df.to_excel(writer, sheet_name=tab, index=False)
    writer.save()


# --------------------------------------------------------------------------------------
# Upload Changes to Training Data
# --------------------------------------------------------------------------------------


def validate_override_fixes(
    validated_connections, ferc1_eia, training_data, expect_override_overrides=False
):
    """Process the verified and/or fixed matches.

    Args:
        validated_connections (pd.DataFrame): A dataframe in the add_to_training
            directory that is ready to be added to validated and subsumed into the
            training data.
        ferc1_eia (pd.DataFrame): The current FERC-EIA table
        expect_override_overrides (boolean): Whether you expect the tables to have
            overridden matches already in the training data.
    Raises:
        AssertionError: If there are ferc record ids that aren't in the original
            FERC-EIA connection
        AssertionError: If there are eia override id records that aren't in the original
            FERC-EIA connection
        AssertionError: If there are eia override id records that don't correspond to
            the correct report year
        AssertionError: If you didn't expect to override overrides but the data does
    Returns:
        pd.DataFrame: The validated FERC-EIA dataframe you're trying to add to the
            training data.

    """
    logger.info("Validating overrides")

    # Make sure there are no rouge descriptions in the verified field (besides TRUE)
    match_language = validated_connections.verified.unique()
    assert (
        len(outliers := [x for x in match_language if x not in [True, False, pd.NA]])
        == 0
    ), f"All correct matches must be marked TRUE; found {outliers}"

    # Get TRUE records
    true_connections = validated_connections[validated_connections["verified"]].copy()

    # Make sure that the eia and ferc ids haven't been tampered with
    assert (
        len(
            bad_eia := [
                x
                for x in true_connections.dropna().record_id_eia_override_1.unique()
                if x not in ferc1_eia.record_id_eia.unique()
            ]
        )
        == 0
    ), f"Found record_id_eia_override_1 values that aren't in the existing FERC-EIA \
          connection: {bad_eia}"
    assert (
        len(
            bad_ferc := [
                x
                for x in true_connections.dropna().record_id_ferc1.unique()
                if x not in ferc1_eia.record_id_ferc1.unique()
            ]
        )
        == 0
    ), f"Found record_id_ferc1 values that aren't in the existing FERC-EIA \
        connection: {bad_ferc}"

    # Make sure the year in the suggested eia id overrides match the year in the
    # report_year column
    year_ser = true_connections.record_id_eia_override_1.str.extract(r"(_20\d{2})")[
        0
    ].str.replace("_", "")
    year_ser_int = pd.to_numeric(year_ser, errors="coerce").astype("Int64")
    assert (
        len(bad_eia := true_connections["report_year"].compare(year_ser_int)) == 0
    ), f"Found record_id_eia_override_1 values that don't correspond to the right \
        report year: {bad_eia}"

    if not expect_override_overrides:
        # Make sure that these aren't already in the overrides (this should be
        # impossible, but just in case)
        assert (
            len(
                bad_eia := [
                    x
                    for x in true_connections.record_id_eia_override_1.unique()
                    if x
                    in training_data.dropna(
                        subset=["record_id_eia"]
                    ).record_id_eia.unique()
                ]
            )
            == 0
        ), f"Found record_id_eia_override_1 values that are already in the existing \
            FERC-EIA training data: {bad_eia}"
        assert (
            len(
                bad_ferc := [
                    x
                    for x in true_connections.record_id_ferc1.unique()
                    if x
                    in training_data.dropna(
                        subset=["record_id_eia"]
                    ).record_id_ferc1.unique()
                ]
            )
            == 0
        ), f"Found record_id_ferc1 values that are already in the existing FERC-EIA \
            training data: {bad_ferc}"

    return true_connections


def add_to_training(new_overrides, training_data):
    """Add the new overrides to the old override sheet."""
    logger.info("Combining all new overrides with existing training data")
    training_data_out = training_data.append(
        new_overrides[["record_id_eia", "record_id_ferc1", "signature_1", "notes"]]
    ).set_index(["record_id_eia", "record_id_ferc1"])

    return training_data_out


def validate_and_add_to_training(ferc1_eia, expect_override_overrides=False):
    """Validate, combine, and add overrides to the training data.

    Validating and combinging the records so you only have to loop through the files
    once. Runs the validate_override_fixes() function and add_to_training.

    Args:
        expect_override_overrides (bool): This value is explicitly assigned at the top
            of the notebook.
        ferc1_eia (pandas.DataFrame): The current FERC-EIA connection table, not the
            overrides. This is passed to the validate_override_fixes() function to
            verify that a record in the override sheet shows up in the FERC-EIA sheet.

    Returns:
        pandas.DataFrame: A DataFrame with all of the new overrides combined.
    """
    all_overrides = pd.DataFrame(
        columns=["record_id_eia", "record_id_ferc1", "signature_1", "notes"]
    )

    # Loop through all the files, validate, and combine them.
    all_files = os.listdir(fixed_overrides_path)
    good_files = [file for file in all_files if file.endswith(".xlsx")]
    for file in good_files:
        logger.info(f"Processing fixes in {file}")
        file_df = (
            pd.read_excel(
                (fixed_overrides_path / file), sheet_name="ferc_eia_util_subset"
            )
            .assign(verified=lambda x: x.verified.astype("boolean").fillna(False))
            .pipe(
                validate_override_fixes,
                ferc1_eia,
                training_data,
                expect_override_overrides=expect_override_overrides,
            )
            .rename(
                columns={
                    "record_id_eia": "record_id_eia_old",
                    "record_id_eia_override_1": "record_id_eia",
                }
            )
        )
        all_overrides = all_overrides.append(
            file_df[["record_id_eia", "record_id_ferc1", "signature_1", "notes"]]
        )

    # Add the records to the training data
    logger.info("Adding overrides to training data")
    new_training_df = add_to_training(all_overrides, training_data)
    new_training_df.to_csv(training_path)

    return new_training_df
