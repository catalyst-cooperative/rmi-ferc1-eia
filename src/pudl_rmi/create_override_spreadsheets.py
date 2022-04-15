"""
Create output spreadsheets used for overriding and checking FERC-EIA record matches.

The connect_ferc1_to_eia.py module uses machine learning to link records from FERC
with records from EIA. While this process is waaay better more efficient and logical
than a human, it requires a set of hand-compiled training data in order to do it's job.
On top of that, it's not always capable of mapping some of the trickier records that
humans might be able to make sense of based on past or surrounding records.

This module creates an output spreadsheet, based on a certain utility, that makes the
matching and machine-matched human validation process much easier. It also contains
functions that will read those new/updated/validated matches from the spreadsheet,
validate them, and incorporate them into the existing training data.

"""
import logging
import os

import numpy as np
import pandas as pd

import pudl_rmi

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RELEVANT_COLS_FERC_EIA = [
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

RELEVANT_COLS_PPL = [
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

# --------------------------------------------------------------------------------------
# Generate Override Tools
# --------------------------------------------------------------------------------------


def _pct_diff(df, col) -> None:
    """Calculate percent difference between EIA and FERC versions of a column."""
    df.loc[
        (df[f"{col}_eia"] > 0) & (df[f"{col}_ferc1"] > 0), f"{col}_pct_diff"
    ] = round(((df[f"{col}_ferc1"] - df[f"{col}_eia"]) / df[f"{col}_ferc1"] * 100), 2)


def is_best_match(df):  # not currently included!
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


def _prep_ferc_eia(ferc1_eia, pudl_out) -> pd.DataFrame:
    """Prep FERC-EIA for use in override output sheet pre-utility subgroups."""
    logger.debug("Prepping FERC-EIA table")
    check_connections = ferc1_eia[RELEVANT_COLS_FERC_EIA].copy()

    # Add columns for user input and percent diff. These lists are specificalled
    # ordered so that they appear in the df as they appear in each list.
    ordered_input_values_cols = [
        "verified",
        "used_match_record",
        "signature_1",
        "signature_2",
        "notes",
        "record_id_override_1",
        "record_id_override_2",
        "record_id_override_3",
        "best_match",
    ]
    ordered_pct_diff_cols = [
        "fuel_type_code_pudl_diff",
        "net_generation_mwh_pct_diff",
        "capacity_mw_pct_diff",
        "capacity_factor_pct_diff",
        "total_fuel_cost_pct_diff",
        "total_mmbtu_pct_diff",
        "fuel_cost_per_mmbtu_pct_diff",
        "installation_year_diff",
    ]

    for row in range(0, len(ordered_input_values_cols)):
        check_connections.insert(row, ordered_input_values_cols[row], np.nan)

    pct_diff_col_loc = check_connections.columns.get_loc("fuel_type_code_pudl_eia") + 1
    for col in ordered_pct_diff_cols:
        check_connections.insert(pct_diff_col_loc, col, np.nan)
        pct_diff_col_loc = pct_diff_col_loc + 3

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
        check_connections.dropna(
            subset=["record_id_ferc1"]
        ),  # dropna for now becuase somehow there is a full NA row in ferc-eia....figure out later
        utils,
        on=["utility_id_eia", "report_year"],
        how="left",
        validate="m:1",
    )

    utility_name_eia = check_connections.pop("utility_name")
    check_connections.insert(19, "utility_name_eia", utility_name_eia)

    return check_connections


def _prep_ppl(ppl, pudl_out) -> pd.DataFrame:
    """Prep PPL table for use in override output sheet pre-utility subgroups."""
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
        )[RELEVANT_COLS_PPL]
        .copy()
    )

    return ppl_out


def _prep_deprish(deprish, pudl_out) -> pd.DataFrame:
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


def _generate_input_dfs(pudl_out, rmi_out) -> dict:
    """Load ferc_eia, ppl, and deprish tables into a dictionary.

    Loading all of these tables once is much faster than loading then repreatedly for
    every utility/year iteration. These tables will be segmented by utility and year
    in _get_util_year_subsets() and loaded as seperate tabs in a spreadsheet in
    _output_override_sheet().

    Returns:
        dict: A dictionary where keys are string names for ferc_eia, ppl, and deprish
            tables and values are the actual tables in full.

    """
    logger.debug("Generating inputs")
    inputs_dict = {
        "ferc_eia": rmi_out.ferc1_to_eia().pipe(_prep_ferc_eia, pudl_out),
        "ppl": rmi_out.plant_parts_eia().pipe(_prep_ppl, pudl_out),
        "deprish": rmi_out.deprish().pipe(_prep_deprish, pudl_out),
    }

    return inputs_dict


def _get_util_year_subsets(inputs_dict, util_id_eia_list, years) -> dict:
    """Get utility and year subsets for each of the input dfs.

    After generating the dictionary with all of the inputs tables loaded, we'll want to
    create subsets of each of those tables based on the utility and year inputs we're
    given. This function takes the input dict generated in _generate_input_dfs() and
    outputs an updated version with df values pertaining to the utilities in
    util_id_eia_list and years in years.

    Args:
        inputs_dict (dict): The output of running _generation_input_dfs()
        util_id_eia_list (list): A list of the utility_id_eia values you want to
            include in a single spreadsheet output. Generally this is a list of the
            subsidiaries that pertain to a single parent company.
        years (list): A list of the years you'd like to add to the override sheets.

    Returns:
        dict: A subset of the inputs_dict that contains versions of the value dfs that
            pertain only to the utilites and years specified in util_id_eia_list and
            years.

    """
    util_year_subset_dict = {}
    for df_name, df in inputs_dict.items():
        logger.debug(f"Getting utility-year subset for {df_name}")
        subset_df = df[
            (df["report_year"].isin(years))
            & (df["utility_id_eia"].isin(util_id_eia_list))
        ].copy()
        # Make sure dfs aren't too big...
        if len(subset_df) < 500000:
            raise AssertionError(
                "Your subset is more than 500,000 rows...this \
                is going to make excel reaaalllllyyy slow. Try entering a smaller utility \
                or year subset"
            )

        if df_name == "ferc_eia":
            # Add column with excel formula to check if the override record id is the
            # same as the AI assigend id. Doing this here instead of prep_ferc_eia
            # because it is based on index number which is changes when you take a
            # subset of the data.
            subset_df = (
                subset_df.reset_index()
                .assign(
                    used_match_record=lambda x: (  # can this be moved to prep?
                        "=(J"
                        + (x.index + 2).astype("str")
                        + "=F"
                        + (x.index + 2).astype("str")
                        + ")"
                    )
                )
                .drop(columns=["index"])
            )

        util_year_subset_dict[f"{df_name}_util_year_subset"] = subset_df

    return util_year_subset_dict


def _output_override_spreadsheet(util_year_subset_dict, util_name) -> None:
    """Output spreadsheet with tabs for ferc-eia, ppl, deprish for one utility.

    Args:
        util_year_subset_dict (dict): The output from _get_util_year_subsets()
        util_name (str): A string indicating the name of the utility that you are
            creating an override sheet for. The string will be used as the suffix for
            the name of the excel file. Ex: for util_name = "BHE", the file name will be
            BHE_fix_FERC-EIA_overrides.xlsx.

    """
    # Enable unique file names and put all files in directory called overrides
    new_output_path = (
        pudl_rmi.OUTPUTS_DIR / "overrides" / f"{util_name}_fix_FERC-EIA_overrides.xlsx"
    )
    # Output file to a folder called overrides
    logger.info("Outputing table subsets to tabs\n")
    writer = pd.ExcelWriter(new_output_path, engine="xlsxwriter")
    for df_name, df in util_year_subset_dict.items():
        df.to_excel(writer, sheet_name=df_name, index=False)
    writer.save()


def generate_all_override_spreadsheets(pudl_out, rmi_out, util_dict, years) -> None:
    """Output override spreadsheets for all specified utilities and years.

    These manual override files will be output to a folder called "overrides" in the
    output directory.

    Args:
        pudl_out (PudlTabl): the pudl_out object generated in a notebook and passed in.
        rmi_out (Output): the rmi_out object generated in a notebook and passed in.
        util_dict (dict): A dictionary with keys that are the names of utility
            parent companies and values that are lists of subsidiary utility_id_eia
            values. EIA values are used instead of PUDL in this case because PUDL values
            are subject to change.
        years (list): A list of the years you'd like to add to the override sheets.

    """
    # Generate full input tables
    inputs_dict = _generate_input_dfs(pudl_out, rmi_out)

    # Make sure overrides dir exists
    if not os.path.isdir(pudl_rmi.OUTPUTS_DIR / "overrides"):
        os.mkdir(pudl_rmi.OUTPUTS_DIR / "overrides")

    # For each utility, make an override sheet with the correct input table slices
    for util_name, util_id_eia_list in util_dict.items():
        logger.info(f"Developing outputs for {util_name}")
        util_year_subset_dict = _get_util_year_subsets(
            inputs_dict, util_id_eia_list, years
        )
        _output_override_spreadsheet(util_year_subset_dict, util_name)


# --------------------------------------------------------------------------------------
# Upload Changes to Training Data
# --------------------------------------------------------------------------------------


def _check_id_consistency(id_type, df, actual_ids, error_message) -> None:
    """Check for rogue FERC or EIA ids that don't exist.

    Args:
        id_type (str): A string, either 'ferc' or 'eia' indicating whether to check
            ferc or eia id columns.
        actual_ids (list): A list of the ferc or eia ids that are valid and come from
            either the ppl or official ferc-eia record linkage.
        error_message (str): A short string to indicate the type of error you're
            checking for. This could be looking for values that aren't in the official
            list or values that are already in the training data.

    """
    logger.debug(f"Checking {id_type} record id consistency for {error_message}")

    if id_type not in ["ferc", "eia"]:
        raise ValueError("id_type must be either 'ferc' or 'eia'.")
    if id_type == "eia":
        id_col = "record_id_eia_override_1"
    elif id_type == "ferc":
        id_col = "record_id_ferc1"

    assert (
        len(bad_ids := df[~df[id_col].isin(actual_ids)][id_col].to_list()) == 0
    ), f"{id_col} {error_message}: {bad_ids}"


def validate_override_fixes(
    validated_connections,
    utils_eia860,
    ppl,
    ferc1_eia,
    training_data,
    expect_override_overrides=False,
) -> pd.DataFrame:
    """Process the verified and/or fixed matches and look for human error.

    Args:
        validated_connections (pd.DataFrame): A dataframe in the add_to_training
            directory that is ready to be added to be validated and subsumed into the
            training data.
        utils_eia860 (pd.DataFrame): A dataframe resulting from the
            pudl_out.utils_eia860() function.
        ferc1_eia (pd.DataFrame): The current FERC-EIA table
        expect_override_overrides (boolean): Whether you expect the tables to have
            overridden matches already in the training data.
    Raises:
        AssertionError: If there are EIA override id records that aren't in the original
            FERC-EIA connection.
        AssertionError: If there are FERC record ids that aren't in the original
            FERC-EIA connection.
        AssertionError: If there are EIA override ids that are duplicated throughout the
            override document.
        AssertionError: If the utility id in the EIA override id doesn't match the pudl
            id cooresponding with the FERC record.
        AssertionError: If there are EIA override id records that don't correspond to
            the correct report year.
        AssertionError: If you didn't expect to override overrides but the new training
            data implies an override to the existing training data.
    Returns:
        pd.DataFrame: The validated FERC-EIA dataframe you're trying to add to the
            training data.

    """
    logger.info("Validating overrides")
    # When there are NA values in the verified column in the excel doc, it seems that
    # the TRUE values become 1.0 and the column becomes a type float. Let's replace
    # those here and make it a boolean.
    validated_connections["verified"] = validated_connections["verified"].replace(
        {1: True, np.nan: False}
    )
    # Make sure the verified column doesn't contain non-boolean outliers. This will fail
    # if there are bad values.
    validated_connections.astype({"verified": pd.BooleanDtype()})

    # From validated records, get only records with an override
    only_overrides = (
        validated_connections[validated_connections["verified"]]
        .dropna(subset=["record_id_eia_override_1"])
        .reset_index()
        .copy()
    )

    # Make sure that the override EIA ids actually match those in the original FERC-EIA
    # record linkage.
    actual_eia_ids = ppl.record_id_eia.unique()
    _check_id_consistency(
        "eia", only_overrides, actual_eia_ids, "values that don't exist"
    )

    # It's unlikely that this changed, but check FERC id too just in case!
    actual_ferc_ids = ferc1_eia.record_id_ferc1.unique()
    _check_id_consistency(
        "ferc", only_overrides, actual_ferc_ids, "values that don't exist"
    )

    # Make sure there are no duplicate EIA id overrides
    logger.debug("Checking for duplicate override ids")
    assert (
        len(
            override_dups := only_overrides[
                only_overrides["record_id_eia_override_1"].duplicated(keep=False)
            ]
        )
        == 0
    ), f"Found record_id_eia_override_1 duplicates: \
    {override_dups.record_id_eia_override_1.unique()}"

    # Make sure the EIA utility id from the override matches the PUDL id from the FERC
    # record. Start by mapping utility_id_eia from PPL onto each
    # record_id_eia_override_1.
    logger.debug("Checking for mismatched utility ids")
    only_overrides = only_overrides.merge(
        ppl[["record_id_eia", "utility_id_eia"]].drop_duplicates(),
        left_on="record_id_eia_override_1",
        right_on="record_id_eia",
        how="left",
        suffixes=("", "_ppl"),
    )
    # Now merge the utility_id_pudl from EIA in so that you can compare it with the
    # utility_id_pudl from FERC that's already in the overrides
    only_overrides = only_overrides.merge(
        utils_eia860[["utility_id_eia", "utility_id_pudl"]].drop_duplicates(),
        left_on="utility_id_eia_ppl",
        right_on="utility_id_eia",
        how="left",
        suffixes=("", "_utils"),
    )
    # Now we can actually compare the two columns
    if (
        len(
            bad_utils := only_overrides["utility_id_pudl"].compare(
                only_overrides["utility_id_pudl_utils"]
            )
        )
        > 0
    ):
        raise AssertionError(f"Found mismatched utilities: {bad_utils}")

    # # Compare utility_id_eia from record_id_eia_override_1 to the pudl_id from
    # # utils_eia860 and make sure that matches the pudl_id from ferc in the overrides
    # utils = (
    #     utils_eia860[["utility_id_eia", "utility_id_pudl"]]
    #     .drop_duplicates()
    #     .set_index("utility_id_eia")
    # )
    # only_overrides["utility_id_eia_override"]
    #
    # # To do this, we'll make a dictionary mapping PUDL id to a list of EIA ids
    # logger.debug("Checking for mismatched utility ids")
    # eia_id_list_series = utils_eia860.groupby("utility_id_pudl").apply(
    #     lambda x: x.utility_id_eia.unique().tolist()
    # )
    # eia_id_dict = dict(zip(eia_id_list_series.index, eia_id_list_series))
    # # Map utility_id_eia from PPL onto each record_id_eia_override_1.
    # record_id_util_id_dict = dict(zip(ppl["record_id_eia"], ppl["utility_id_eia"]))
    # only_overrides[
    #     "utility_id_eia_override"
    # ] = only_overrides.record_id_eia_override_1.map(record_id_util_id_dict)
    #
    # assert (
    #     len(
    #         mismatched_utilities := only_overrides[
    #             ~only_overrides.apply(
    #                 lambda x: x.utility_id_eia_override
    #                 in eia_id_dict[x.utility_id_pudl],
    #                 axis=1,
    #             )
    #         ]
    #     )
    #     == 0
    # ), f"Found mismatched utilities: \
    # {mismatched_utilities.record_id_eia_override_1}"

    # Make sure the year in the EIA id overrides match the year in the report_year
    # column.
    logger.debug("Checking that year in override id matches report year")
    only_overrides = only_overrides.merge(
        ppl[["record_id_eia", "report_year"]].drop_duplicates(),
        left_on="record_id_eia_override_1",
        right_on="record_id_eia",
        how="left",
        suffixes=("", "_ppl"),
    )
    if (
        len(
            bad_eia_year := only_overrides["report_year"].compare(
                only_overrides["report_year_ppl"]
            )
        )
        > 0
    ):
        raise AssertionError(
            f"Found record_id_eia_override_1 values that don't correspond to the right \
            report year:\
            {[only_overrides.iloc[x].record_id_eia_override_1 for x in bad_eia_year.index]}"
        )

    # If you don't expect to override values that have already been overridden, make
    # sure the ids you fixed aren't already in the training data.
    if not expect_override_overrides:
        existing_training_eia_ids = training_data.record_id_eia.dropna().unique()
        _check_id_consistency(
            "eia", only_overrides, existing_training_eia_ids, "already in training"
        )
        existing_training_ferc_ids = training_data.record_id_ferc1.dropna().unique()
        _check_id_consistency(
            "ferc", only_overrides, existing_training_ferc_ids, "already in training"
        )

    # Only return the results that have been verified
    verified_connections = validated_connections[
        validated_connections["verified"]
    ].copy()

    return verified_connections


def _add_to_training(new_overrides) -> None:
    """Add the new overrides to the old override sheet."""
    logger.info("Combining all new overrides with existing training data")
    current_training = pd.read_csv(pudl_rmi.TRAIN_FERC1_EIA_CSV)
    new_training = (
        new_overrides[
            ["record_id_eia", "record_id_ferc1", "signature_1", "signature_2", "notes"]
        ]
        .copy()
        .drop_duplicates(subset=["record_id_eia", "record_id_ferc1"])
        # .set_index(["record_id_eia", "record_id_ferc1"])
    )
    logger.debug(f"Found {len(new_training)} new overrides")
    # Combine new and old training data
    training_data_out = current_training.append(new_training).drop_duplicates(
        subset=["record_id_eia", "record_id_ferc1"]
    )
    # Output combined training data
    training_data_out.to_csv(pudl_rmi.TRAIN_FERC1_EIA_CSV, index=False)


def _add_to_null_overrides(null_matches) -> None:
    """Take record_id_ferc1 values verified to have no EIA match and add them to csv."""
    logger.info("Adding record_id_ferc1 values with no EIA match to null_overrides csv")
    # Get new null matches
    new_null_matches = null_matches[["record_id_ferc1"]].copy()
    logger.debug(f"Found {len(new_null_matches)} new null matches")
    # Get current null matches
    current_null_matches = pd.read_csv(pudl_rmi.NULL_FERC1_EIA_CSV)
    # Combine new and current record_id_ferc1 values that have no EIA match
    out_null_matches = current_null_matches.append(new_null_matches).drop_duplicates()
    # Write the combined values out to the same location as before
    out_null_matches.to_csv(pudl_rmi.NULL_FERC1_EIA_CSV, index=False)


def validate_and_add_to_training(
    pudl_out, rmi_out, expect_override_overrides=False
) -> None:
    """Validate, combine, and add overrides to the training data.

    Validating and combinging the records so you only have to loop through the files
    once. Runs the validate_override_fixes() function and add_to_training.

    Args:
        pudl_out (PudlTabl): the pudl_out object generated in a notebook and passed in.
        rmi_out (Output): the rmi_out object generated in a notebook and passed in.
        expect_override_overrides (bool): This value is explicitly assigned at the top
            of the notebook.

    Returns:
        pandas.DataFrame: A DataFrame with all of the new overrides combined.
    """
    # Define all relevant tables
    ferc1_eia_df = rmi_out.ferc1_to_eia()
    ppl_df = rmi_out.plant_parts_eia().reset_index()
    utils_df = pudl_out.utils_eia860()
    training_df = pd.read_csv(pudl_rmi.TRAIN_FERC1_EIA_CSV)
    path_to_new_training = pudl_rmi.INPUTS_DIR / "add_to_training"
    override_cols = [
        "record_id_eia",
        "record_id_ferc1",
        "signature_1",
        "signature_2",
        "notes",
    ]
    null_match_cols = ["record_id_ferc1"]
    all_overrides_list = []
    all_null_matches_list = []

    # Loop through all the files, validate, and combine them.
    all_files = os.listdir(path_to_new_training)
    excel_files = [file for file in all_files if file.endswith(".xlsx")]

    for file in excel_files:
        logger.info(f"Processing fixes in {file}")
        file_df = (
            pd.read_excel(path_to_new_training / file)
            .pipe(
                validate_override_fixes,
                utils_df,
                ppl_df,
                ferc1_eia_df,
                training_df,
                expect_override_overrides=expect_override_overrides,
            )
            .rename(
                columns={
                    "record_id_eia": "record_id_eia_old",
                    "record_id_eia_override_1": "record_id_eia",
                }
            )
        )
        # Get just the overrides and combine them to full list of overrides
        only_overrides = file_df[file_df["record_id_eia"].notna()][override_cols].copy()
        all_overrides_list.append(only_overrides)
        # Get just the null matches and combine them to full list of overrides
        only_null_matches = file_df[file_df["record_id_eia"].isna()][
            null_match_cols
        ].copy()
        all_null_matches_list.append(only_null_matches)

    # Combine all training data and null matches
    all_overrides_df = pd.concat(all_overrides_list)
    all_null_matches_df = pd.concat(all_null_matches_list)

    # Add the records to the training data and null overrides
    _add_to_training(all_overrides_df)
    _add_to_null_overrides(all_null_matches_df)
