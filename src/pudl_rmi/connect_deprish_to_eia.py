"""
Connect the master unit list with depreciation records from PUCs and FERC1.

This module connects the records from the depreciation data to their
appropirate ids in the EIA master unit list. The master unit list is generated
via `make_plant_parts_eia.py`; see the documenation there for additional
details about the master unit list. The depreciation data is annual
depreciation studies from PUC and FERC1 data that have been compiled into an
excel spreadsheet. The master unit list is a compilation of various slices of
EIA records.
"""

import argparse
import logging
import pathlib
import sys

import pandas as pd
from fuzzywuzzy import fuzz, process
from openpyxl import load_workbook
from xlrd import XLRDError

import pudl_rmi.make_plant_parts_eia as make_plant_parts_eia
import pudl_rmi.deprish as deprish
import pudl

logger = logging.getLogger(__name__)

STRINGS_TO_CLEAN = {
    "combined cycle": ["CC"],
    "combustion turbine": ["CT"],
}

RESTRICT_MATCH_COLS = ['plant_id_pudl', 'utility_id_pudl', 'report_year']

MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]

MUL_RENAME = {
    'record_id_eia': 'record_id_eia_name_match',
    'appro_record_id_eia': 'record_id_eia_fuzzy',
    'plant_part': 'plant_part_name_match',
    'appro_part_label': 'plant_part',
    'true_gran': 'true_gran_name_match'
}
"""dict: to rename the columns from the master unit list. Because we want
to fuzzy match with all possible MUL records names but only use the true
granualries."""

DEPRISH_COLS = [
    'utility_id_ferc1', 'utility_id_pudl', 'utility_name_ferc1', 'state',
    'plant_id_pudl', 'plant_part_name', 'report_year']

###############################################################################
# Prep the inputs
###############################################################################


def prep_deprish(file_path_deprish, plant_parts_df,
                 sheet_name_deprish,  key_deprish):
    """
    Prep the depreciation dataframe for use in match_merge.

    Grab only the records which can be associated with EIA master unit list
    records by plant_id_pudl, and do some light cleaning.
    """
    deprish_df = (
        deprish.Transformer(
            deprish.Extractor(file_path=file_path_deprish,
                              sheet_name=sheet_name_deprish).execute())
        .execute()
        .dropna(subset=RESTRICT_MATCH_COLS)
    )

    deprish_df.loc[:, key_deprish] = pudl.helpers.cleanstrings_series(
        deprish_df.loc[:, key_deprish], str_map=STRINGS_TO_CLEAN)

    # because we are comparing to the EIA-based master unit list, we want to
    # only include records which are associated with plant_id_pudl's that are
    # in the master unit list.
    deprish_ids = (pd.merge(
        deprish_df,
        plant_parts_df[RESTRICT_MATCH_COLS].drop_duplicates().dropna(),
        how='outer', indicator=True, on=RESTRICT_MATCH_COLS)
    )
    deprish_ids = (
        deprish_ids.loc[deprish_ids._merge == 'both']
        .drop_duplicates(subset=['plant_part_name', 'report_date'])
        # there are some records in the depreciation df that have no
        # names.... so they've got to go
        .dropna(subset=['plant_part_name'])
        .pipe(pudl.helpers.convert_cols_dtypes, 'depreciation')
    )
    return deprish_ids


def prep_master_parts_eia(plant_parts_df, deprish_df, key_mul):
    """Prepare the EIA master plant parts."""
    # restrict the possible matches to only those that match on the
    # RESTRICT_MATCH_COLS
    options_index = (deprish_df[RESTRICT_MATCH_COLS].drop_duplicates()
                     .set_index(RESTRICT_MATCH_COLS).index)
    plant_parts_df = plant_parts_df.set_index(
        RESTRICT_MATCH_COLS).loc[options_index].reset_index()

    plant_parts_df.loc[:, key_mul] = pudl.helpers.cleanstrings_series(
        plant_parts_df[key_mul], str_map=STRINGS_TO_CLEAN)
    return plant_parts_df


###############################################################################
# Fuzzy Matching
###############################################################################


def get_plant_year_util_list(plant_name, deprish_df, mul_df, key_mul):
    """
    Get the possible key matches from df2 a plant_id_pudl and report_year.

    This selects for the plant id and year for each of the df1 records to
    match. This is for use within `get_fuzzy_matches`.
    """
    logger.debug(plant_name)
    options_index = (
        deprish_df.loc[deprish_df.plant_part_name ==
                       plant_name, RESTRICT_MATCH_COLS]
        .drop_duplicates().set_index(RESTRICT_MATCH_COLS).index)

    # get the set of possible names
    names = (mul_df.set_index(RESTRICT_MATCH_COLS)
             .loc[options_index, key_mul].to_list())
    return names


def get_fuzzy_matches(deprish_df, mul_df, key_deprish, key_mul, threshold=75):
    """
    Get fuzzy matches on df1 using token_sort_ratio and extractOne.

    Using fuzzywuzzy's fuzzy string matching, this function matches each value
    in the key1 column with the best matched key1.

    Args:
        deprish_df (pandas.Dataframe): is the left table to join
        mul_df (pandas.Dataframe): is the right table to join
        key_deprish (str): is the key column of the left table
        key_mul (str): is the key column of the right table
        threshold (int): is how close the matches should be to return a match,
            based on Levenshtein distance. Range between 0 and 100.

    Returns:
        pandas.DataFrame
    """
    # get the best match for each valye of the key1 column
    match_tuple_series = deprish_df[key_deprish].apply(
        lambda x: process.extractOne(
            x, get_plant_year_util_list(x, deprish_df, mul_df, key_mul),
            scorer=fuzz.token_sort_ratio)
    )
    # process.extractOne returns a tuple with the matched name and the
    # match's score, so match_tuple_series contains tuples of the matching
    # plant name and the score. The plant_name_match assign only assigns the
    # match if the score is greater than or equal to the threshold.
    deprish_df = deprish_df.assign(
        matches=match_tuple_series,
        plant_name_match=match_tuple_series.apply(
            lambda x: x[0] if x[1] >= threshold else None)
    )

    matches_perct = (len(deprish_df[deprish_df.plant_name_match.notnull()])
                     / len(deprish_df))
    logger.info(f"Matches: {matches_perct:.02%}")
    return deprish_df


def match_merge(deprish_df, mul_df, key_deprish, key_mul):
    """Generate fuzzy matches and merge relevant colums from eia."""
    logger.info("Merging fuzzy matches.")
    # we are going to match with all of the names from the
    # master unit list, but then use the "true granularity"
    match_merge_df = (pd.merge(
        get_fuzzy_matches(
            deprish_df, mul_df,
            key_deprish=key_deprish, key_mul=key_mul,
            threshold=75)[DEPRISH_COLS + ['plant_name_match']],
        mul_df.reset_index().drop_duplicates(
            subset=['report_year', 'plant_name_new'])[MUL_COLS],
        left_on=['report_year', 'utility_id_pudl', 'plant_name_match'],
        right_on=['report_year', 'utility_id_pudl', key_mul], how='left')
        # rename the ids so that we have the "true granularity"
        # Every MUL record has identifying columns for it's true granualry,
        # even when the true granularity is the same record, so we can use the
        # true gran columns across the board.
        .rename(columns=MUL_RENAME)
    )
    logger.info(f"Matching resulted in {len(match_merge_df)} connections.")
    return match_merge_df


def add_overrides(deprish_match, file_path_deprish, sheet_name_output):
    """
    Add the overrides into the matched depreciation records.

    Args:
        deprish_match (pandas.DataFrame):
        file_path_deprish (os.PathLike): path to the excel workbook which
           contains depreciation data.
       sheet_name_output (string): name of the excel tab which the matches
          will be output.

    Returns:
        pandas.DataFrame: augmented version of deprish_match with overrides.
    """
    try:
        overrides_df = (
            pd.read_excel(
                file_path_deprish, skiprows=0, sheet_name=sheet_name_output,)
        )
        overrides_df = (
            overrides_df[overrides_df.filter(like='record_id_eia_override')
                         .notnull().any(axis='columns')]
            [DEPRISH_COLS +
             list(overrides_df.filter(like='record_id_eia_override').columns)])
        logger.info(f"Adding overrides from {sheet_name_output}.")
        # concat, sort so the True overrides are at the top and drop dupes
        deprish_match_full = (
            pd.merge(deprish_match, overrides_df, on=DEPRISH_COLS, how='left')
            .assign(record_id_eia=lambda x:
                    x.record_id_eia_override.fillna(x.record_id_eia_fuzzy))
        )
    except XLRDError:
        logger.info(f"No sheet {sheet_name_output}, so no overrides added.")
        deprish_match_full = deprish_match
    return deprish_match_full


def match_deprish_eia(file_path_mul, file_path_deprish,
                      sheet_name_deprish, sheet_name_output):
    """Prepare the depreciation and master unit list and match on name cols."""
    key_deprish = 'plant_part_name'
    key_mul = 'plant_name_new'
    logger.info('Grab or generate master unit list.')
    plant_parts_df = (
        make_plant_parts_eia.get_master_unit_list_eia(file_path_mul)
        .reset_index()
    )
    deprish_df = prep_deprish(
        file_path_deprish,
        plant_parts_df,
        sheet_name_deprish=sheet_name_deprish,
        key_deprish=key_deprish
    )
    mul_df = prep_master_parts_eia(plant_parts_df, deprish_df, key_mul=key_mul)
    deprish_match = (
        match_merge(deprish_df, mul_df,
                    key_deprish=key_deprish, key_mul=key_mul)
        .pipe(add_overrides, file_path_deprish=file_path_deprish,
              sheet_name_output=sheet_name_output)
        .pipe(make_plant_parts_eia.reassign_id_ownership_dupes)
        .pipe(pudl.helpers.organize_cols,
              # we want to pull the used columns to the front, but there is
              # some overlap in columns from these two datasets. And we have
              # renamed some of the columns from the master unit list.
              list(set(DEPRISH_COLS + [MUL_RENAME.get(c, c)
                                       for c in MUL_COLS])))
    )

    first_cols = [
        'plant_part_name', 'plant_name_match', 'record_id_eia',
        'record_id_eia_fuzzy',
        'record_id_eia_override', 'record_id_eia_override2',
        'record_id_eia_override3', 'record_id_eia_override4',
        'record_id_eia_override5', 'record_id_eia_override6',
        'record_id_eia_override7', 'record_id_eia_override8',
        'record_id_eia_override9', 'record_id_eia_override10',
    ]
    deprish_match = deprish_match.loc[
        :, first_cols + [x for x in deprish_match.columns
                         if x not in first_cols]]

    possible_matches_mul = (
        pd.merge(
            plant_parts_df.dropna(subset=RESTRICT_MATCH_COLS),
            deprish_df[RESTRICT_MATCH_COLS].drop_duplicates(),
            on=RESTRICT_MATCH_COLS)
        .pipe(pudl.helpers.organize_cols, MUL_COLS)
    )
    return deprish_match, possible_matches_mul

###############################################################################
# EXPORT
###############################################################################


def generate_depreciation_matches(file_path_mul,
                                  file_path_deprish,
                                  sheet_name_deprish,
                                  sheet_name_output,
                                  save_to_xls=True,
                                  ):
    """
    Generate the matched names and save to excel.

    This method generates a link between depreciation records and the master
    unit list. It generates all of the options that could have been matched
    from the master unit list; this will help with generating mannual
    overrides. It then saves these outputs into the same spreadsheet that the
    depreciation records were pulled from.

    Args:
         file_path_mul (pathlib.Path): path to the master unit list.
         file_path_deprish (os.PathLike): path to the excel workbook which
            contains depreciation data.
         sheet_name_deprish (str): name of the excel tab which contains the
            plant names from the depreciation data.
         sheet_name_output (string): name of the excel tab which the matches
            will be output.

    Returns:
        pandas.DataFrame : dataframe including matched names from depreciation
            data to names in the master unit list, including appropirate id's
            from the master unit list.
    """
    if not file_path_deprish.is_file():
        raise FileNotFoundError(
            f"File does not exist: {file_path_deprish}"
            "Depretiation file must exist."
        )
    deprish_match_df, possible_matches_mul_df = match_deprish_eia(
        file_path_mul, file_path_deprish,
        sheet_name_deprish=sheet_name_deprish,
        sheet_name_output=sheet_name_output
    )
    if save_to_xls:
        sheets_df_dict = {
            sheet_name_output: deprish_match_df,
            "Subset of Master Unit List": possible_matches_mul_df}
        save_to_workbook(file_path=file_path_deprish,
                         sheets_df_dict=sheets_df_dict)
    return deprish_match_df


def save_to_workbook(file_path, sheets_df_dict):
    """
    Save dataframes to sheets in an existing workbook.

    This method enables us to save multiple dataframes into differnt tabs in
    the same excel workbook. If those tabs already exist, the default process
    is to make save a new tab with a suffix, so we remove the tab if it exists
    before saving.

    Args:
        file_path (path-like): the location of the excel workbook.
        sheets_df_dict (dict): dictionary of the names of sheets (keys) of
            where their corresponding dataframes (values) should end up in the
            workbook.
    """
    logger.info(f"Saving dataframe to {file_path}")
    if not file_path.exists():
        raise AssertionError(f'file path {file_path} does not exist')
    workbook1 = load_workbook(file_path)
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    writer.book = workbook1
    for sheet_name, df in sheets_df_dict.items():
        if sheet_name in workbook1.sheetnames:
            logger.info(f"Removing {sheet_name} from {file_path}")
            workbook1.remove(workbook1[sheet_name])
            if sheet_name in workbook1.sheetnames:
                raise AssertionError(f"{sheet_name} was not removed")
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()
    writer.close()

###############################################################################
# Command line script
###############################################################################


def parse_command_line(argv):
    """
    Parse script command line arguments. See the -h option.

    Args:
        argv (list): command line arguments including caller file name.

    Returns:
        dict: A dictionary mapping command line arguments to their values.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'file_path_deprish',
        type=str,
        help='path to the excel workbook which contains depreciation data.')
    parser.add_argument(
        '--file_path_mul',
        default=pathlib.Path('master_unit_list.pkl.gz'),
        type=str,
        help='path to the master unit list. Default: master_unit_list.csv.gz')
    parser.add_argument(
        '--sheet_name_deprish',
        default='Depreciation Studies Raw',
        type=str,
        help="""name of the excel tab which contains the plant names from the
        depreciation data. Default: Depreciation Studies Raw""")
    parser.add_argument(
        '--sheet_name_output',
        default='EIA to depreciation matches',
        type=str,
        help="""name of the excel tab which the matches will be output.
        Default: EIA to depreciation matches""")
    arguments = parser.parse_args(argv[1:])
    return arguments


def main():
    """Match depreciation and master unit list records. Save to excel."""
    args = parse_command_line(sys.argv)

    _ = generate_depreciation_matches(
        file_path_mul=pathlib.Path(args.file_path_mul),
        file_path_deprish=pathlib.Path(args.file_path_deprish),
        sheet_name_deprish=args.sheet_name_deprish,
        sheet_name_output=args.sheet_name_output)


if __name__ == '__main__':
    sys.exit(main())
