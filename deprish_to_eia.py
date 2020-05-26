"""Connect the master unit list with depreciation records."""

import argparse
import logging
import pathlib
import sys

import coloredlogs
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process
from openpyxl import load_workbook

import plant_part_agg_eia
import pudl

logger = logging.getLogger(__name__)
log_format = '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s'
coloredlogs.install(fmt=log_format, level='INFO', logger=logger)

STRINGS_TO_CLEAN = {
    "combined cycle": "CC",
    "combustion turbine": "CT"
}

INT_IDS = ['utility_id_ferc', 'utility_id_pudl',
           'plant_id_pudl', 'report_year']


###############################################################################
# Prep the inputs
###############################################################################


def prep_int_ids(int_ids):
    """Prep dictionary of column names (key) with nullable int dype (value)."""
    # prep dtype arg for integer columns for read_excel
    dtypes_dict = {}
    for i in int_ids:
        dtypes_dict[i] = pd.Int64Dtype()
    return dtypes_dict


def grab_sheet_from_excel(file_path, sheet_name, dtypes_dict):
    """Grab a sheet from an excel file."""
    logger.info(f'grabbing sheet from: {file_path}')
    deprish_df = pd.read_excel(
        file_path, skiprows=1, sheet_name=sheet_name, dtype=dtypes_dict)
    return deprish_df


def clean_strings(df, key, strings_to_clean):
    """Clean the strings in a given column."""
    df = pudl.helpers.strip_lower(df, [key])
    df.loc[:, key] = df.loc[:, key].replace(strings_to_clean, regex=True)
    return df


def prep_deprish(file_path_deprish, sheet_name_deprish, plant_parts_df, key1):
    """Prep the depreciation dataframe."""
    logger.info(f"Reading the depreciation data from {file_path_deprish}")
    deprish_df = (pd.read_excel(
        file_path_deprish, skiprows=0, sheet_name=sheet_name_deprish,
        dtypes=prep_int_ids(INT_IDS))
        .astype({'report_date': 'datetime64[ns]',
                 'plant_id_pudl': pd.Int64Dtype()}))
    ids = ['plant_id_pudl', 'report_year']
    deprish_ids = (pd.merge(
        (deprish_df
         .astype({'report_date': 'datetime64[ns]'})
         .assign(report_year=deprish_df.report_date.dt.year)
         .dropna(subset=ids)
         ),
        plant_parts_df[ids].drop_duplicates(),
        how='outer', indicator=True, on=ids))
    deprish_ids = (deprish_ids.loc[deprish_ids._merge == 'both']
                   .drop_duplicates(subset=['plant_name'])
                   .dropna(subset=['plant_name'])
                   )
    return clean_strings(deprish_ids, key1, STRINGS_TO_CLEAN)


def prep_master_parts_eia(plant_parts_df, key2):
    """Prepare the EIA master plant parts."""
    plant_parts_df = (clean_strings(plant_parts_df, key2, STRINGS_TO_CLEAN)
                      .reset_index())
    return plant_parts_df


###############################################################################
# Fuzzy Matching
###############################################################################


def get_plant_year_list(plant_name, df1, df2, key1, key2):
    """
    Get the possible key matches from df2 a plant_id_pudl and report_year.

    This selects for the plant id and year for each of the df1 records to
    match. This is for use within `get_fuzzy_matches`.
    """
    logger.debug(plant_name)
    plant_id_pudl = df1.loc[df1.plant_name ==
                            plant_name, 'plant_id_pudl'].values[0]
    report_year = df1.loc[df1.plant_name ==
                          plant_name, 'report_year'].values[0]
    names = df2.loc[(df2['plant_id_pudl'] == plant_id_pudl)
                    & (df2.report_year == report_year)][key2].to_list()
    return names


def get_fuzzy_matches(df1, df2, key1, key2, threshold=75):
    """
    Get fuzzy matches on df1 using token_sort_ratio and extractOne.

    Args:
        df1 (pandas.Dataframe): is the left table to join
        df2 (pandas.Dataframe): is the right table to join
        key1 (str): is the key column of the left table
        key2 (str): is the key column of the right table
        threshold (int): is how close the matches should be
            to return a match, based on Levenshtein distance
    Returns:
        pandas.DataFrame
    """
    m = df1[key1].apply(
        lambda x: process.extractOne(
            x, get_plant_year_list(x, df1, df2, key1, key2),
            scorer=fuzz.token_sort_ratio)
    )

    df1 = df1.assign(
        matches=m,
        plant_name_match=m.apply(
            lambda x: x[0] if x[1] >= threshold else np.NaN)
    )

    logger.info(
        f"Matches: {len(df1[df1.plant_name_match.notnull()])/len(df1):.02%}")
    return df1


def match_merge(df1, df2, key1, key2):
    """Generate fuzzy matches and merge relevant colums from eia."""
    cols1 = [
        'utility_id_ferc', 'utility_id_pudl', 'utility_name_ferc1', 'state',
        'plant_id_pudl', 'plant_name', 'report_year', 'plant_name_match']
    cols2 = [
        'record_id_eia', 'plant_name_new', 'plant_name_eia', 'plant_id_eia',
        'report_year', 'plant_part', 'generator_id', 'unit_id_pudl',
        'prime_mover_code', 'energy_source_code_1',
        'technology_description', 'ferc_acct_name', 'utility_id_eia',
        'true_gran', 'appro_part_label', 'appro_record_id_eia',
    ]
    logger.info("Merging fuzzy matches.")
    match_merge = pd.merge(
        get_fuzzy_matches(df1, df2, key1, key2, threshold=75)[cols1],
        df2.reset_index().drop_duplicates(
            subset=['report_year', 'plant_name_new'])[cols2],
        left_on=['report_year', 'plant_name_match'],
        right_on=['report_year', key2], how='left')
    logger.info(f"Mathing resulted in {len(match_merge)} connections.")
    return match_merge


def match_deprish_eia(file_path_mul, file_path_deprish, sheet_name_deprish):
    """Match."""
    plant_parts_df = plant_part_agg_eia.get_master_unit_list_eia(file_path_mul)

    key1 = 'plant_name'
    key2 = 'plant_name_new'
    df1 = prep_deprish(file_path_deprish, sheet_name_deprish,
                       plant_parts_df, key1)
    df2 = prep_master_parts_eia(plant_parts_df, key2)
    return match_merge(df1, df2, key1, key2)

###############################################################################
# EXPORT
###############################################################################


def generate_depreciation_matches(file_path_mul,
                                  file_path_deprish,
                                  sheet_name_deprish,
                                  sheet_name_output):
    """
    Generate the matched names and save to execl.

    Args:
         file_path_mul (path-like): path to the master unit list/.
         file_path_deprish (path-like): path to the excel workbook which
            contains depreciation data.
         sheet_name_deprish (string): name of the excel tab which contains the
            plant names from the depreciation data.
         sheet_name_output (string): name of the excel tab which the matches
            will be output.
    """
    if not file_path_deprish.is_file():
        raise AssertionError(
            f"File does not exist: {file_path_deprish}"
            "Depretiation file must exist."
        )

    deprish_match = match_deprish_eia(
        file_path_mul, file_path_deprish,
        sheet_name_deprish=sheet_name_deprish)

    save_to_workbook(deprish_match, file_path=file_path_deprish,
                     sheet_name=sheet_name_output)
    return deprish_match


def save_to_workbook(df, file_path, sheet_name):
    """
    Save a dataframe to an existing workbook.

    Args:
        df (pandas.DataFrame): the table you want to export.
        file_path (path-like): the location of the excel workbook.
        sheet_name (string): the name of your new sheet. If this sheet exists
            in your workbook, the new sheet will be sheet_name_1.
    """
    logger.info(f"Saving dataframe to {file_path}")
    if not file_path.exists():
        raise AssertionError(f'file path {str(file_path)} does not exist')
    workbook1 = load_workbook(file_path)
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    writer.book = workbook1
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()
    writer.close()

###############################################################################
# EXPORT
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
        default=pathlib.Path('fake path to master unit list'),
        type=str,
        help='path to the master unit list.')
    parser.add_argument(
        '--sheet_name_deprish',
        default='Depreciation Studies Raw',
        type=str,
        help="""name of the excel tab which contains the plant names from the
        depreciation data.""")
    parser.add_argument(
        '--sheet_name_output',
        default='EIA to depreciation matches',
        type=str,
        help="""name of the excel tab which the matches will be output.""")
    arguments = parser.parse_args(argv[1:])
    return arguments


def main():
    """Match deprecation and master unit list records. Save to excel."""
    args = parse_command_line(sys.argv)

    _ = generate_depreciation_matches(
        file_path_mul=pathlib.Path(args.file_path_mul),
        file_path_deprish=pathlib.Path(args.file_path_deprish),
        sheet_name_deprish=args.sheet_name_deprish,
        sheet_name_output=args.sheet_name_output)


if __name__ == '__main__':
    sys.exit(main())
