"""
EIA Plant-parts list is actively being moved into PUDL.

Here are some dangly bits that we are actively using to read the plant-parts
list from PUDL into the other realms of this repo.
"""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]

DTYPES_MUL = {
    "plant_id_eia": "int64",
    "report_date": "datetime64[ns]",
    "plant_part": "object",
    "generator_id": "object",
    "unit_id_pudl": "object",
    "prime_mover_code": "object",
    "energy_source_code_1": "object",
    "technology_description": "object",
    "ferc_acct_name": "object",
    "utility_id_eia": "object",
    "true_gran": "bool",
    "appro_part_label": "object",
    "appro_record_id_eia": "object",
    "capacity_factor": "float64",
    "capacity_mw": "float64",
    "fraction_owned": "float64",
    "fuel_cost_per_mmbtu": "float64",
    "fuel_cost_per_mwh": "float64",
    "heat_rate_mmbtu_mwh": "float64",
    "installation_year": "Int64",
    "net_generation_mwh": "float64",
    "ownership": "category",
    "plant_id_pudl": "Int64",
    "plant_name_eia": "string",
    "total_fuel_cost": "float64",
    "total_mmbtu": "float64",
    "utility_id_pudl": "Int64",
    "utility_name_eia": "string",
    "report_year": "int64",
    "plant_id_report_year": "object",
    "plant_name_new": "string"
}

FIRST_COLS = ['plant_id_eia', 'report_date', 'plant_part', 'generator_id',
              'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
              'technology_description', 'ferc_acct_name',
              'utility_id_eia', 'true_gran', 'appro_part_label']


############################
# Keep in rmi repo functions
############################


def reassign_id_ownership_dupes(plant_parts_df):
    """
    Reassign the record_id for the records that are labeled ownership_dupe.

    This function is used after the EIA plant-parts table is created.

    Args:
        plant_parts_df (pandas.DataFrame): master unit list. Result of
            ``generate_master_unit_list()`` or ``get_master_unit_list_eia()``.
            Must have boolean column ``ownership_dupe`` and string column or
            index of ``record_id_eia``.

    """
    # the record_id_eia's need to be a column to mess with it and record_id_eia
    # is typically the index of plant_parts_df, so we are going to reset index
    # if record_id_eia is the index
    og_index = False
    if plant_parts_df.index.name == "record_id_eia":
        plant_parts_df = plant_parts_df.reset_index()
        og_index = True

    plant_parts_df = plant_parts_df.assign(record_id_eia=lambda x: np.where(
        x.ownership_dupe,
        x.record_id_eia.str.replace("owned", "total"),
        x.record_id_eia))
    # then we reset the index so we return the dataframe in the same structure
    # as we got it.
    if og_index:
        plant_parts_df = plant_parts_df.set_index("record_id_eia")
    return plant_parts_df


def get_master_unit_list_eia(file_path_mul, pudl_out, clobber=False):
    """
    Get the master unit list; generate it or get if from a file.

    If you generate the MUL, it will be saved at the file path given.

    Args:
        file_path_mul (pathlib.Path): where you want the master unit list to
            live. Must be a compressed pickle file ('.pkl.gz').
        clobber (boolean): True if you want to regenerate the master unit list
            whether or not it is saved at the file_path_mul
    """
    if '.pkl' not in file_path_mul.suffixes:
        raise AssertionError(f"{file_path_mul} must be a pickle file")
    if not file_path_mul.is_file() or clobber:
        logger.info(
            f"Master unit list not found {file_path_mul}"
            "Generating a new master unit list. This should take ~10 minutes."
        )
        # actually make the master plant parts list
        plant_parts_eia = pudl_out.plant_parts_eia()
        # export
        plant_parts_eia.to_pickle(file_path_mul, compression='gzip')

    elif file_path_mul.is_file():
        logger.info(f"Reading the master unit list from {file_path_mul}")
        plant_parts_eia = pd.read_pickle(file_path_mul, compression='gzip')
    return plant_parts_eia
