"""
EIA Plant-parts list is actively being moved into PUDL.

Here are some dangly bits that we are actively using to read the plant-parts
list from PUDL into the other realms of this repo.
"""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


<<<<<<< HEAD
PLANT_PARTS = {
    'plant': {
        'id_cols': ['plant_id_eia'],
    },
    'plant_gen': {
        'id_cols': ['plant_id_eia', 'generator_id'],
    },
    'plant_unit': {
        'id_cols': ['plant_id_eia', 'unit_id_pudl'],
    },
    'plant_technology': {
        'id_cols': ['plant_id_eia', 'technology_description'],
    },
    'plant_prime_fuel': {
        'id_cols': ['plant_id_eia', 'energy_source_code_1'],
    },
    'plant_prime_mover': {
        'id_cols': ['plant_id_eia', 'prime_mover_code'],
    },
    'plant_ferc_acct': {
        'id_cols': ['plant_id_eia', 'ferc_acct_name'],
    },
    #    'plant_install_year': {
    #        'id_cols': ['plant_id_eia', 'installation_year'],
    #    },
}
"""
dict: this dictionary contains a key for each of the 'plant parts' that should
end up in the mater unit list. The top-level value for each key is another
dictionary, which contains keys:
    * id_cols (the primary key type id columns for this plant part)
"""

PLANT_PARTS_ORDERED = [
    'plant',
    'plant_unit',
    'plant_prime_mover',
    'plant_technology',
    'plant_prime_fuel',
    'plant_ferc_acct',
    'plant_gen'
]


IDX_TO_ADD = ['report_date', 'operational_status_pudl']
"""
iterable: list of additional columns to add to the id_cols in `PLANT_PARTS`.
The id_cols are the base columns that we need to aggregate on, but we also need
to add the report date to keep the records time sensitive and the
operational_status_pudl to separate the operating plant-parts from the
non-operating plant-parts.
"""

IDX_OWN_TO_ADD = ['utility_id_eia', 'ownership']
"""
iterable: list of additional columns beyond the IDX_TO_ADD to add to the
id_cols in `PLANT_PARTS` when we are dealing with plant-part records that have
been broken out into "owned" and "total" records for each of their owners.
"""

SUM_COLS = [
    'total_fuel_cost',
    'net_generation_mwh',
    'capacity_mw',
    'capacity_mw_eoy',
    'total_mmbtu',
]
"""
iterable: list of columns to sum when aggregating a table.
"""

WTAVG_DICT = {
    'fuel_cost_per_mwh': 'capacity_mw',
    'heat_rate_mmbtu_mwh': 'capacity_mw',
    'fuel_cost_per_mmbtu': 'capacity_mw',
}
"""
dict: a dictionary of columns (keys) to perform weighted averages on and
the weight column (values)"""


QUAL_RECORDS = [
    'fuel_type_code_pudl',
    'operational_status',
    'planned_retirement_date',
    'retirement_date',
    'generator_id',
    'unit_id_pudl',
    'technology_description',
    'energy_source_code_1',
    'prime_mover_code',
    'ferc_acct_name',
    # 'installation_year'
]
"""
dict: a dictionary of qualifier column name (key) and original table (value).
"""
=======
MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]
>>>>>>> main

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

MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]


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

    Args:
        file_path_mul (pathlib.Path)
        clobber (boolean): True if you want to regenerate the master unit list
            whether or not it is saved at the file_path_mul
    """
    if not file_path_mul.is_file() or clobber:
        logger.info(
            f"Master unit list not found {file_path_mul}"
            "Generating a new master unit list. This should take ~10 minutes."
        )
<<<<<<< HEAD
        # initilize the compilers
        gens_maker = MakeMegaGenTbl(pudl_out)
        grans_labeler = LabelTrueGranularities(gens_maker)
        parts_compiler = MakePlantParts(pudl_out, gens_maker, grans_labeler)
=======
>>>>>>> main
        # actually make the master plant parts list
        plant_parts_eia = pudl_out.plant_parts_eia()
        # export
<<<<<<< HEAD
        plant_parts_df.to_csv(file_path_mul, compression='gzip')
=======
        plant_parts_eia.to_csv(file_path_mul, compression='gzip')
>>>>>>> main

    elif file_path_mul.is_file():
        logger.info(f"Reading the master unit list from {file_path_mul}")
        plant_parts_eia = pd.read_pickle(file_path_mul, compression='gzip')
    return plant_parts_eia
