"""Beginning of compilation of FERC/EIA granular connections."""

import logging

import pandas as pd

from plant_parts import plant_parts

logger = logging.getLogger(__name__)


def _grab_test_xlxs():
    """TODO: Add file path."""
    logger.info('')
    return pd.read_excel('pudl_learning.xlsx',
                         skiprows=1,
                         dtype={'EIA Plant Code': pd.Int64Dtype(),
                                'Generator': pd.Int64Dtype(),
                                'EIA Utility Code': pd.Int64Dtype(),
                                'report_year': pd.Int64Dtype(),
                                'report_prd': pd.Int64Dtype(),
                                'respondent_id': pd.Int64Dtype(),
                                'spplmnt_num': pd.Int64Dtype(),
                                })


def _prep_test_connections(compiler):
    """TODO: Clean and condense."""
    test_df = _grab_test_xlxs()
    cols_to_rename = {
        'EIA Plant Code': 'plant_id_eia',
        'FERC Line Type': 'plant_part',
        'EIA Utility Code': 'utility_id_eia',
        'Unit Code': 'unit_id_pudl',
        'EIA Technology': 'technology_description',
        'Generator': 'generator_id',
        'EIA Prime Mover': 'prime_mover_code',
        'EIA Energy Source Code': 'energy_source_code_1',
        'eia_ownership': 'ownership', }
    string_cols = ['FERC Line Type', 'EIA Technology',
                   'EIA Prime Mover', 'EIA Energy Source Code',
                   'Owned or Total']
    plant_part_rename = {
        'plant': 'plant',
        'generator': 'plant_gen',
        'unit': 'plant_unit',
        'technology': 'plant_technology',
        'plant_prime_fuel': 'plant_prime_fuel',
        'plant_prime': 'plant_prime_mover'}

    for col in string_cols:
        if col in test_df.columns:
            test_df.loc[:, col] = (
                test_df[col].astype(str).
                str.strip().
                str.lower().
                str.replace(r'\s+', '_')
            )

    test_df = (test_df.assign(report_date='2018-01-01').
               rename(columns=cols_to_rename).
               astype({'report_date': 'datetime64[ns]',
                       'utility_id_eia': pd.Int64Dtype()}).
               replace(plant_part_rename))

    test_df_ids = pd.DataFrame()
    for part in plant_parts:
        test_df_ids = pd.concat(
            [test_df_ids,
             compiler.add_record_id(test_df[test_df['plant_part'] == part],
                                    plant_parts[part]['id_cols'])])
    test_df_ids['record_id_ferc'] = (
        test_df_ids.Source + '_' +
        test_df_ids.report_year.astype(str) + '_' +
        test_df_ids.report_prd.astype(str) + '_' +
        test_df_ids.respondent_id.astype(str) + '_' +
        test_df_ids.spplmnt_num.astype(str)
    )
    if "row_number" in test_df_ids.columns:
        test_df_ids["record_id_ferc"] = test_df_ids["record_id_ferc"] + \
            "_" + test_df_ids.row_number.astype('Int64').astype(str)

    return test_df_ids.set_index(['record_id_ferc', 'record_id_eia'])
