"""Connect the deprecation data with FERC1 steam plant records."""

import logging
import warnings

import numpy as np
import pandas as pd
import sqlalchemy as sa

import connect_deprish_to_eia
import make_plant_parts_eia
import pudl

logger = logging.getLogger(__name__)

CONNECT_COLS = ['plant_id_pudl',
                'utility_id_pudl',
                'report_date']


class DeprishToFERC1Inputs():
    """Input manager for connections betweenn FERC 1 and depreciation data."""

    def __init__(self, file_path_mul, file_path_steam_ferc1,
                 file_path_ferc1_eia, file_path_deprish_eia):
        """
        Initialize input manager for connecting depreciation to FERC1.

        Args:
            file_path_mul (path-like): file path to the EIA master unit list
            file_path_steam_ferc1 (path-like): file path to the compiled FERC1
                steam table
            file_path_ferc1_eia (path-like): file path to the table connecting
                FERC1 steam records to the master unit list
            file_path_deprish_eia (path-like): file path to the table
                connecting the deprecation records to the EIA master unit list
        """
        self.file_path_mul = file_path_mul
        self.file_path_steam_ferc1 = file_path_steam_ferc1
        self.file_path_ferc1_eia = file_path_ferc1_eia
        self.file_path_deprish_eia = file_path_deprish_eia

        # TODO: This is a bit of a placeholder riht now. I'd like to make
        # functions like the get_master_unit_list_eia for each of these
        # components. Right now, the pickled outputs are expected to be there.

        self.plant_parts_eia_raw = (
            make_plant_parts_eia.get_master_unit_list_eia(self.file_path_mul))
        self.steam_cleaned_ferc1_raw = pd.read_pickle(
            self.file_path_steam_ferc1, compression='gzip')
        self.connects_ferc1_eia_raw = pd.read_pickle(
            self.file_path_ferc1_eia, compression='gzip')
        self.connects_deprish_eia_raw = pd.read_pickle(
            self.file_path_deprish_eia, compression='gzip')

    def _prep_plant_parts_eia(self):
        self.plant_parts_eia = (
            self.plant_parts_eia_raw.reset_index()
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']))

    def _prep_connects_ferc1_eia(self):
        # should this be done over in connect_ferc1_to_eia land?
        # i think much of this and _prep_steam_cleaned_ferc1 should be moved.
        self.connects_ferc1_eia = (
            pd.merge(
                self.connects_ferc1_eia_raw.reset_index()[
                    ['record_id_ferc', 'record_id_eia']],
                self.plant_parts_eia[
                    ['record_id_eia', 'plant_part', 'report_date',
                     'plant_id_pudl', 'utility_id_pudl']],
                how='left')
            .astype(connect_deprish_to_eia.prep_int_ids(
                ['plant_id_pudl', 'utility_id_pudl', ]))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
        )

    def _prep_connects_deprish_eia(self):
        self.connects_deprish_eia = (
            self.connects_deprish_eia_raw[
                self.connects_deprish_eia_raw.record_id_eia.notnull()]
            .pipe(pudl.helpers.convert_to_date)
            .astype(connect_deprish_to_eia.prep_int_ids(
                ['plant_id_pudl', 'utility_id_pudl', 'utility_id_ferc1']))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
        )

    def _prep_steam_cleaned_ferc1(self):
        self.steam_cleaned_ferc1 = (
            pd.merge(
                self.steam_cleaned_ferc1_raw.reset_index()
                .pipe(pudl.helpers.convert_to_date),
                self.connects_ferc1_eia[
                    ['record_id_ferc', 'record_id_eia', 'plant_part', ]],
                on=['record_id_ferc'],
                how='left'
            )
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
        )

    def _prep_plant_parts_ordered(self):
        pudl_settings = pudl.workspace.setup.get_defaults()
        pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
        table_compiler = make_plant_parts_eia.CompileTables(
            pudl_engine, freq='AS', rolling=True)
        parts_compilers = make_plant_parts_eia.CompilePlantParts(
            table_compiler, clobber=True)
        self.plant_parts_ordered = parts_compilers.plant_parts_ordered

    def prep_inputs(self):
        """Prepare all inputs needed for connecting deprecation to FERC1."""
        # the order here is important. We are preping the inputs needed
        # for later inputs
        self._prep_plant_parts_eia()
        self._prep_connects_ferc1_eia()
        self._prep_connects_deprish_eia()
        self._prep_steam_cleaned_ferc1()

        self._prep_plant_parts_ordered()


def check_high_level_connections(
        connects_all_deprish_ferc1,
        steam_cleaned_ferc1):
    """Check the connections between deprecation data and ferc1."""
    # there was a merge iindicator here and left df was the depreciation data
    connected_plant_ids = connects_all_deprish_ferc1[
        connects_all_deprish_ferc1._merge == 'both'].plant_id_pudl.unique()
    # how many plant_id_pudl's didn't get a corresponding FERC1 record
    not_in_ferc1_plant_ids = (connects_all_deprish_ferc1[
        connects_all_deprish_ferc1._merge == 'left_only']
        .plant_id_pudl.unique())
    # these are a subset of the not_in_ferc1_plant_ids that
    missing_plant_ids = (steam_cleaned_ferc1[
        steam_cleaned_ferc1.plant_id_pudl.isin(not_in_ferc1_plant_ids)]
        .plant_id_pudl.unique())
    logger.info(f"Connected plants:    {len(connected_plant_ids)}")
    logger.info(f"Not connected:       {len(not_in_ferc1_plant_ids)}")
    logger.info(f"Missing connections: {len(missing_plant_ids)}")
    # Investigation of 3 missing connections [1204, 1147, 1223] determined that
    # these plants are missing from ferc because there was no reporting
    # for the specific years in question
    if len(missing_plant_ids) > 3:
        warnings.warn(
            f'There are {len(missing_plant_ids)} missing plant records.')


def prep_all_options(inputs):
    """Prepare all options between deprecation and ferc1 steam."""
    # inputs.prep_inputs()

    options_all_deprish_ferc1 = pd.merge(
        inputs.connects_deprish_eia,
        inputs.steam_cleaned_ferc1,
        on=CONNECT_COLS,
        suffixes=('_deprish', '_ferc1'),
        how='left', indicator=True
    )
    # reorder cols so they are easier to see, maybe remove later
    first_cols = ['plant_part_deprish', 'plant_part_ferc1',
                  'record_id_eia_deprish', 'record_id_eia_ferc1']
    options_all_deprish_ferc1 = options_all_deprish_ferc1[
        first_cols + [x for x in options_all_deprish_ferc1.columns
                      if x not in first_cols]]

    check_high_level_connections(
        options_all_deprish_ferc1, inputs.steam_cleaned_ferc1)

    # a rename dict with the ordered plant part names with numbered prefixes
    replace_dict = {
        x:
        f"{inputs.plant_parts_ordered.index(x)}_{x}"
        for x in inputs.plant_parts_ordered
    }

    options_all_deprish_ferc1 = (
        options_all_deprish_ferc1.assign(
            part_no_deprish=lambda x:
                x.plant_part_deprish.replace(replace_dict),
            part_no_ferc1=lambda x:
                x.plant_part_ferc1.replace(replace_dict),
            level_deprish=lambda x:
                np.where(x.part_no_deprish == x.part_no_ferc1,
                         'samezies', None),
        .assign(
            level_deprish=lambda x:
                np.where(x.part_no_deprish < x.part_no_ferc1,
                         'beeeg', x.level_deprish),)
        .assign(
            level_deprish=lambda x:
                np.where(x.part_no_deprish > x.part_no_ferc1,
                         'smol', x.level_deprish))
        .drop(columns=['part_no_deprish', 'part_no_ferc1'])

    )
    return options_all_deprish_ferc1


def get_matches_at_diff_ownership(options_df):
    """
    G.

    Note: Is there a cleaner way to do this??
    """
    ph='placeholder'
    diff_own=(options_df[
        options_df.record_id_eia_deprish.str
        .replace('owned', ph, regex=True).replace('total', ph, regex=True) ==
        options_df.record_id_eia_ferc1.str
        .replace('owned', ph, regex=True).replace('total', ph, regex=True)])
    return diff_own


def connect(inputs):
    """
    Connect depreciation records with ferc1 steam records.

    TODO: This is a big messy WIP function right now.

    Args:
        inputs (object)

    Returns:
        pandas.DataFrame
    """
    options_all_deprish_ferc1=prep_all_options(inputs)
    # grab the connections
    same_true=options_all_deprish_ferc1[
        options_all_deprish_ferc1.record_id_eia_deprish ==
        options_all_deprish_ferc1.record_id_eia_ferc1]
    # and remove them from the options
    options_deprish_ferc1=options_all_deprish_ferc1[
        ~options_all_deprish_ferc1.record_id_eia_deprish.isin(
            same_true.record_id_eia_deprish.unique())]

    same_diff_own=get_matches_at_diff_ownership(options_deprish_ferc1)
    # remove from options
    options_deprish_ferc1=options_deprish_ferc1[
        ~options_deprish_ferc1.record_id_eia_deprish.isin(
            same_diff_own.record_id_eia_deprish.unique())]
    # same = options_deprish_ferc1[
    #    options_deprish_ferc1.level_deprish == 'samezies']

    smol=options_deprish_ferc1[
        options_deprish_ferc1.level_deprish == 'smol']
    beeeg=options_deprish_ferc1[
        options_deprish_ferc1.level_deprish == 'beeeg']

    nah=options_deprish_ferc1[
        options_deprish_ferc1.level_deprish == 'who knows??']

    connects=pd.concat([same_true, same_diff_own])
    ids_to_match=options_all_deprish_ferc1.record_id_eia_deprish.unique()
    logger.info("Portion of unique depreciation records:")
    logger.info(f"    Connected:  {len(connects)/len(ids_to_match):.02%}")
    logger.info(f"    Same level: {len(same_true)/len(ids_to_match):.02%}")
    logger.info(f"    Smaller:    {len(smol)/len(ids_to_match):.02%}")
    logger.info(f"    Bigger:     {len(beeeg)/len(ids_to_match):.02%}")
    logger.info(f"    No link:    {len(nah)/len(ids_to_match):.02%}")

    # evertually this will be a dealt w/ squished together output
    # for now, this is a few important outputs
    return options_all_deprish_ferc1, options_deprish_ferc1, connects
