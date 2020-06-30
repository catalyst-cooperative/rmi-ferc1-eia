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
        # TODO: This is a bit of a placeholder riht now. I'd like to make
        # functions like the get_master_unit_list_eia for each of these
        # components. Right now, the pickled outputs are expected to be there.
        self.plant_parts_eia_raw = (
            make_plant_parts_eia.get_master_unit_list_eia(file_path_mul))
        # right now we need both the steam table and the ferc1_eia connection
        # because mostly
        self.steam_cleaned_ferc1_raw = pd.read_pickle(
            file_path_steam_ferc1, compression='gzip')
        self.connects_ferc1_eia_raw = pd.read_pickle(
            file_path_ferc1_eia, compression='gzip')
        self.connects_deprish_eia_raw = pd.read_pickle(
            file_path_deprish_eia, compression='gzip')

    def _prep_plant_parts_eia(self):
        self.plant_parts_eia = (
            self.plant_parts_eia_raw.reset_index()
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']))

    def _prep_steam_cleaned_ferc1(self):
        self.steam_cleaned_ferc1 = (
            self.steam_cleaned_ferc1_raw.reset_index()
                .pipe(pudl.helpers.convert_to_date))

    def _prep_connects_ferc1_eia(self):
        # should this be done over in connect_ferc1_to_eia land?
        # i think much of this and _prep_steam_cleaned_ferc1 should be moved.
        id_cols = ['plant_id_pudl', 'utility_id_pudl', ]
        self.connects_ferc1_eia = (
            pd.merge(
                self.connects_ferc1_eia_raw.reset_index()[
                    ['record_id_ferc', 'record_id_eia']],
                # we only want the identifying columns from the MUL
                self.plant_parts_eia[connect_deprish_to_eia.MUL_COLS
                                     + id_cols],
                how='left')
            .astype(connect_deprish_to_eia.prep_int_ids(id_cols))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
            # we want the backbone of this table to be the steam records
            # so we have all possible steam records, even the unmapped ones
            .pipe(pd.merge, self.steam_cleaned_ferc1, how='right')
        )

        if len(self.connects_ferc1_eia) != len(self.steam_cleaned_ferc1):
            raise AssertionError(
                """Merge between steam_cleaned_ferc1 and connects_ferc1_eia erred.
                The output and the orignal table should be the same length.
                Check the columns included.""")

    def _prep_connects_deprish_eia(self):
        self.connects_deprish_eia = (
            # we only want options that have actually been connected to the MUL
            self.connects_deprish_eia_raw[
                self.connects_deprish_eia_raw.record_id_eia.notnull()]
            .pipe(pudl.helpers.convert_to_date)
            .astype(connect_deprish_to_eia.prep_int_ids(
                ['plant_id_pudl', 'utility_id_pudl', 'utility_id_ferc1']))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
        )
        # we are going to merge the master unit list into this output,
        # because we want all of the id columns from MUL_COLS.
        # There are some overlapping columns. We really only want to
        # merge on the 'record_id_eia' and we trust the master unit list
        # more than the spreadsheet based connection for deprish to eia
        # so we are going to only use columns from the deprish_to_eia that
        # don't show up in the MUL_COLS
        cols_to_use_deprish_eia = (
            ['record_id_eia'] +
            [c for c in self.connects_deprish_eia.columns
             if c not in connect_deprish_to_eia.MUL_COLS])

        self.connects_deprish_eia = pd.merge(
            self.connects_deprish_eia[cols_to_use_deprish_eia],
            self.plant_parts_eia[connect_deprish_to_eia.MUL_COLS])

    def _prep_plant_parts_ordered(self):
        pudl_settings = pudl.workspace.setup.get_defaults()
        pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
        table_compiler = make_plant_parts_eia.CompileTables(
            pudl_engine, freq='AS', rolling=True)
        parts_compilers = make_plant_parts_eia.CompilePlantParts(
            table_compiler, clobber=True)
        self.plant_parts_ordered = parts_compilers.plant_parts_ordered
        # we need a dictionary of plant part named (keys) to their
        # corresponding id columns (values). parts_compilers has the inverse of
        # that sowe are just going to swap the ks and vs
        self.parts_to_ids = {v: k for k, v
                             in parts_compilers.ids_to_parts.items()}

    def prep_inputs(self):
        """Prepare all inputs needed for connecting deprecation to FERC1."""
        # the order here is important. We are preping the inputs needed
        # for later inputs
        self._prep_plant_parts_eia()
        self._prep_steam_cleaned_ferc1()
        self._prep_connects_ferc1_eia()
        self._prep_connects_deprish_eia()

        self._prep_plant_parts_ordered()


def check_high_level_connections(
        connects_all_deprish_ferc1,
        connects_ferc1_eia):
    """Check the connections between deprecation data and ferc1."""
    # there was a merge iindicator here and left df was the depreciation data
    connected_plant_ids = connects_all_deprish_ferc1[
        connects_all_deprish_ferc1._merge == 'both'].plant_id_pudl.unique()
    # how many plant_id_pudl's didn't get a corresponding FERC1 record
    not_in_ferc1_plant_ids = (connects_all_deprish_ferc1[
        connects_all_deprish_ferc1._merge == 'left_only']
        .plant_id_pudl.unique())
    # these are a subset of the not_in_ferc1_plant_ids that
    missing_plant_ids = (connects_ferc1_eia[
        connects_ferc1_eia.plant_id_pudl.isin(not_in_ferc1_plant_ids)]
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
    """
    Prepare all options between deprecation and ferc1 steam.

    Args:
        inputs (object): an instance of DeprishToFERC1Inputs

    Returns:
        pandas.DataFrame: a dataframe with all of the possible combinations of
        the deprecation data and ferc1 (with their relative EIA master unit
        list records associated).
    """
    inputs.prep_inputs()

    options_all_deprish_ferc1 = pd.merge(
        inputs.connects_deprish_eia,
        inputs.connects_ferc1_eia,
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
        options_all_deprish_ferc1, inputs.connects_ferc1_eia)

    # a rename dict with the ordered plant part names with numbered prefixes
    replace_dict = {
        x:
        f"{inputs.plant_parts_ordered.index(x)}_{x}"
        for x in inputs.plant_parts_ordered
    }
    # we're going to add a column level_deprish, which indicates the relative
    # size of the plant part granularity between deprecaition and ferc1
    options_all_deprish_ferc1 = (
        options_all_deprish_ferc1.assign(
            part_no_deprish=lambda x:
                x.plant_part_deprish.replace(replace_dict),
            part_no_ferc1=lambda x:
                x.plant_part_ferc1.replace(replace_dict),
            level_deprish=lambda x:
                np.where(x.part_no_deprish == x.part_no_ferc1,
                         'samezies', None),)
        .assign(
            level_deprish=lambda x:
                np.where(x.part_no_deprish < x.part_no_ferc1,
                         'beeg', x.level_deprish),)
        .assign(
            level_deprish=lambda x:
                np.where(x.part_no_deprish > x.part_no_ferc1,
                         'smol', x.level_deprish))
        .drop(columns=['part_no_deprish', 'part_no_ferc1'])
    )
    # we are going to make a count_ferc1 column to know how many possible
    # ferc1 connections are possible.
    options_all_deprish_ferc1 = pd.merge(
        options_all_deprish_ferc1,
        options_all_deprish_ferc1.groupby(['record_id_eia_deprish'])
        .agg({'record_id_ferc': 'count'})
        .rename(columns={'record_id_ferc': 'count_ferc1', })
        .reset_index(),
    )
    return options_all_deprish_ferc1


def get_matches_at_diff_ownership(options_df):
    """
    Get matches when the record_id_eia matches except for the ownership.

    Note: Is there a cleaner way to do this??
    """
    ph = 'placeholder'
    diff_own = (options_df[
        options_df.record_id_eia_deprish.str
        .replace('owned', ph, regex=True).replace('total', ph, regex=True) ==
        options_df.record_id_eia_ferc1.str
        .replace('owned', ph, regex=True).replace('total', ph, regex=True)])
    return diff_own


def get_matches_same_qualifiers_ids(inputs, options_deprish_ferc1, source):
    """
    Get matches that have the same qualifier ids part level.

    In the generation of the master unit list, we  the identifying columns
    for the plant parts/levels are reported consi

    If one dataset reports at the level of a prime mover and another dataset
    reports at the level of units, when the prime mover matches the consistent
    reported prime

    Args:
        inputs (object): an instance of DeprishToFERC1Inputs
        options_deprish_ferc1 (pandas.DataFrame): dataframe of the options of
            possible connections between the ferc1 and deprecation.
        source (string): either `ferc1` or `deprish`
    """
    df = pd.DataFrame()
    for part_name in inputs.plant_parts_ordered:
        # this doesn't work for the plant bc all of the records within a plant
        # share the plant id
        if part_name != 'plant':
            part_df = options_deprish_ferc1[
                options_deprish_ferc1[f"plant_part_{source}"] == part_name]
            # add the slice of the df for this part if the id columns for both
            df = df.append(part_df[
                part_df[f"{inputs.parts_to_ids[part_name]}_deprish"]
                == part_df[f"{inputs.parts_to_ids[part_name]}_ferc1"]])
    df = df.assign(connect_method=f"same_qualifiers_{source}")
    return df


def remove_connections_from_options(options_df_current, connections_df):
    """
    Remove the connections from the possible options.

    Because we are iteratively generating connections, we want to remove the
    connections we've determined from the options for future iterations.
    """
    # if the record_id_eia_deprish shows up in the options, remove it
    # the only options left are the ones that do not show up in the connections
    options_df_remainder = options_df_current[
        ~options_df_current.record_id_eia_deprish
        .isin(connections_df.record_id_eia_deprish.unique())]
    return options_df_remainder


def connect(inputs):
    """
    Connect depreciation records with ferc1 steam records.

    TODO: This is a big messy WIP function right now.

    Args:
        inputs (object)

    Returns:
        pandas.DataFrame
    """
    # Connections are known to be connected records and options are the
    # possible connections.
    options_all_deprish_ferc1 = prep_all_options(inputs)

    # we are going to go through various methods for grabbing the true
    # connections out of the options.
    # we will then label those options with a connect_method column
    # we are going to continually remove the known connections from the options

    # method 1:
    # grab the obvious connections which have the same record_id_eia
    same_true = (options_all_deprish_ferc1[
        options_all_deprish_ferc1.record_id_eia_deprish ==
        options_all_deprish_ferc1.record_id_eia_ferc1]
        .assign(connect_method="same_true")
    )
    options_deprish_ferc1 = remove_connections_from_options(
        options_all_deprish_ferc1, same_true)

    # method 2:
    # potential matches that has the same record id except for their ownership
    # level.
    same_diff_own = (get_matches_at_diff_ownership(options_deprish_ferc1)
                     .assign(connect_method="same_diff_own"))
    options_deprish_ferc1 = remove_connections_from_options(
        options_deprish_ferc1, same_diff_own)

    # method 3:
    # if there is only one ferc1 option, then we have no other choice
    one_ferc1_opt = (options_deprish_ferc1[
        options_deprish_ferc1.count_ferc1 == 1]
        .assign(connect_method="one_ferc1_opt"))
    options_deprish_ferc1 = remove_connections_from_options(
        options_deprish_ferc1, one_ferc1_opt)

    # method 4:
    # if the different leveled records have the same qualifier id columns
    # then they can be considered the same
    same_quals = pd.DataFrame()
    for source in ['ferc1', 'deprish']:
        same_quals = same_quals.append(get_matches_same_qualifiers_ids(
            inputs, options_deprish_ferc1, source))
    options_deprish_ferc1 = remove_connections_from_options(
        options_deprish_ferc1, same_quals)

    # squish all of the known connections together
    connects = pd.concat([same_true, same_diff_own, one_ferc1_opt, same_quals])

    # everything below here is just for logging
    ids_to_match = (options_all_deprish_ferc1[
        options_all_deprish_ferc1.record_id_eia_ferc1.notnull()]
        .record_id_eia_deprish.unique())
    ids_connected = connects.record_id_eia_deprish.unique()
    ids_no_match = (options_deprish_ferc1[
        options_deprish_ferc1.record_id_eia_ferc1.notnull()]
        .plant_id_pudl.unique())
    logger.info("Portion of unique depreciation records:")
    logger.info(f"    Matched:   {len(ids_connected)/len(ids_to_match):.02%}")
    logger.info(f"    No link:   {len(ids_no_match)/len(ids_to_match):.02%}")
    logger.debug(f"    Same true: {len(same_true)}")
    logger.info(f"""Connected:
{connects.connect_method.value_counts(dropna=False)}""")
    logger.info(f"""Connection Levels:
{options_deprish_ferc1.level_deprish.value_counts(dropna=False)}""")
    logger.debug(f"""Only one ferc1 option levels:
{one_ferc1_opt.level_deprish.value_counts(dropna=False)}""")

    # evertually this will be a dealt w/ squished together output
    # for now, this is a few important outputs
    return options_all_deprish_ferc1, options_deprish_ferc1, connects
