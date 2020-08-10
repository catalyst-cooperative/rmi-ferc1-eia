"""
Connect the depreciation data with FERC1 steam plant records.

This module attempts to connect the depreciation data with FERC1 steam records.
Both the depreciation records and FERC1 steam has been connected to the EIA
master unit list, which is a compilation of various possible combinations of
generator records.

Matches are determined to be correct record linkages.
Candidate matches are potential matches.
"""

import logging
import warnings
from copy import deepcopy

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

SPLIT_COLS_STANDARD = [
    'total_fuel_cost_deprish',
    'net_generation_mwh_deprish',
    'capacity_mw_deprish',
]
"""
list: the standard columns to split ferc1 data columns to be used in
``DATA_COLS_TO_SPLIT``.
"""

DATA_COLS_TO_SPLIT = {
    'opex_nonfuel': SPLIT_COLS_STANDARD,
    'net_generation_mwh_ferc1': SPLIT_COLS_STANDARD,
}
"""
dictionary: FERC1 data columns (keys) that we want to associated with
depreciation records. When the FERC1 record is larger than the depreciation
record (e.g. a group of depreciation generators matched with a FERC1 plant),
this module attemps to split the depreciation record based on the list of
columns to weight the split (values). See  ``split_ferc1_data_on_split_cols()``
for more details.
"""


class InputsCompiler():
    """
    Input manager for matches between FERC 1 and depreciation data.

    This input mananger reads in and stores data from four sources. The data is
    prepared, which generally involved ensuring that all output tables have
    all of the neccesary columns from the master unit list to determine if
    candidate matches are indeed true matches.

    The outputs that are generated from this class and used later on are:
    * connects_deprish_eia: a connection between the depreciation data and
        eia's master unit list
    * connects_ferc1_eia.
    * plant_parts_ordered: list of ordered plant parts for the master unit list
    * parts_to_ids: dictionary of plant part names (keys) to identifying
        columns (values)
    """

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
                connecting the depreciation records to the EIA master unit list
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

        # store a bool which will indicate whether the inputs have been prepped
        self.inputs_prepped = False

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
                    ['record_id_ferc1', 'record_id_eia']],
                # we only want the identifying columns from the MUL
                self.plant_parts_eia[list(set(
                    connect_deprish_to_eia.MUL_COLS + id_cols
                    + ['total_fuel_cost', 'net_generation_mwh', 'capacity_mw']
                ))],
                how='left', on=['record_id_eia'])
            .astype(connect_deprish_to_eia.prep_int_ids(id_cols))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
            # we want the backbone of this table to be the steam records
            # so we have all possible steam records, even the unmapped ones
            .merge(self.steam_cleaned_ferc1,
                   how='right', on=['record_id_ferc1'] + id_cols,
                   suffixes=('_eia_ferc1', ''))
            .assign(opex_nonfuel=lambda x: (x.opex_production_total -
                                            x.opex_fuel))
        )

        if len(self.connects_ferc1_eia) != len(self.steam_cleaned_ferc1):
            raise AssertionError(
                """Merge between steam_cleaned_ferc1 and connects_ferc1_eia erred.
                The output and the orignal table should be the same length.
                Check the columns included.""")

    def _prep_connects_deprish_eia(self):
        self.connects_deprish_eia = (
            # we only want candidate matches that have actually been connected
            # to the MUL
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
            self.plant_parts_eia[
                connect_deprish_to_eia.MUL_COLS
                + ['total_fuel_cost', 'net_generation_mwh', 'capacity_mw']])

    def _prep_info_from_parts_compiler_eia(self):
        """
        Prepare some info from make_plant_parts_eia.CompilePlantParts.

        We need a few things from the class that generates the master unit
        list; mostly info about the identifying columns for the plant parts.
        """
        # CompilePlantParts requires and instance of CompileTables
        pudl_engine = sa.create_engine(
            pudl.workspace.setup.get_defaults()["pudl_db"])
        table_compiler = make_plant_parts_eia.CompileTables(
            pudl_engine, freq='AS')
        parts_compilers = make_plant_parts_eia.CompilePlantParts(
            table_compiler, clobber=True)
        self.plant_parts_ordered = parts_compilers.plant_parts_ordered
        # we need a dictionary of plant part named (keys) to their
        # corresponding id columns (values). parts_compilers has the inverse of
        # that sowe are just going to swap the ks and vs
        self.parts_to_ids = {v: k for k, v
                             in parts_compilers.ids_to_parts.items()}

    def prep_inputs(self, clobber=False):
        """Prepare all inputs needed for connecting depreciation to FERC1."""
        # the order here is important. We are preping the inputs needed
        # for later inputs
        if not self.inputs_prepped or clobber:
            self._prep_info_from_parts_compiler_eia()

            self._prep_plant_parts_eia()
            self._prep_steam_cleaned_ferc1()
            self._prep_connects_ferc1_eia()
            self._prep_connects_deprish_eia()

            self.inputs_prepped = True


class MatchMaker():
    """
    Generate matches between depreciation and FERC1 steam records.

    The coordinating method in this class is `connect` - see it for additional
    details.
    """

    def __init__(self, inputs):
        """
        Initialize MatchMaker.

        Store instance of InputsCompiler so we can use the prepared
        dataframes.

        Args:
            inputs (object): an instance of InputsCompiler

        """
        self.inputs = inputs
        inputs.prep_inputs()

    def check_all_candidate_matches(self, candidate_matches_all):
        """
        Check the candidate matches between depreciation data and ferc1.

        This method explores the generation of matches between depreciation
        and ferc1 records. We want to know how many depreciation records aren't
        associated with any ferc1 record. We want to know if there are any
        plant ids that show up in the depreciation data and aren't mapped to
        ferc records but do show up in the ferc data somewhere. At a high
        level, we want a gut check of whether or not connects_all_deprish_ferc1
        was connected properly.
        """
        # there was a merge iindicator here and left df was the depreciation
        # data
        connected_plant_ids = candidate_matches_all[
            candidate_matches_all._merge == 'both'].plant_id_pudl.unique()
        # how many plant_id_pudl's didn't get a corresponding FERC1 record
        not_in_ferc1_plant_ids = (candidate_matches_all[
            candidate_matches_all._merge == 'left_only']
            .plant_id_pudl.unique())
        # these are a subset of the not_in_ferc1_plant_ids that show up in the
        # steam table
        missing_plant_ids = (self.inputs.connects_ferc1_eia[
            self.inputs.connects_ferc1_eia.plant_id_pudl.isin(
                not_in_ferc1_plant_ids)].plant_id_pudl.unique())
        logger.info(f"Matched plants:    {len(connected_plant_ids)}")
        logger.info(f"Not connected:       {len(not_in_ferc1_plant_ids)}")
        logger.info(f"Missing connections: {len(missing_plant_ids)}")
        # Investigation of 3 missing connections [1204, 1147, 1223] determined
        # that these plants are missing from ferc because there was no
        # reporting for the specific years in question
        if len(missing_plant_ids) > 3:
            warnings.warn(
                f'There are {len(missing_plant_ids)} missing plant records.')

    def get_candidate_matches(self):
        """
        Prepare all candidate matches between depreciation and ferc1 steam.

        Before choosing the specific match between depreciation and ferc1, we
        need to compile all possible options - or candidate links.

        Returns:
            pandas.DataFrame: a dataframe with all of the possible combinations
            of the deprecation data and ferc1 (with their respective EIA master
            unit list records associated).
        """
        candidate_matches_all = pd.merge(
            self.inputs.connects_deprish_eia,
            self.inputs.connects_ferc1_eia,
            on=CONNECT_COLS,
            suffixes=('_deprish', '_ferc1'),
            how='left', indicator=True
        )
        # reorder cols so they are easier to see, maybe remove later
        first_cols = ['plant_part_deprish', 'plant_part_ferc1',
                      'record_id_eia_deprish', 'record_id_eia_ferc1',
                      'plant_name', 'plant_name_match',
                      'fraction_owned_deprish', 'fraction_owned_ferc1',
                      'record_count_deprish', 'record_count_ferc1'
                      ]
        candidate_matches_all = candidate_matches_all[
            first_cols + [x for x in candidate_matches_all.columns
                          if x not in first_cols]]

        self.check_all_candidate_matches(candidate_matches_all)

        # rename dict with the ordered plant part names with numbered prefixes
        replace_dict = {
            x:
            f"{self.inputs.plant_parts_ordered.index(x)}_{x}"
            for x in self.inputs.plant_parts_ordered
        }
        # we're going to add a column level_deprish, which indicates the
        # relative size of the plant part granularity between deprecaition and
        # ferc1
        candidate_matches_all = (
            candidate_matches_all.assign(
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
        candidate_matches_all = pd.merge(
            candidate_matches_all,
            candidate_matches_all.groupby(['record_id_eia_deprish'])
            .agg({'record_id_ferc1': 'count'})
            .rename(columns={'record_id_ferc1': 'count_ferc1', })
            .reset_index(),
        )
        return candidate_matches_all

    def get_same_true(self, candidate_matches):
        """
        Grab the obvious matches which have the same record_id_eia.

        If an candidation match has the same id from both the depreciation
        records and ferc1.... then we have found a match.

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the matches
                of possible matches between the ferc1 and deprecation.

        Returns:
            pandas.DataFrame: dataframe of matches that have the same EIA
                record id

        """
        return candidate_matches[
            candidate_matches.record_id_eia_deprish ==
            candidate_matches.record_id_eia_ferc1]

    def get_matches_at_diff_ownership(self, candidate_matches):
        """
        Get matches when the record_id_eia matches except for the ownership.

        The master unit list includes various levels of ownership associated
        with each record. Some are labeled "owned" (for the utilty's owned
        portion of the plant part) and some are labeled "total" (for the full
        plant part).

        The method selects the matches where the potential matches have the
        same EIA record id expect for the ownership level (owned vs total).

        Note: Is there a cleaner way to do this??

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the matches
                of possible matches between the ferc1 and deprecation.
        """
        diff_own = (
            candidate_matches[
                candidate_matches.record_id_eia_deprish.str.replace(
                    '(owned|total)', "", regex=True) ==
                candidate_matches.record_id_eia_ferc1.str.replace(
                    '(owned|total)', "", regex=True)
            ]
        )
        return diff_own

    def get_only_ferc1_matches(self, candidate_matches):
        """
        Get the matches when there is only one FERC1 match.

        If there is only one optioin from FERC1, then we have no other choices,
        so we are going to assume the one match is the right match. We've
        alredy generated a `count_ferc1` column in `get_candidate_matches`, so
        we can use that here.

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the canidate
                matches between the ferc1 and depreciation.
        """
        return candidate_matches[
            (candidate_matches.count_ferc1 == 1)
            & (candidate_matches.plant_part_ferc1.notnull())
        ]

    def get_matches_same_qualifiers_ids_by_source(
            self, candidate_matches, source):
        """
        Get matches that have the same qualifier ids part level by source.

        See `get_matches_same_qualifiers_ids` for details.

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the canidate
                matches between the ferc1 and depreciation.
            source (string): either `ferc1` or `deprish`
        """
        df = pd.DataFrame()
        for part_name in self.inputs.plant_parts_ordered:
            part_df = candidate_matches[
                candidate_matches[f"plant_part_{source}"] == part_name]
            # add the slice of the df for this part if the id columns for
            # both contain the same values
            df = df.append(part_df[
                part_df[f"{self.inputs.parts_to_ids[part_name]}_deprish"]
                ==
                part_df[f"{self.inputs.parts_to_ids[part_name]}_ferc1"]])
        df = df.assign(match_method=f"same_qualifiers_{source}")
        return df

    def get_matches_same_qualifiers_ids(self, candidate_matches):
        """
        Get the matches that have the same plant part id columns.

        The master unit list is make up of records which correspond to
        different levels or granularies of plants. These records are all
        cobbled together at different levels from generator records. When a
        certain record is compiled from generators that have a consistent
        identifyier for another plant part, that record is associated with the
        record (ie. if a prime move level record was compiled from generators
        that all have the same unit id).

        Each of the datasets that we are connecting are now associated with
        master unit list records. If one dataset reports at the level of a
        prime mover and another dataset reports at the level of units, when
        the prime mover of the prime mover record matches the prime mover of
        the units, then these units are parts of that prime mover and we can
        associate them together.

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the canidate
                matches between the ferc1 and depreciation.
        """
        # we are going to check if things are consistent from both "directions"
        # meaning from each of our data sets
        same_quals = pd.DataFrame()
        for source in ['ferc1', 'deprish']:
            same_quals = same_quals.append(
                self.get_matches_same_qualifiers_ids_by_source(
                    candidate_matches, source))
        return same_quals

    def remove_matches_from_candidates(self,
                                       candidate_matches_current, matches_df):
        """
        Remove the matches from the candidate matches.

        Because we are iteratively generating matches, we want to remove
        the matches we've determined from the candidate matches for future
        iterations.

        Args:
            candidate_matches (pandas.DataFrame): dataframe of the canidate
                matches between the ferc1 and depreciation.
            matches_df (pandas.DataFrame): the known matches which
                will be removed as possibilities for future matches
        Returns:
            pandas.DataFrame:
        """
        # if the record_id_eia_deprish shows up in the candidates, remove it
        # the only candidates left are the ones that do not show up in the
        # matches
        candidate_matches_remainder = candidate_matches_current[
            ~candidate_matches_current.record_id_eia_deprish
            .isin(matches_df.record_id_eia_deprish.unique())]
        return candidate_matches_remainder

    def match(self):
        """
        Connect depreciation records with ferc1 steam records.

        TODO: This is a big messy WIP function right now. The returns below is
        where this is going, not what is currently happening...

        Returns:
            pandas.DataFrame: a dataframe of records from the depreciation data
            with ferc1 steam data associated in a many to many relationship.
        """
        # matches are known to be connected records and candidate matches are
        # the options for matches
        candidate_matches_all = self.get_candidate_matches()

        # we are going to go through various methods for grabbing the true
        # matches out of the candidate matches. we will then label those
        # candidate matches with a match_method column. we are going to
        # continually remove the known matches from the candidates

        methods = {
            "same_true": self.get_same_true,
            "same_diff_own": self.get_matches_at_diff_ownership,
            "same_quals": self.get_matches_same_qualifiers_ids,
            "one_ferc1_opt": self.get_only_ferc1_matches,
        }
        # compile the connected dfs in a dictionary
        matches_dfs = {}
        candidate_matches = deepcopy(candidate_matches_all)
        for method in methods:
            connects_method_df = (
                methods[method](candidate_matches)
                .assign(match_method=method)
            )
            logger.info(
                f"Matches for {method}:   {len(connects_method_df)}")
            candidate_matches = self.remove_matches_from_candidates(
                candidate_matches, connects_method_df)
            matches_dfs[method] = connects_method_df
        # squish all of the known matches together
        matches_df = pd.concat(matches_dfs.values())

        ###########################################
        # everything below here is just for logging
        ###########################################
        ids_to_match = (candidate_matches_all[
            candidate_matches_all.record_id_eia_ferc1.notnull()]
            .record_id_eia_deprish.unique())
        ids_connected = matches_df.record_id_eia_deprish.unique()
        ids_no_match = (candidate_matches[
            candidate_matches.record_id_eia_ferc1.notnull()]
            .plant_id_pudl.unique())
        logger.info("Portion of unique depreciation records:")
        logger.info(
            f"    Matched:   {len(ids_connected)/len(ids_to_match):.02%}")
        logger.info(
            f"    No link:   {len(ids_no_match)/len(ids_to_match):.02%}")
        logger.info(f"""Connected:
{matches_df.match_method.value_counts(dropna=False)}""")
        logger.info(f"""Connection Levels:
{matches_df.level_deprish.value_counts(dropna=False)}""")
        logger.debug(f"""Only one ferc1 match levels:
{matches_dfs['one_ferc1_opt'].level_deprish.value_counts(dropna=False)}""")

        # for debugging return all outputs.. remove when this all feels stable
        # return candidate_matches_all, candidate_matches, matches_df
        self.matches_df = matches_df
        return self.matches_df


class Scaler(object):
    """Scales FERC1 data to matching depreciation records."""

    def __init__(self, match_maker):
        """
        Initialize Scaler.

        Args:
            match_maker (instance): instance of MatchMaker. If `matches_df` has
                not been generated, then `match()` will be run in this method.
        """
        try:
            self.matches_df = deepcopy(match_maker.matches_df)
        except AttributeError:
            self.matches_df = deepcopy(match_maker.match())

    def scale(self):
        """
        WIP. Scale ferc1 data to the depreciation records.

        Note: the following return is the aspirational desire for where this
        method is going.

        Returns:
            pandas.DataFrame: FERC1 steam data has been either aggregated
            or disaggregated to match the level of the depreciation records.
            The data columns properly scaled will be labled as
            "{data_col}_ferc1_deprish".
        """
        self.scale_by_ownership_fraction()
        same_true = self.assign_ferc1_data_cols_same()
        same_smol = self.split_ferc1_data_cols()
        same_beeg = self.agg_ferc_data_cols()
        # TODO: Add the remaining scaling methods
        scaled_df = pd.concat([same_true, same_smol, same_beeg])

        self.test_same_true_fraction_owned(same_true)
        return scaled_df

    def scale_by_ownership_fraction(self):
        """
        Scale the data columns by the fraction owned ratio in matches_df.

        This method makes new columns in matches_df for each of the data
        columns in `DATA_COLS_TO_SPLIT` that scale the ferc1 data based on the
        ownership fractions of the depreciation and ferc1 records.
        """
        for data_col in DATA_COLS_TO_SPLIT:
            self.matches_df.loc[:, f"{data_col}_own_frac"] = (
                self.matches_df[data_col]
                * (self.matches_df.fraction_owned_deprish
                   / self.matches_df.fraction_owned_ferc1)
            )

    def assign_ferc1_data_cols_same(self):
        """
        Assign FERC1 data cols to deprecation when records are the same.

        For matched depreciation and FERC1 records are exactly the same, this
        method simply assigns the original data column to the new associated
        data column (with the format of "{data_col}_ferc1_deprish"). This
        method scales ferc1 data for both the `same_true` match method and the
        `same_diff_own` method - it uses the scaled `{data_col}_own_frac`
        generated in `scale_by_ownership_fraction()`.

        Relies on:
        * matches_df (pandas.DataFrame): a dataframe of records from the
            depreciation data with ferc1 steam data associated in a many to
            many relationship.
        """
        same_true = deepcopy(
            self.matches_df.loc[
                (self.matches_df.match_method == 'same_true')
                | (self.matches_df.match_method == 'same_diff_own')
            ]
        )
        for data_col in DATA_COLS_TO_SPLIT:
            new_data_col = self._get_clean_new_data_col(data_col)
            same_true.loc[:, new_data_col] = (
                same_true[f"{data_col}_own_frac"]
            )
        return same_true

    def split_ferc1_data_cols(self):
        """
        Split and assign portions of FERC1 columns to depreciation records.

        For each FERC1 data column that we want to associated with depreciation
        records, this method splits the FERC1 data based on a prioritized list
        of columns to weight and split the FERC1 data on.

        Relies on:
        * matches_df (pandas.DataFrame): a dataframe of records from the
            depreciation data with ferc1 steam data associated in a many to
            many relationship.
        """
        # get the records that are matches with the same qualifier records and
        # the deprecation records are at a smaller level than the FERC1 records
        same_smol = self.matches_df.loc[
            (self.matches_df.match_method == 'same_quals')
            & (self.matches_df.level_deprish == 'smol')
        ]

        idx_cols_ferc1 = ['plant_id_pudl', 'record_id_eia_ferc1']
        # add a count for the nuber of depreciation records that match to each
        # ferc1 record
        df_fgb = (
            same_smol
            .groupby(by=idx_cols_ferc1)
            [['record_id_eia_deprish']].count()
            .rename(columns={
                'record_id_eia_deprish': 'record_count_matches_deprish'})
            .reset_index()
        )

        same_smol = pd.merge(same_smol, df_fgb)

        for data_col, split_cols in DATA_COLS_TO_SPLIT.items():
            same_smol = self.split_ferc1_data_on_split_cols(
                same_smol,
                data_col=f"{data_col}_own_frac",
                idx_cols_ferc1=idx_cols_ferc1,
                split_cols=split_cols,
            )
        return same_smol

    def split_ferc1_data_on_split_cols(self,
                                       same_smol,
                                       data_col,
                                       idx_cols_ferc1,
                                       split_cols,):
        """
        Split larger ferc1 records porportionally by depreciation columns.

        This method associates slices of ferc1 records - which are larger than
        their depreciation counter parts - via prioritized columns.

        Args:
            same_smol (pandas.DataFrame): table with matched records from ferc1
                and depreciation records.
            data_col (string): name of the ferc1 data column.
            idx_cols_ferc1 (list): columns to group by.
            split_cols (list): ordered list of columns to split porportionally
                based on. Ordered based on priority: if non-null result from
                frist column, result will include first column result, then
                second and so on.
        Returns:
            pandas.DataFrame: a modified version of `same_smol` with a new
                assigned data_col

        """
        # we want to know the sum of the potential split_cols for each ferc1
        # option
        df_fgb = (
            same_smol.loc[:, idx_cols_ferc1 + split_cols]
            .fillna(0)  # remove w/ pandas 1.1
            .groupby(by=idx_cols_ferc1)  # add dropna=False w/ pandas 1.1
            .sum()
            .reset_index()
        )
        df_w_tots = (
            pd.merge(same_smol, df_fgb,
                     on=idx_cols_ferc1,
                     suffixes=("", "_fgb"))
        )
        # for each of the columns we want to split the frc data by
        # generate the % of the total group, so we can split the data_col
        new_data_col = self._get_clean_new_data_col(data_col)
        df_w_tots[new_data_col] = pd.NA
        for split_col in split_cols:
            df_w_tots[f"{split_col}_pct"] = (
                df_w_tots[split_col] / df_w_tots[f"{split_col}_fgb"])
            # choose the first non-null option.
            df_w_tots[new_data_col] = (
                df_w_tots[new_data_col].fillna(
                    df_w_tots[data_col] * df_w_tots[f"{split_col}_pct"]))

        # merge in the newly generated split/assigned data column
        df_final = pd.merge(
            same_smol,
            df_w_tots[idx_cols_ferc1 + ['record_id_eia_deprish',
                                        new_data_col]].drop_duplicates(),
        )
        return df_final

    def agg_ferc_data_cols(self):
        """
        Aggregate smaller level FERC1 data columns to depreciation records.

        When the depreciation matches are at a higher level than the FERC1,
        this method aggregates the many FERC1 records into the level of the
        depreciation records.
        """
        data_cols_own_frac = [f"{col}_own_frac" for col in DATA_COLS_TO_SPLIT]
        same_beeg = self.matches_df.loc[
            (self.matches_df.match_method == 'same_quals')
            & (self.matches_df.level_deprish == 'beeg')
        ]

        # dict to rename the summed data columns to their new name
        rename_dict = {}
        for data_col_of, data_col in zip(data_cols_own_frac,
                                         DATA_COLS_TO_SPLIT):
            rename_dict[data_col_of] = self._get_clean_new_data_col(data_col)
        # sum the data columns at the level of the depreciation records
        same_beeg_sum = (
            same_beeg.groupby('record_id_eia_deprish')
            [data_cols_own_frac]
            .sum()
            .reset_index()
            .rename(columns=rename_dict)
        )

        # squish the new summed columns back into the og df
        same_beeg = pd.merge(same_beeg,
                             same_beeg_sum,
                             on=['record_id_eia_deprish'],
                             how='left')
        return same_beeg

    def _get_clean_new_data_col(self, data_col):
        # some of the data columns already have a ferc1 suffix because the same
        # column name also shows up in the EIA data... so we want to remove the
        # double ferc1 if it shows up
        new_data_col = f"{data_col}_ferc1_deprish"
        new_data_col = new_data_col.replace("ferc1_ferc1", "ferc1")
        return new_data_col

    def test_same_true_fraction_owned(self, same_true):
        """
        Test the same_true scaling.

        Raises:
            AssertionError: If there is some error in the standard scaling with
            fraction_owned columns.
        """
        for data_col in DATA_COLS_TO_SPLIT:
            new_data_col = self._get_clean_new_data_col(data_col)
            not_same = same_true[
                (same_true.match_method == 'same_true')
                & ~(same_true[data_col] == same_true[new_data_col])
                & (same_true[data_col].notnull())
                & (same_true[new_data_col].notnull())
            ]
            if not not_same.empty:
                raise AssertionError(
                    "Scaling for same_true match method errored with "
                    f"{len(not_same)}. Check fraction owned split in "
                    "scale_by_ownership_fraction."
                )
            else:
                logger.debug(
                    f"testing the same_true match method passed for {data_col}"
                )
