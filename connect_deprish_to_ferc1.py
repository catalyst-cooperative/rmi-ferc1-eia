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


class DeprishToFERC1Inputs():
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

        self.prep_inputs()

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
            self.plant_parts_eia[connect_deprish_to_eia.MUL_COLS])

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
        """Prepare all inputs needed for connecting depreciation to FERC1."""
        # the order here is important. We are preping the inputs needed
        # for later inputs
        self._prep_plant_parts_eia()
        self._prep_steam_cleaned_ferc1()
        self._prep_connects_ferc1_eia()
        self._prep_connects_deprish_eia()

        self._prep_info_from_parts_compiler_eia()


class ConnectorDeprishFERC1():
    """
    Generate matches between depreciation and FERC1 steam records.

    The coordinating method in this class is `connect` - see it for additional
    details.
    """

    def __init__(self, inputs):
        """
        Initialize ConnectorDeprishFERC1.

        Store instance of DeprishToFERC1Inputs so we can use the prepared
        dataframes.

        Args:
            inputs (object): an instance of DeprishToFERC1Inputs

        """
        self.inputs = inputs

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

        Args:
            inputs (object): an instance of DeprishToFERC1Inputs

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
                      'record_id_eia_deprish', 'record_id_eia_ferc1']
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
        return candidate_matches[candidate_matches.count_ferc1 == 1]

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
            # we have to exclude the plant as a MUL level bc all of the records
            # within a plant share the plant id
            if part_name != 'plant':
                part_df = candidate_matches[
                    candidate_matches[f"plant_part_{source}"] == part_name]
                # add the slice of the df for this part if the id columns for
                # both contain the same values
                df = df.append(part_df[
                    part_df[f"{self.inputs.parts_to_ids[part_name]}_deprish"]
                    ==
                    part_df[f"{self.inputs.parts_to_ids[part_name]}_ferc1"]])
        df = df.assign(connect_method=f"same_qualifiers_{source}")
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

    def connect(self):
        """
        Connect depreciation records with ferc1 steam records.

        TODO: This is a big messy WIP function right now. The returns below is
        where this is going, not what is currently happening...

        Returns:
            pandas.DataFrame: a dataframe of unique records from the
            depreciation data with ferc1 steam data associated with as many
            records as possible. FERC1 steam data has been either aggregated
            or disaggregated to match the level of the depreciation records.
        """
        # matches are known to be connected records and candidate matches are
        # the options for matches
        candidate_matches_all = self.get_candidate_matches()

        # we are going to go through various methods for grabbing the true
        # matches out of the candidate matches. we will then label those
        # candidate matches with a connect_method column. we are going to
        # continually remove the known matches from the candidates

        methods = {
            "same_true": self.get_same_true,
            "same_diff_own": self.get_matches_at_diff_ownership,
            "one_ferc1_opt": self.get_only_ferc1_matches,
            "same_quals": self.get_matches_same_qualifiers_ids,
        }
        # compile the connected dfs in a dictionary
        matches_dfs = {}
        candidate_matches = deepcopy(candidate_matches_all)
        for method in methods:
            logger.info(f"Generating matches for {method}")
            connects_method_df = (
                methods[method](candidate_matches)
                .assign(connect_method=method)
            )
            candidate_matches = self.remove_matches_from_candidates(
                candidate_matches, connects_method_df)
            matches_dfs[method] = connects_method_df
        # squish all of the known matches together
        matches_df = pd.concat(matches_dfs.values())

        # everything below here is just for logging
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
{matches_df.connect_method.value_counts(dropna=False)}""")
        logger.info(f"""Connection Levels:
{candidate_matches.level_deprish.value_counts(dropna=False)}""")
        logger.debug(f"""Only one ferc1 match levels:
{matches_dfs['one_ferc1_opt'].level_deprish.value_counts(dropna=False)}""")

        # evertually this will be a dealt w/ squished together output
        # for now, this is a few important outputs
        return candidate_matches_all, candidate_matches, matches_df
