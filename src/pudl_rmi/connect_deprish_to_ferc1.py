"""
Connect the depreciation data with FERC1 steam plant records.

This module attempts to connect the depreciation data with FERC1 steam records.
Both the depreciation records and FERC1 steam has been connected to the EIA
master unit list, which is a compilation of various possible combinations of
generator records.

Matches are determined to be correct record linkages.
Candidate matches are potential matches.

Inputs:
* A
"""

import logging
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

import pudl_rmi.connect_deprish_to_eia as connect_deprish_to_eia
# from pudl_rmi.connect_deprish_to_eia import PPL_COLS
import pudl

logger = logging.getLogger(__name__)

CONNECT_COLS = [
    'plant_id_pudl',
    # 'utility_id_pudl',
    'report_year'
]

SPLIT_COLS_STANDARD = [
    'total_fuel_cost',
    'net_generation_mwh',
    'capacity_mw',
]
"""
list: the standard columns to split ferc1 data columns to be used in
``DATA_COLS_TO_SPLIT``.
"""

DATA_COLS_TO_SPLIT = {
    'opex_nonfuel': SPLIT_COLS_STANDARD,
    'net_generation_mwh_ferc1': SPLIT_COLS_STANDARD,
    'capex_total': SPLIT_COLS_STANDARD,
    'capex_annual_addt': SPLIT_COLS_STANDARD,
}
"""
dictionary: FERC1 data columns (keys) that we want to associated with
depreciation records. When the FERC1 record is larger than the depreciation
record (e.g. a group of depreciation generators matched with a FERC1 plant),
this module attemps to split the depreciation record based on the list of
columns to weight the split (values). See  ``split_ferc1_data_on_split_cols()``
for more details.
"""

IDX_STEAM = ['utility_id_ferc1', 'plant_id_ferc1', 'report_date']
IDX_COLS_FERC1 = ['plant_id_pudl', 'record_id_eia_ferc1']


{
    'plant_id_eia': {1, 1, 1, 2, 2, 3},
    'generator_id': {'1a', '1b', '1c', '2a', '2b', 'a'},
    'report_date': {
        '2020-01-01',
        '2020-01-01',
        '2020-01-01',
        '2020-01-01',
        '2020-01-01',
        '2020-01-01'},
    'operational_status_pudl': {
        'operating',
        'operating',
        'operating',
        'operating',
        'operating',
        'operating'},
    'utility_id_eia': {1, 1, 1, 1, 1, 1},
    'ownership': {'total',
                  'total',
                  'total',
                  'total',
                  'total',
                  'total'},
    'record_id_eia': {
        'record_id_1_1a',
        'record_id_1_1b',
        'record_id_1_1c',
        'record_id_2_2a',
        'record_id_2_2b',
        'record_id_3_a'},
    'record_id_test': {
        'record_1',
        'record_1',
        'record_1',
        'record_2',
        'record_2',
        'record_3'},
    'record_id_eia_og': {
        'record_id_1_1a_1b_1c',
        'record_id_1_1a_1b_1c',
        'record_id_1_1a_1b_1c',
        'reocrd_id_2_2a_2b',
        'reocrd_id_2_2a_2b',
        'record_id_3_a'},
    'data_col': {300, 300, 300, 100, 100, 10},
    'index': {0, 1, 2, 3, 4, 5},
    'plant_part': {
        'plant_gen',
        'plant_gen',
        'plant_gen',
        'plant_gen',
        'plant_gen',
        'plant_gen'},
    'ferc_acct_name': {'steam', 'steam', 'steam', 'steam', 'steam', 'steam'},
    'unit_id_pudl': {pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, },
    'technology_description': {
        'Conventional Steam Coal',
        'Conventional Steam Coal',
        'Conventional Steam Coal',
        'Conventional Steam Coal',
        'Conventional Steam Coal',
        'Conventional Steam Coal'},
    'prime_mover_code': {'ST', 'ST', 'ST', 'ST', 'ST', 'ST'},
    'energy_source_code_1': {'BIT', 'BIT', 'BIT', 'BIT', 'BIT', 'BIT'},
    'capacity_mw': {50, 30, 20, 10, 10, 100},
    'net_generation_mwh': {10000, 1000, 100, 250, 200, 5000},
    'total_fuel_cost': {500, 100, 50, 250, 200, 300},
    'data_col_scaled': {150, 90, 60, 50, 50, 10}}


def str_squish(x):
    """Squish strings from a groupby into a list."""
    return '; '.join(list(map(str, [x for x in x.unique() if x is not pd.NA])))


def pre_aggregate(
        df_to_scale: pd.DataFrame,
        ppl: pd.DataFrame,
        data_set_idx_cols: list,
        data_cols: list,
        other_cols_to_keep: list) -> pd.DataFrame:
    """
    Aggregate.

    TODO:
    * aggregate only the duplicate records and just concat
    """
    all_cols = data_set_idx_cols + data_cols + other_cols_to_keep
    all_sum_cols = [
        # ferc cols
        'capex_total',
        'capex_annual_addt',
        'opex_nonfuel',
        'capacity_mw_ferc1',
        # deprish cols
        'plant_balance_w_common',
        'book_reserve_w_common',
        'unaccrued_balance_w_common',
        'net_salvage_w_common',
        'depreciation_annual_epxns_w_common'
    ]
    all_wtavg_dict = {
        # ferc cols
        'avg_num_employees': 'capacity_mw_ferc1',
        # deprish cols
        'net_removal_rate': 'unaccrued_balance_w_common',
        'depreciation_annual_rate': 'unaccrued_balance_w_common',
        'remaining_life_avg': 'unaccrued_balance_w_common',
    }
    all_str_cols = [
        'record_id_ferc1',
        'line_id',
        'utility_name_ferc1',
    ]

    sum_cols = [c for c in all_cols if c in all_sum_cols]
    wtavg_dict = {k: v for (k, v) in all_wtavg_dict.items() if k in all_cols}
    str_cols = [c for c in all_cols if c in all_str_cols]

    # if len(df_to_scale.filter(like='_scaled').columns) > 0:
    #     logger.info("Post-scale aggregate.")
    #     sum_cols = [f"{x}_scaled" for x in sum_cols]
    #     wtavg_dict = {
    #         k: f"{v}_scaled" for (k, v) in wtavg_dict.items()
    #     }
    # else:
    #     logger.info("Pre-scale aggregate.")
    by_data_set_cols = ['data_source']
    by = ['record_id_eia'] + [x for x in by_data_set_cols if x in df_to_scale]
    # sum and weighted average!
    df_out = pudl.helpers.sum_and_weighted_average_agg(
        df_in=df_to_scale,
        by=by,
        sum_cols=sum_cols,
        wtavg_dict=wtavg_dict
    )
    # add in the string columns
    df_out = df_out.merge(
        (
            df_to_scale
            .groupby(by, as_index=False)
            .agg({k: str_squish for k in str_cols})
        ),
        on=by,
        validate='1:1',
        how='left'
    ).pipe(pudl.helpers.convert_cols_dtypes, 'eia')

    # merge back in the ppl idx columns
    df_w_ppl = (
        df_out.set_index('record_id_eia')
        .merge(
            ppl,  # [[c for c in PPL_COLS if c != 'record_id_eia']],
            left_index=True,
            right_index=True,
            how='left',
            validate='m:1',
        )
        .reset_index()
    )
    return df_w_ppl


def many_merge_on_scale_part(
        plant_part: str,
        df_to_scale: pd.DataFrame,
        cols_to_keep: list,
        ppl: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a particular EIA plant-part list plant-part onto a dataframe.

    Returns:
        a table.
    """
    ppl_part_df = ppl[ppl.plant_part == plant_part]
    # convert the date to year start
    df_to_scale.loc[:, 'report_date'] = (
        pd.to_datetime(df_to_scale.report_date.dt.year, format='%Y')
    )
    scale_parts = []
    for plant_part in pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED:
        idx_part = (
            pudl.analysis.plant_parts_eia.PLANT_PARTS[plant_part]['id_cols']
            + pudl.analysis.plant_parts_eia.IDX_TO_ADD
            + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
        )
        # grab just the part of the df that cooresponds to the plant_part
        part_df = pd.merge(
            (
                df_to_scale[df_to_scale.plant_part == plant_part]
                [idx_part + ['record_id_eia'] + cols_to_keep]
            ),
            ppl_part_df,
            on=idx_part,
            how='left',
            validate='m:m',
            suffixes=('_og', '')
        )
        scale_parts.append(part_df)
    scale_parts_df = pd.concat(scale_parts)
    return scale_parts_df


def split_data_on_split_cols(
        df_to_scale: pd.DataFrame,
        merge_cols: list,
        data_col: str,
        split_cols: list) -> pd.DataFrame:
    """
    Split larger dataset records porportionally by EIA plant-part list columns.

    This method associates slices of a dataset's records - which are larger
    than their EIA counter parts - via prioritized EIA plant-part list columns.

    Args:
        df_to_scale (pandas.DataFrame): table of data that has been merged with
            the EIA plant-part list records of the scale that you want the
            output to be in.
        data_col (string): name of the ferc1 data column.
        merge_cols (list): columns to group by.
        split_cols (list): ordered list of columns to split porportionally
            based on. Ordered based on priority: if non-null result from
            frist column, result will include first column result, then
            second and so on.
    Returns:
        pandas.DataFrame: a modified version of `same_smol` with a new
            assigned data_col

    """
    df_gb = (
        df_to_scale.loc[:, split_cols]
        .groupby(by=merge_cols, dropna=False)
        .sum(min_count=1)
    )
    df_w_tots = (
        pd.merge(
            df_to_scale,
            df_gb,
            right_index=True,
            left_index=True,
            suffixes=("", "_fgb")
        )
    )
    # for each of the columns we want to split the frc data by
    # generate the % of the total group, so we can split the data_col
    new_data_col = f"{data_col}_scaled"
    df_w_tots[new_data_col] = pd.NA
    for split_col in split_cols:
        df_w_tots[f"{split_col}_pct"] = (
            df_w_tots[split_col] / df_w_tots[f"{split_col}_fgb"])
        # choose the first non-null option.
        df_w_tots[new_data_col] = (
            df_w_tots[new_data_col].fillna(
                df_w_tots[data_col] * df_w_tots[f"{split_col}_pct"]))
    return df_w_tots[[new_data_col]]


def scale_to_plant_part(
        scale_part: str,
        df_to_scale: pd.DataFrame,
        ppl: pd.DataFrame,
        data_set_idx_cols: list,
        data_cols: list,
        other_cols_to_keep: list,
        skip_agg: bool = False
) -> pd.DataFrame:
    """
    hi.

    TODO:
     * only aggregate the actually duplicated records
    """
    if skip_agg:
        pass
    else:
        df_to_scale = pre_aggregate(
            df_to_scale=df_to_scale,
            ppl=ppl,
            data_set_idx_cols=data_set_idx_cols,
            data_cols=data_cols,
            other_cols_to_keep=other_cols_to_keep
        )
    logger.info(len(df_to_scale))
    merged_df = many_merge_on_scale_part(
        plant_part=scale_part,
        df_to_scale=df_to_scale,
        cols_to_keep=data_cols + data_set_idx_cols + other_cols_to_keep,
        ppl=ppl.reset_index()
    )
    logger.info(len(merged_df))
    # grab all of the ppl columns, plus data set's id column(s)
    # this enables us to have a unique index
    idx_cols = (
        pudl.analysis.plant_parts_eia.PLANT_PARTS[scale_part]['id_cols']
        + pudl.analysis.plant_parts_eia.IDX_TO_ADD
        + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
        + data_set_idx_cols
    )
    scaled_df = merged_df.set_index(idx_cols)
    for data_col in data_cols:
        scaled_df.loc[:, f"{data_col}_scaled"] = split_data_on_split_cols(
            df_to_scale=scaled_df,
            merge_cols=data_set_idx_cols,
            data_col=data_col,
            split_cols=[
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ]
        )
    logger.info(len(scaled_df))
    scaled_df = (
        scaled_df.drop(columns=data_cols)
        .rename(columns={
            c: c.replace('_scaled', '') for c in [c for c in scaled_df.columns if "_scaled" in c]}))
    if skip_agg:
        scaled_df_post_agg = (
            scaled_df.reset_index(data_set_idx_cols)
            .set_index(['record_id_eia'], append=True))
    else:
        scaled_df_post_agg = pre_aggregate(
            df_to_scale=scaled_df.reset_index(),
            ppl=ppl,
            data_set_idx_cols=data_set_idx_cols,
            data_cols=data_cols,
            other_cols_to_keep=other_cols_to_keep
        )
        scaled_df_post_agg = (
            scaled_df_post_agg.set_index(idx_cols + ['record_id_eia'])
            .reset_index(data_set_idx_cols)
        )
    logger.info(len(scaled_df_post_agg))
    return scaled_df_post_agg


######################
# OLD ################
######################

def execute(plant_parts_eia, deprish_eia, ferc1_to_eia, clobber=False):
    """
    Connect depreciation data to FERC1 via EIA and scale to depreciation.

    Args:
        plant_parts_eia (pandas.DataFrame): EIA plant-part list - table of
            "plant-parts" which are groups of aggregated EIA generators
            that coorespond to portions of plants from generators to fuel
            types to whole plants.
        deprish_eia (pandas.DataFrame): table of the connection between the
            depreciation studies and the EIA plant-parts list.
        ferc1_to_eia (pandas.DataFrame): a table of the connection between
            the FERC1 plants and the EIA plant-parts list.
        clobber (boolean):
    """
    inputs = InputsManager(
        plant_parts_eia=plant_parts_eia,
        deprish_eia=deprish_eia,
        ferc1_to_eia=ferc1_to_eia
    )

    scaler = Scaler(MatchMaker(inputs))
    scaled_df = scaler.scale()
    return scaled_df


class InputsManager():
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

    def __init__(
        self,
        plant_parts_eia,
        deprish_eia,
        ferc1_to_eia
    ):
        """
        Initialize input manager for connecting depreciation to FERC1.

        Args:
            plant_parts_eia (pandas.DataFrame): EIA plant-part list - table of
                "plant-parts" which are groups of aggregated EIA generators
                that coorespond to portions of plants from generators to fuel
                types to whole plants.
            deprish_eia (pandas.DataFrame): table of the connection between the
                depreciation studies and the EIA plant-parts list.
            ferc1_to_eia (pandas.DataFrame): a table of the connection between
                the FERC1 plants and the EIA plant-parts list.
        """
        self.plant_parts_eia = plant_parts_eia
        self.deprish_eia = deprish_eia
        self.connects_ferc1_eia = ferc1_to_eia
        # store a bool which will indicate whether the inputs have been prepped
        self.inputs_prepped = False

    def _prep_plant_parts_eia(self):
        self.plant_parts_eia = (
            self.plant_parts_eia.reset_index()
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']))

    def _prep_connects_deprish_eia(self):
        self.connects_deprish_eia = (
            # we only want candidate matches that have actually been connected
            # to the MUL
            self.deprish_eia[
                self.deprish_eia.record_id_eia.notnull()]
            .pipe(pudl.helpers.convert_to_date)
            .astype({i: pd.Int64Dtype() for i in
                     ['plant_id_eia', 'utility_id_pudl', 'utility_id_ferc1']}
                    )
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
        )
        # we are going to merge the master unit list into this output,
        # because we want all of the id columns from PPL_COLS.
        # There are some overlapping columns. We really only want to
        # merge on the 'record_id_eia' and we trust the master unit list
        # more than the spreadsheet based connection for deprish to eia
        # so we are going to only use columns from the deprish_to_eia that
        # don't show up in the PPL_COLS
        cols_ppl = (
            connect_deprish_to_eia.PPL_COLS
            + ['plant_id_pudl', 'total_fuel_cost',
               'net_generation_mwh', 'capacity_mw']
        )
        cols_to_use_deprish_eia = (
            ['record_id_eia'] +
            [c for c in self.connects_deprish_eia.columns
             if c not in cols_ppl])

        self.connects_deprish_eia = pd.merge(
            self.connects_deprish_eia[cols_to_use_deprish_eia],
            self.plant_parts_eia[
                connect_deprish_to_eia.PPL_COLS
                + ['plant_id_pudl', 'total_fuel_cost',
                   'net_generation_mwh', 'capacity_mw']],
            on=['record_id_eia']
        )

    def _prep_connects_ferc1_eia(self):
        self.connects_ferc1_eia = calc_annual_capital_addts_ferc1(
            self.connects_ferc1_eia)

    def prep_inputs(self, clobber=False):
        """Prepare all inputs needed for connecting depreciation to FERC1."""
        # the order here is important. We are preping the inputs needed
        # for later inputs
        if not self.inputs_prepped or clobber:
            # we need a dictionary of plant part named (keys) to their
            # corresponding id columns (values). parts_compilers has the
            # inverse of that sowe are just going to swap the ks and vs
            self.parts_to_ids = (
                pudl.analysis.plant_parts_eia.make_parts_to_ids_dict()
            )

            self._prep_plant_parts_eia()
            self._prep_connects_deprish_eia()
            self._prep_connects_ferc1_eia()

            self.inputs_prepped = True


def calc_annual_capital_addts_ferc1(steam_df, window=3):
    """
    Calculate annual capital additions for FERC1 steam records.

    Convert the capex_total column into annual capital additons the
    `capex_total` column is the cumulative capital poured into the plant over
    time. This function takes the annual difference should generate the annual
    capial additions. It also want generates a rolling average, to smooth out
    the big annual fluxuations.

    Args:
        steam_df (pandas.DataFrame): result of `prep_plants_ferc()`

    Returns:
        pandas.DataFrame: augemented version of steam_df with two additional
        columns: `capex_annual_addt` and `capex_annual_addt_rolling`.
    """
    # we need to sort the df so it lines up w/ the groupby
    steam_df = steam_df.sort_values(IDX_STEAM)
    # we group on everything but the year so the groups are multi-year unique
    # plants the shift happens within these multi-year plant groups
    steam_df['capex_total_shifted'] = steam_df.groupby(
        [x for x in IDX_STEAM if x != 'report_date'])[['capex_total']].shift()
    steam_df = steam_df.assign(
        capex_annual_addt=lambda x: x.capex_total - x.capex_total_shifted
    )

    addts = pudl.helpers.generate_rolling_avg(
        steam_df,
        group_cols=[x for x in IDX_STEAM if x != 'report_date'],
        data_col='capex_annual_addt',
        window=window
    ).pipe(pudl.helpers.convert_cols_dtypes, 'ferc1')

    steam_df_w_addts = (
        pd.merge(
            steam_df,
            addts[IDX_STEAM + ['capex_total', 'capex_annual_addt_rolling']],
            on=IDX_STEAM + ['capex_total'],
            how='left',
        )
        .assign(
            capex_annual_per_mwh=lambda x:
                x.capex_annual_addt / x.net_generation_mwh_ferc1,
            capex_annual_per_mw=lambda x:
                x.capex_annual_addt / x.capacity_mw_ferc1,
            capex_annual_per_kw=lambda x:
                x.capex_annual_addt / x.capacity_mw_ferc1 / 1000,
            capex_annual_per_mwh_rolling=lambda x:
                x.capex_annual_addt_rolling / x.net_generation_mwh_ferc1,
            capex_annual_per_mw_rolling=lambda x:
                x.capex_annual_addt_rolling / x.capacity_mw_ferc1,
        )
    )

    steam_df_w_addts = add_mean_cap_addts(steam_df_w_addts)
    # bb tests for volumne of negative annual capex
    neg_cap_addts = len(
        steam_df_w_addts[steam_df_w_addts.capex_annual_addt_rolling < 0]) \
        / len(steam_df_w_addts)
    neg_cap_addts_mw = (
        steam_df_w_addts[
            steam_df_w_addts.capex_annual_addt_rolling < 0]
        .net_generation_mwh_ferc1.sum()
        / steam_df_w_addts.net_generation_mwh_ferc1.sum())
    message = (f'{neg_cap_addts:.02%} records have negative capitial additions'
               f': {neg_cap_addts_mw:.02%} of capacity')
    if neg_cap_addts > .1:
        warnings.warn(message)
    else:
        logger.info(message)
    return steam_df_w_addts


def add_mean_cap_addts(steam_df):
    """Add mean capital additions over lifetime of plant (via `IDX_STEAM`)."""
    idx_steam_no_date = [c for c in IDX_STEAM if c != 'report_year']
    gb_cap_an = steam_df.groupby(idx_steam_no_date)[['capex_annual_addt']]
    # calcuate the standard deviatoin of each generator's capex over time
    df = (
        steam_df
        .merge(
            gb_cap_an.std().add_suffix('_gen_std').reset_index().pipe(
                pudl.helpers.convert_cols_dtypes, 'ferc1'),
            how='left',
            on=idx_steam_no_date,
            validate='m:1'  # should this really be 1:1?
        )
        .merge(
            gb_cap_an.mean().add_suffix('_gen_mean').reset_index().pipe(
                pudl.helpers.convert_cols_dtypes, 'ferc1'),
            how='left',
            on=idx_steam_no_date,
            validate='m:1'  # should this really be 1:1?
        )
        .assign(
            capex_annual_addt_diff_mean=lambda x: x.capex_annual_addt - \
            x. capex_annual_addt_gen_mean,
        )
    )
    return df


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
        not_in_ferc1_plant_ids = (
            candidate_matches_all[candidate_matches_all._merge == 'left_only']
            .plant_id_pudl.unique()
        )
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
        # add dtype enforcer bc OMIGOSH they keep defauliting to non-nullables
        candidate_matches_all = pd.merge(
            self.inputs.connects_deprish_eia.pipe(
                pudl.helpers.convert_cols_dtypes, 'ferc1'),
            self.inputs.connects_ferc1_eia.pipe(
                pudl.helpers.convert_cols_dtypes, 'ferc1'),
            on=CONNECT_COLS,
            suffixes=('_deprish', '_ferc1'),
            how='left', indicator=True
        )
        # reorder cols so they are easier to see, maybe remove later
        first_cols = ['plant_part_deprish', 'plant_part_ferc1',
                      'record_id_eia_deprish', 'record_id_eia_ferc1',
                      'plant_part_name', 'plant_name_match',
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
            f"{pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED.index(x)}_{x}"
            for x in pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED
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
        for part_name in pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED:
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

        TODO: Add explicit arguments for the input matches_df into methods
        instead of accessing the cached df.

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
        logger.info(f"Scaled via same/true: {len(same_true)}")
        logger.info(f"Scaled via same/smol: {len(same_smol)}")
        logger.info(f"Scaled via same/beeg: {len(same_beeg)}")
        scaled_df = pd.concat([same_true, same_smol, same_beeg])

        self.test_same_true_fraction_owned(same_true)
        logger.info(
            f"output is len {len(scaled_df)} while input was "
            f"{len(self.matches_df)}"
        )
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
            new_data_col = _get_clean_new_data_col(data_col)
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

        # add a count for the nuber of depreciation records that match to each
        # ferc1 record
        same_smol.loc[:, 'record_count_matches_deprish'] = (
            same_smol
            .groupby(by=IDX_COLS_FERC1)
            [['record_id_eia_deprish']].transform('count')
        )

        for data_col, split_cols in DATA_COLS_TO_SPLIT.items():
            same_smol = split_ferc1_data_on_split_cols(
                same_smol,
                data_col=f"{data_col}_own_frac",
                split_cols=split_cols,
            )
        return same_smol

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
            rename_dict[data_col_of] = _get_clean_new_data_col(data_col)
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

    def test_same_true_fraction_owned(self, same_true):
        """
        Test the same_true scaling.

        Raises:
            AssertionError: If there is some error in the standard scaling with
            fraction_owned columns.
        """
        for data_col in DATA_COLS_TO_SPLIT:
            new_data_col = _get_clean_new_data_col(data_col)
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


def split_ferc1_data_on_split_cols(scale_df, merge_cols, data_col, split_cols):
    """
    Split larger ferc1 records porportionally by depreciation columns.

    This method associates slices of ferc1 records - which are larger than
    their depreciation counter parts - via prioritized columns.

    Args:
        scale_df (pandas.DataFrame): table with matched records from ferc1
            and depreciation records.
        data_col (string): name of the ferc1 data column.
        merge_cols (list): columns to group by.
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
        scale_df.loc[:, merge_cols + split_cols]
        .groupby(by=merge_cols, dropna=False)
        .sum(min_count=1)
        .reset_index()
    )
    df_w_tots = (
        pd.merge(scale_df, df_fgb,
                 on=merge_cols,
                 suffixes=("", "_fgb"))
    )
    # for each of the columns we want to split the frc data by
    # generate the % of the total group, so we can split the data_col
    new_data_col = _get_clean_new_data_col(data_col)
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
        scale_df,
        df_w_tots[
            merge_cols + ['record_id_eia_deprish', new_data_col]
        ]
        .drop_duplicates(),
    )
    return df_final


def _get_clean_new_data_col(data_col):
    # some of the data columns already have a ferc1 suffix because the same
    # column name also shows up in the EIA data... so we want to remove the
    # double ferc1 if it shows up
    new_data_col = f"{data_col}_ferc1_deprish"
    new_data_col = new_data_col.replace("ferc1_ferc1", "ferc1")
    return new_data_col


def fake_duke_deprish_eia_for_mod(de):
    """Temp function to fake Duke's deprish records for modernization."""
    logger.info("Adding fake years of Duke data....")
    # omigosh bc there are double in here some how!
    if len(de.filter(like='record_id_eia_fuzzy').columns) == 2:
        de = de.drop(columns=['record_id_eia_fuzzy'])
    # cols_to_keep = [
    #     'plant_part_name', 'utility_name_ferc1', 'report_year', 'report_date',
    #     'plant_name_match', 'record_id_eia', 'line_id', 'utility_id_pudl',
    #     'data_source'
    # ]
    fake_year_dfs = []
    de_2018 = (
        de[
            de.utility_id_pudl.isin([90, 97])
            & (de.report_date.dt.year == 2018)
        ]
    )
    for fake_year in [2019, 2020]:
        de_fake_new_year = (
            de_2018.copy()
            .assign(
                report_year=fake_year,
                report_date=pd.to_datetime(fake_year, format="%Y")
            )
            .replace(
                {"record_id_eia": "_2018_",
                 "line_id": "2018_", },
                {"record_id_eia": f"_{fake_year}_",
                 "line_id": f"{fake_year}_"},
                regex=True
            )
            # [cols_to_keep]
            .reset_index()
        )
        fake_year_dfs.append(de_fake_new_year)
    de_faked = pd.concat([de] + fake_year_dfs, ignore_index=True)
    assert (~de_faked[de_faked.report_date.dt.year == 2020].empty)
    return de_faked


def append_non_plant_deprish_records(d, scaled_de):
    """Add the T&D records into the output with faked record_id_eia."""
    scaled_append = scaled_de.reset_index()
    # ensure the depreciation data does not have stray columns that aren't in
    # the deprish/EIA combo
    d_non_plant = (
        d[~d.line_id.isin(scaled_append.line_id.unique())]
        .assign(
            report_year=lambda x: x.report_date.dt.year
        )
        .pipe(fake_duke_deprish_eia_for_mod)
    )

    # make up a fake "record_id_eia" for just the T&D records
    de_w_td = (
        pd.concat([scaled_append, d_non_plant])
        .assign(
            operational_status=lambda x: np.where(
                (
                    x.record_id_eia.isnull()
                    & x.plant_part_name.notnull()
                    & x.ferc_acct_name.str.lower()
                    .isin(['distribution', 'general', 'transmission', 'intangible'])
                ),
                'existing',
                x.operational_status
            ),
        )
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        .pipe(pudl.helpers.convert_cols_dtypes, 'ferc1')
    )
    de_w_td.loc[:, 'faked_id'] = (
        de_w_td.ferc_acct_name + "_" +
        de_w_td.utility_id_pudl.astype(str) + "_" +
        de_w_td.data_source + "_" +
        de_w_td.plant_part_name + "_" +
        de_w_td.report_year.astype(str)
    )
    de_w_td.loc[:, 'record_id_eia'] = (
        de_w_td.record_id_eia.fillna(de_w_td.faked_id))
    de_w_td = (
        de_w_td.drop(columns=['plant_part_name', 'ferc_acct_name'])
        .set_index('record_id_eia')
    )
    return de_w_td
