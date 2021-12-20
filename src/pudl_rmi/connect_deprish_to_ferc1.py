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


from typing import List, Optional, Dict
import pydantic
from pydantic import BaseModel

import pandas as pd

import pudl

logger = logging.getLogger(__name__)


META_DEPRISH_EIA: Dict = {
    'line_id':
        {
            'data_set_idx_col': True,
            'str_col': True
        },
    'plant_balance_w_common':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'book_reserve_w_common':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'unaccrued_balance_w_common':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_salvage_w_common':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'depreciation_annual_epxns_w_common':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_removal_rate':
        {
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'depreciation_annual_rate':
        {
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'remaining_life_avg':
        {
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'utility_name_ferc1':
        {
            'str_col': True
        },
    'data_source':
        {
            'str_col': True,
        }
}


META_FERC1_EIA: Dict = {
    'record_id_ferc1':
        {
            'data_set_idx_col': True,
            'str_col': True
        },
    'capex_total':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capex_annual_addt':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'opex_nonfuel':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capacity_mw_ferc1':
        {
            'sum_col': True,
            'scale_col': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },

    'avg_num_employees':
        {
            'wtavg_col': 'capacity_mw_ferc1',
        },
}


def execute(plant_parts_eia, deprish_eia, ferc1_to_eia, clobber=False):
    """
    Connect depreciation data to FERC1 via EIA and scale to depreciation.

    TODO: WIP!! The final output will live here when done.

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
    return


class FieldTreatment(BaseModel):
    """
    How to process specific a field.

    * data_set_idx_col: primary key column for input dataset
    * sum_col: a column that will be summed when aggregating
    * wtavg_col: a column that will be averaged by a weighting column
    * str_col: a column that will be aggregated by combining unique strings
    * scale_col: a column that needs to be scaled to the plant-part level
    """

    data_set_idx_col: Optional[pydantic.StrictBool] = False
    eia_idx: Optional[pydantic.StrictBool] = False
    sum_col: Optional[pydantic.StrictBool] = False
    scale_col: Optional[List] = False
    wtavg_col: Optional[str]
    str_col: Optional[pydantic.StrictBool]


class TableTreatment(BaseModel):
    """How to process a table."""

    columns: Dict[str, FieldTreatment]
    eia_pk: List[str] = ['record_id_eia']


class ScaleToPlantParter():
    """Scale to EIA plant-part."""

    def __init__(self, plant_part, metadata):
        """Initialize scaler of df to plant-part."""
        self.metadata: TableTreatment = metadata
        self.plant_part: str = plant_part
        self.eia_pk: List = metadata.eia_pk

        # HALP: I couldn't figure out how to use the "columns" in the
        # TableTreatment to set... anything. If I set it as an empty dict
        # all of these extracted columns/dictionaries would come up blank but
        # if I didn't let it = {}, columns wouldn't be set and thus I could
        # do nothing with columns

        # So here are a much of little functions to extract info from
        # TableTreatment.columns
        self.sum_cols: List = self._extract_sum_cols()
        self.str_cols: List = self._extract_str_cols()
        self.wtavg_dict: Dict[str, str] = self._extract_wtavg_dict()
        self.data_set_idx_cols: List = self._extract_data_set_idx_cols()
        self.scale_cols: Dict[str, List[str]] = self._extract_scale_cols()

    def _extract_sum_cols(self):
        return [
            col for (col, treatments) in self.metadata.columns.items()
            if treatments.sum_col
        ]

    def _extract_str_cols(self):
        return [
            col for (col, treatments) in self.metadata.columns.items()
            if treatments.str_col
        ]

    def _extract_wtavg_dict(self):
        return {
            col: treatments.wtavg_col for (col, treatments)
            in self.metadata.columns.items() if treatments.wtavg_col
        }

    def _extract_data_set_idx_cols(self):
        return [
            col for (col, treatments) in self.metadata.columns.items()
            if treatments.data_set_idx_col
        ]

    def _extract_scale_cols(self):
        return {
            col: treatments.scale_col for (col, treatments)
            in self.metadata.columns.items() if treatments.scale_col
        }

    def execute(self, df_to_scale: pd.DataFrame, ppl: pd.DataFrame):
        """Do it."""
        # extract the records that are NOT connected to the EIA plant-part list
        # Note: Right now we are just dropping the non-connected
        # not_connected = df_to_scale[df_to_scale.record_id_eia.isnull()]
        connected_to_scale = df_to_scale[~df_to_scale.record_id_eia.isnull()]
        # Deduplicate when there is more than one source record associated with
        # the same EIA plant-part.
        to_scale = self.aggregate_duplicate_eia(connected_to_scale, ppl)

        merged_df = self.many_merge_on_scale_part(
            to_scale=to_scale,
            ppl=ppl.reset_index(),
            cols_to_keep=list(self.metadata.columns.keys())
        )

        # grab all of the ppl columns, plus data set's id column(s)
        # this enables us to have a unique index
        idx_cols = (
            pudl.analysis.plant_parts_eia.PLANT_PARTS
            [self.plant_part]['id_cols']
            + pudl.analysis.plant_parts_eia.IDX_TO_ADD
            + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
            + self.data_set_idx_cols
        )
        scaled_df = merged_df.set_index(idx_cols)
        for scale_col, split_cols in self.scale_cols.items():
            scaled_df.loc[:, f"{scale_col}_scaled"] = split_data_on_split_cols(
                df_to_scale=scaled_df,
                merge_cols=self.data_set_idx_cols,
                data_col=scale_col,
                split_cols=split_cols
            )
        # HALP: I want to just assign the output of split_data_on_split_cols
        # to the frickin scale_col, but it keeps returning a column of nulls
        # So i'm doing this janky drop and rename
        scaled_df = (
            scaled_df.drop(columns=self.scale_cols.keys())
            .rename(columns={
                c: c.replace('_scaled', '')
                for c in [c for c in scaled_df.columns if "_scaled" in c]}
            )
        )
        # second aggregation of the duplicate EIA records.
        scaled_df_post_agg = self.aggregate_duplicate_eia(
            connected_to_scale=scaled_df.reset_index(),
            ppl=ppl
        )
        # set the index to be the main EIA plant-part index columns
        scaled_df_post_agg = (
            scaled_df_post_agg.set_index(idx_cols + ['record_id_eia'])
            .reset_index(self.data_set_idx_cols)
        )

        return scaled_df_post_agg

    def aggregate_duplicate_eia(self, connected_to_scale, ppl):
        """Aggregate duplicate EIA plant-part records."""
        dupe_mask = connected_to_scale.duplicated(
            subset=self.eia_pk, keep=False
        )
        # two dfs
        dupes = connected_to_scale[dupe_mask]
        non_dupes = connected_to_scale[~dupe_mask]
        # If there are no duplicate records, then the following aggs will fail
        # bc there is nothing to merge. If there is a cleaner way to skip this
        # without a dangly mid-function retrun LMK!
        if dupes.empty:
            return non_dupes
        logger.info(
            f"Aggergating {len(dupes)} duplicate records "
            f"({len(dupes)/len(connected_to_scale):.1%})")

        # sum and weighted average!
        de_duped = pudl.helpers.sum_and_weighted_average_agg(
            df_in=dupes,
            by=self.eia_pk,
            sum_cols=self.sum_cols,
            wtavg_dict=self.wtavg_dict
        )
        # add in the string columns
        de_duped = de_duped.merge(
            (
                dupes.groupby(self.eia_pk, as_index=False)
                .agg({k: str_squish for k in self.str_cols})
            ),
            on=self.eia_pk,
            validate='1:1',
            how='left'
        ).pipe(pudl.helpers.convert_cols_dtypes, 'eia')

        # merge back in the ppl idx columns
        de_duped_w_ppl = (
            de_duped.set_index('record_id_eia')
            .merge(
                ppl,  # [[c for c in PPL_COLS if c != 'record_id_eia']],
                left_index=True,
                right_index=True,
                how='left',
                validate='m:1',
            )
            .reset_index()
        )
        # merge the non-dupes and de-duplicated records
        # we're doing an inner merge here bc we don't want columns with
        # partially null values
        return pd.concat([non_dupes, de_duped_w_ppl], join='inner')

    def many_merge_on_scale_part(
            self,
            to_scale: pd.DataFrame,
            cols_to_keep: list,
            ppl: pd.DataFrame) -> pd.DataFrame:
        """
        Merge a particular EIA plant-part list plant-part onto a dataframe.

        Returns:
            a table.
        """
        ppl_part_df = ppl[ppl.plant_part == self.plant_part]
        # convert the date to year start
        to_scale.loc[:, 'report_date'] = (
            pd.to_datetime(to_scale.report_date.dt.year, format='%Y')
        )
        scale_parts = []
        for merge_part in pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED:
            idx_part = (
                pudl.analysis.plant_parts_eia.PLANT_PARTS
                [merge_part]['id_cols']
                + pudl.analysis.plant_parts_eia.IDX_TO_ADD
                + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
            )
            # grab just the part of the df that cooresponds to the plant_part
            part_df = pd.merge(
                (
                    to_scale[to_scale.plant_part == merge_part]
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


def str_squish(x):
    """Squish strings from a groupby into a list."""
    return '; '.join(list(map(str, [x for x in x.unique() if x is not pd.NA])))


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
