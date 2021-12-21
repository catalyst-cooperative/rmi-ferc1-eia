"""
Connect the depreciation data with FERC1 steam plant records.

This module attempts to connect the depreciation data with FERC1 steam records
through the EIA plant-part list. Both the depreciation records and FERC1 steam
has been connected to the EIA plant-part list, which is a compilation of
various possible combinations of generator records.

Currently Implemented:
* A scale-to-generator-er.
    Inputs:
    * Any dataset that has been connected to the EIA plant-part list. This
     dataset can have heterogeneous plant-parts (i.e. one record can be
     associated with a full plant while the next can be associated with a ).
    * Metadata regarding the input dataset and how to operate on each of the
     columns.
    Outputs:
    * The initial dataset scaled to the generator level.

Future Needs:
* A merger of generator-based records. This is currently implemented in the
``connect_deprish_to_ferc1`` notebook, but it needs to be buttoned up and
integrated here.
* (Possible) Enable the scaler to scale to any plant-part. Right now only
splitting is integrated and thus we can only scale to the smallest plant-part
(the generator). Enabling scaling to any plant-part would require both
splitting and aggregating, as well as labeling which method to apply to each
record. This labeling is required becuase
"""

import logging


from typing import List, Optional, Dict
import pydantic
from pydantic import BaseModel

import pandas as pd

import pudl

logger = logging.getLogger(__name__)

META_DEPRISH_EIA: Dict[str, "FieldTreatment"] = {
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


def execute(plant_parts_eia, deprish_eia, ferc1_to_eia):
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
    """
    logger.info("Scaling FERC1-EIA to the generator level.")
    scaled_fe = (
        ScaleToPlantParter(**{
            'columns': META_FERC1_EIA,
            'eia_pk': ['record_id_eia'],
            'plant_part': 'plant_gen'
        })
        .execute(
            df_to_scale=ferc1_to_eia,
            ppl=plant_parts_eia)
    )

    logger.info("Scaling Depreciation-EIA to the generator level.")
    scaled_de = (
        ScaleToPlantParter(**{
            'columns': META_DEPRISH_EIA,
            'eia_pk': ['record_id_eia', 'data_source'],
            'plant_part': 'plant_gen'
        })
        .execute(
            df_to_scale=deprish_eia,
            ppl=plant_parts_eia)
    )

    ferc_deprish_eia = (
        pd.merge(
            scaled_de,
            scaled_fe,
            right_index=True,
            left_index=True,
            how='outer',
            suffixes=('_deprish', '_ferc1'),
        )
    )
    return ferc_deprish_eia


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
    str_col: Optional[pydantic.StrictBool] = False

    class Config:
        """
        An attempt to stop getting an error on reload.

        The error:
            ``ValueError: "FieldTreatment" object has no field "__class__"``

        Ref: https://github.com/samuelcolvin/pydantic/issues/288
        """

        allow_population_by_field_name = True


class ScaleToPlantParter(BaseModel):
    """Scale a table process a table."""

    columns: Dict[str, FieldTreatment]
    eia_pk: List[str] = ['record_id_eia']
    plant_part: str

    class Config:
        """
        An attempt to stop getting an error on reload.

        The error:
            ``ValueError: "FieldTreatment" object has no field "__class__"``

        Ref: https://github.com/samuelcolvin/pydantic/issues/288
        """

        allow_population_by_field_name = True

    # def extract_list_of_cols(self, treatment_type):
    #     """
    #     Grab the columns which need to be summed from the column meta.
    #
    #     HALP: This doesn't work bc "treatments" below is a
    #     `FieldTreatment` and `'FieldTreatment' object is not subscriptable`.
    #     Are there ways to extract info from a `FieldTreatment` without the
    #     name of the element like I'm doing below??
    #     """
    #     return [
    #         col for (col, treatments) in self.columns.items()
    #         if treatments[treatment_type]
    #     ]
    #
    # def extract_dict_of_col_treatments(self, treatment_type):
    #     """
    #     Grab the columns which need to be  from the column meta.
    #
    #     Same HALP as above.
    #     """
    #     return {
    #         col: treatments[treatment_type] for (col, treatments)
    #         in self.columns.items() if treatments[treatment_type]
    #     }

    def extract_sum_cols(self):
        """Grab the columns which need a string treatment from the metadata."""
        return [
            col for (col, treatments) in self.columns.items()
            if treatments.sum_col
        ]

    def extract_str_cols(self):
        """Grab the columns which need a string treatment from the metadata."""
        return [
            col for (col, treatments) in self.columns.items()
            if treatments.str_col
        ]

    def extract_wtavg_dict(self):
        """Grab the dict of columns that get a weighted average treatment."""
        return {
            col: treatments.wtavg_col for (col, treatments)
            in self.columns.items() if treatments.wtavg_col
        }

    def extract_data_set_idx_cols(self):
        """Grab the data set index/primary_key columns from the metadata."""
        return [
            col for (col, treatments) in self.columns.items()
            if treatments.data_set_idx_col
        ]

    def extract_scale_cols(self):
        """Grab the columns from the metadata which need to be scaled."""
        return {
            col: treatments.scale_col for (col, treatments)
            in self.columns.items() if treatments.scale_col
        }

    def execute(self, df_to_scale: pd.DataFrame, ppl: pd.DataFrame):
        """
        Scale a dataframe to a generator-level.

        There are four main steps here:
        * STEP 1: Aggregate the dataset records that are associated with the
          same EIA plant-part records. Sometime the original dataset include
          mulitple records that are associated with the same plant-part record.
          We are assuming that the duplicate records are not reporting
          overlapping data (i.e. one depreciation record for a unit and another
          for the emissions control costs associated with the same unit). We
          know this isn't always a perfect bet.
        * STEP 2: Merge in the EIA plant-part list generators. This is
          generally a one-to-many merge where we cast many generators across
          each data-set record. Now we have the dataset records associated with
          generators but duplicated.
        * STEP 3: This is the splitting/scaling step. Here we take the
          dataset records that have been duplicated across their mulitple
          generator components and distribute portions of the data columns
          based on another weighting column (ex: if there are 2 generators
          associated with a dataset record and one has a 100 MW capacity while
          the second has a 200 MW capacity, 1/3 of the data column would be
          allocated to the first generator while the remaining 2/3 would be
          allocated to the second generator). At the end of this step, we have
          generator records with data columns scaled.
        * STEP 4: Aggregate the generator based records. At this step we
          sometimes have multiple records representing the same generator. This
          happens when we have two seperate records reporting overlapping
          peices of infrastructure (ex: a plant's coal ash pound in one
          depreciation record and a unit in another). We are assuming here that
          the records do not contain duplicate data - which we know isn't
          always a perfect bet.
        """
        # extract the records that are NOT connected to the EIA plant-part list
        # Note: Right now we are just dropping the non-connected
        # not_connected = df_to_scale[df_to_scale.record_id_eia.isnull()]
        connected_to_scale = df_to_scale[~df_to_scale.record_id_eia.isnull()]
        # STEP 1
        # Deduplicate when there is more than one source record associated with
        # the same EIA plant-part.
        to_scale = self.aggregate_duplicate_eia(connected_to_scale, ppl)
        # STEP 2
        merged_df = self.many_merge_on_scale_part(
            to_scale=to_scale,
            ppl=ppl.reset_index(),
            cols_to_keep=list(self.columns.keys())
        )
        # STEP 3
        # grab all of the ppl columns, plus data set's id column(s)
        # this enables us to have a unique index
        idx_cols = (
            pudl.analysis.plant_parts_eia.PLANT_PARTS
            [self.plant_part]['id_cols']
            + pudl.analysis.plant_parts_eia.IDX_TO_ADD
            + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
            + self.extract_data_set_idx_cols()
        )
        scaled_df = merged_df.set_index(idx_cols)
        scale_col_dict = self.extract_scale_cols()
        for scale_col, split_cols in scale_col_dict.items():
            scaled_df.loc[:, f"{scale_col}_scaled"] = split_data_on_split_cols(
                df_to_scale=scaled_df,
                merge_cols=self.extract_data_set_idx_cols(),
                data_col=scale_col,
                split_cols=split_cols
            )
        # HALP: I want to just assign the output of split_data_on_split_cols
        # to the frickin scale_col, but it keeps returning a column of nulls
        # So i'm doing this janky drop and rename
        scaled_df = (
            scaled_df.drop(columns=scale_col_dict.keys())
            .rename(columns={
                c: c.replace('_scaled', '')
                for c in [c for c in scaled_df.columns if "_scaled" in c]}
            )
        )
        # STEP 4
        # second aggregation of the duplicate EIA records.
        scaled_df_post_agg = self.aggregate_duplicate_eia(
            connected_to_scale=scaled_df.reset_index(),
            ppl=ppl
        )
        # set the index to be the main EIA plant-part index columns
        scaled_df_post_agg = (
            scaled_df_post_agg.set_index(idx_cols + ['record_id_eia'])
            .reset_index(self.extract_data_set_idx_cols())
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
            sum_cols=self.extract_sum_cols(),
            wtavg_dict=self.extract_wtavg_dict()
        )
        # add in the string columns
        de_duped = de_duped.merge(
            (
                dupes.groupby(self.eia_pk, as_index=False)
                .agg({k: str_squish for k in self.extract_str_cols()})
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
