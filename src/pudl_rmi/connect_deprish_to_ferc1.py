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
     associated with a full plant while the next can be associated with a
     generator or a unit).
    * Information regarding how to transform each of the columns in the input
     dataset.
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


from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, validator

import pandas as pd

import pudl

logger = logging.getLogger(__name__)

META_DEPRISH_EIA: Dict[str, "FieldTreatment"] = {
    'line_id':
        {
            'treat_type': 'str_concat'
        },
    'plant_balance_w_common':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'book_reserve_w_common':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'unaccrued_balance_w_common':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_salvage_w_common':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'depreciation_annual_epxns_w_common':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_removal_rate':
        {
            'treat_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'depreciation_annual_rate':
        {
            'treat_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'remaining_life_avg':
        {
            'treat_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'utility_name_ferc1':
        {
            'treat_type': 'str_concat'
        },
    'data_source':
        {
            'treat_type': 'str_concat'
        }
}


META_FERC1_EIA: Dict[str, "FieldTreatment"] = {
    'record_id_ferc1':
        {
            'treat_type': 'str_concat'
        },
    'capex_total':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capex_annual_addt':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'opex_nonfuel':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capacity_mw_ferc1':
        {
            'treat_type': 'scale',
            'scale_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'avg_num_employees':
        {
            'treat_type': 'wtavg',
            'wtavg_col': 'capacity_mw_ferc1'
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
        PlantPartScaler(
            treatments=META_FERC1_EIA,
            eia_pk=['record_id_eia'],
            data_set_idx_cols=['record_id_ferc1'],
            plant_part='plant_gen'
        )
        .execute(
            df_to_scale=ferc1_to_eia,
            ppl=plant_parts_eia)
    )

    logger.info("Scaling Depreciation-EIA to the generator level.")
    scaled_de = (
        PlantPartScaler(
            treatments=META_DEPRISH_EIA,
            eia_pk=['record_id_eia', 'data_source'],
            data_set_idx_cols=['line_id'],
            plant_part='plant_gen'
        )
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

    scale_cols: Optional[List[str]] = []
    wtavg_col: Optional[str] = []

    treat_type: Literal['scale', 'str_concat', 'wtavg']

    @validator('treat_type')
    def check_treatments(cls, value, values):   # noqa: N805
        """Check treatments that need additional info."""
        if value == 'scale' and not values.get('scale_cols'):
            raise AssertionError("Scale column treatment needs scale_cols")
        if value == 'wtavg' and not values.get('wtavg_col'):
            raise AssertionError(
                "Weighted Average column treatment needs wtavg_col")
        return value

    class Config:
        """
        An attempt to stop getting an error on reload.

        The error:
            ``ValueError: "FieldTreatment" object has no field "__class__"``

        Ref: https://github.com/samuelcolvin/pydantic/issues/288
        """

        allow_population_by_field_name = True


class PlantPartScaler(BaseModel):
    """
    Scale a table process a table.

    Args:
        columns:
        eia_pk:
        data_set_idx_cols:
        plant_part:

    """

    treatments: Dict[str, FieldTreatment]
    eia_pk: List[str] = ['record_id_eia']
    data_set_idx_cols: List[str]
    plant_part: Literal['plant_gen']

    class Config:
        """
        An attempt to stop getting an error on reload.

        The error:
            ``ValueError: "FieldTreatment" object has no field "__class__"``

        Ref: https://github.com/samuelcolvin/pydantic/issues/288
        """

        allow_population_by_field_name = True

    def get_cols_by_treatment(self, treat_type: str):
        """Grab the columns which need a specific treatment type."""
        return [
            col for (col, treat) in self.treatments.items()
            if treat.treat_type == treat_type]

    @property
    def wtavg_dict(self):
        """Grab the dict of columns that get a weighted average treatment."""
        return {
            wtavg_col: self.treatments[wtavg_col].wtavg_col
            for wtavg_col in self.get_cols_by_treatment('wtavg')
        }

    @property
    def scale_cols_dict(self):
        """Grab the columns from the metadata which need to be scaled."""
        return {
            scale_col: self.treatments[scale_col].scale_cols
            for scale_col in self.get_cols_by_treatment('scale')
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
        # Aggregate when there is more than one source record associated with
        # the same EIA plant-part.
        to_scale = self.aggregate_duplicate_eia(connected_to_scale, ppl)
        # STEP 2
        merged_df = self.many_merge_on_scale_part(
            to_scale=to_scale,
            ppl=ppl.reset_index(),
            cols_to_keep=list(self.treatments)
        )
        # STEP 3
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
        for scale_col, split_cols in self.scale_cols_dict.items():
            scaled_df.loc[:, scale_col] = split_data_on_split_cols(
                df_to_scale=scaled_df,
                by=self.data_set_idx_cols,
                data_col=scale_col,
                split_cols=split_cols
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
            sum_cols=self.get_cols_by_treatment('scale'),
            wtavg_dict=self.wtavg_dict
        )
        # add in the string columns
        # TODO: add a test to ensure that the str-squish character doesn't show
        # up in the original data columns
        de_duped = de_duped.merge(
            (
                dupes.groupby(self.eia_pk, as_index=False)
                .agg({k: str_concat for k in self.get_cols_by_treatment('str_concat')})
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


def str_concat(x):
    """Concatenate list of strings with a semicolon-space delimiter."""
    return '; '.join(list(map(str, [x for x in x.unique() if x is not pd.NA])))


def split_data_on_split_cols(
        df_to_scale: pd.DataFrame,
        by: list,
        data_col: str,
        split_cols: list) -> pd.Series:
    """
    Split larger dataset records porportionally by EIA plant-part list columns.

    This method associates slices of a dataset's records - which are larger
    than their EIA counter parts - via prioritized EIA plant-part list columns.

    Args:
        df_to_scale: table of data that has been merged with the EIA plant-part
            list records of the scale that you want the output to be in.
        data_col: name of the data column to scale. The data in this column has
            been broadcast across multiple records in ``df_to_scale``.
        by: columns to group by.
        split_cols: ordered list of columns to split porportionally
            based on. Ordered based on priority: if non-null result from
            frist column, result will include first column result, then
            second and so on.
    Returns:
        a series of the ``data_col`` scaled to the plant-part level.

    """
    # add a total column for all of the split cols. This will enable us to
    # determine each records' proportion of the
    df_to_scale.loc[:, [f"{c}_total" for c in split_cols]] = (
        df_to_scale.loc[:, split_cols]
        .groupby(by=by, dropna=False)
        .transform(sum, min_count=1)
        .add_suffix('_total')
    )
    # for each of the columns we want to split the frc data by
    # generate the % of the total group, so we can split the data_col
    new_data_col = f"{data_col}_scaled"
    df_to_scale[new_data_col] = pd.NA
    for split_col in split_cols:
        df_to_scale[f"{split_col}_proportion"] = (
            df_to_scale[split_col] / df_to_scale[f"{split_col}_total"])
        # choose the first non-null option. The order of the split_cols will
        # determine which split_col will be used
        df_to_scale[new_data_col] = (
            df_to_scale[new_data_col].fillna(
                df_to_scale[data_col] * df_to_scale[f"{split_col}_proportion"])
        )
    return df_to_scale[new_data_col]
