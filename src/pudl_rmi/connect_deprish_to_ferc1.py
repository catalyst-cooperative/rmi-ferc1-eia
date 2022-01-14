"""
Connect the depreciation data with FERC1 steam plant records.

This module attempts to connect the depreciation data with FERC1 steam records
through the EIA plant-part list. Both the depreciation records and FERC1 steam
has been connected to the EIA plant-part list, which is a compilation of
various possible combinations of generator records.

Some defintions:

* Scale: Converting record(s) from one set of plant-parts to another.
* Allocate: A specific implementation of scaling. This is down-scaling. When
  the original dataset has records that are of a larger granularity than your
  desired plant-part granularity, we need to allocate or distribute the data
  from original larger records across the smaller components.
* Aggregate: A specific implementation of scaling. This is up-scaling. When the
  original dataset has records that are of a smaller granularity than your
  desired plant-part granularity, we need to aggregate or sum up the data from
  the original smaller records to the larger granualry. (This is currently not
  implemented!)

Currently Implemented:

* An allocate-to-generator-er.

    * Inputs:

       * Any dataset that has been connected to the EIA plant-part list. This
         dataset can have heterogeneous plant-parts (i.e. one record can be
         associated with a full plant while the next can be associated with a
         generator or a unit).
       * Information regarding how to transform each of the columns in the
         input dataset.

    * Outputs:

       * The initial dataset scaled to the generator level.

Future Needs:

* A merger of generator-based records. This is currently implemented in the
  ``connect_deprish_to_ferc1`` notebook, but it needs to be buttoned up and
  integrated here.
* (Possible) Enable the scaler to scale to any plant-part. Right now only
  allocating is integrated and thus we can only scale to the smallest
  plant-part (the generator). Enabling scaling to any plant-part would require
  both allocating and aggregating, as well as labeling which method to apply to
  each record. This labeling is required becuase we'd need to know whether to
  allocate or aggregate an input record.

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
            'treatment_type': 'str_concat'
        },
    'plant_balance_w_common':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'book_reserve_w_common':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'unaccrued_balance_w_common':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_salvage_w_common':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'depreciation_annual_epxns_w_common':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'net_removal_rate':
        {
            'treatment_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'depreciation_annual_rate':
        {
            'treatment_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'remaining_life_avg':
        {
            'treatment_type': 'wtavg',
            'wtavg_col': 'unaccrued_balance_w_common'
        },
    'utility_name_ferc1':
        {
            'treatment_type': 'str_concat'
        },
    'data_source':
        {
            'treatment_type': 'str_concat'
        }
}


META_FERC1_EIA: Dict[str, "FieldTreatment"] = {
    'record_id_ferc1':
        {
            'treatment_type': 'str_concat'
        },
    'capex_total':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capex_annual_addt':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'opex_nonfuel':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'capacity_mw_ferc1':
        {
            'treatment_type': 'scale',
            'allocator_cols': [
                'capacity_mw',
                'net_generation_mwh',
                'total_fuel_cost'
            ],
        },
    'avg_num_employees':
        {
            'treatment_type': 'wtavg',
            'wtavg_col': 'capacity_mw_ferc1'
        },
}


def execute(plant_parts_eia, deprish_eia, ferc1_to_eia):
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
    """
    logger.info("Scaling FERC1-EIA to the generator level.")
    scaled_fe = (
        PlantPartScaler(
            treatments=META_FERC1_EIA,
            ppl_id=['record_id_eia'],
            data_set_id_cols=['record_id_ferc1'],
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
            ppl_id=['record_id_eia', 'data_source'],
            data_set_id_cols=['line_id'],
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

    The treatment type of scale is currently only being used to allocate
    because we have not implemented an aggregation scale process. Nonetheless,
    the columns we will want to aggregate are the same as those we would want
    to allocate, so we are keeping this more generic treatment name.

    Args:
        allocator_cols: an ordered list of columns that should be used when
            allocating the column. The list should be orderd based on priority
            of which column should attempted to be used as an allocator first.
            If the first column is null, it won't be able to be used as an
            allocator and the second column will be attempted to be used as an
            allocator and so on.
        wtavg_col: a column that will be used as a weighting column when
            applying a weighted average to the column.
        treatment_type: the name of a treatment type from the following types:
            ``scale``, ``str_concat``, ``wtavg``.

    """

    allocator_cols: Optional[List[str]]
    wtavg_col: Optional[str]

    treatment_type: Literal['scale', 'str_concat', 'wtavg']

    @validator('treatment_type')
    def check_treatments(cls, value, values):   # noqa: N805
        """Check treatments that need additional info."""
        if value == 'scale' and not values.get('allocator_cols'):
            raise AssertionError(
                "Scale column treatment needs allocator_cols")
        if value == 'wtavg' and not values.get('wtavg_col'):
            raise AssertionError(
                "Weighted Average column treatment needs wtavg_col")
        return value


class PlantPartScaler(BaseModel):
    """
    Scale a table to a plant-part.

    Args:
        treatments: a dictionary of column name (keys) with field treatments
            (values)
        ppl_id: Identifing columns for the EIA plant-part list.
        data_set_id_cols: Identifing columns for dataset to scale.
        plant_part: name of EIA plant-part to scale to. Current implementation
            only allows for ``plant_gen``.

    """

    treatments: Dict[str, FieldTreatment]
    ppl_id: List[str] = ['record_id_eia']
    data_set_id_cols: List[str]
    plant_part: Literal['plant_gen']

    def _get_treatment_cols(self, treatment_type: str) -> list[str]:
        """Grab the columns which need a specific treatment type."""
        return [
            col for (col, treat) in self.treatments.items()
            if treat.treatment_type == treatment_type]

    @property
    def wtavg_dict(self) -> Dict:
        """Grab the dict of columns that get a weighted average treatment."""
        return {
            wtavg_col: self.treatments[wtavg_col].wtavg_col
            for wtavg_col in self._get_treatment_cols('wtavg')
        }

    @property
    def allocator_cols_dict(self) -> Dict:
        """Grab the columns from the metadata which need to be allocated."""
        return {
            allocate_col: self.treatments[allocate_col].allocator_cols
            for allocate_col in self._get_treatment_cols('scale')
        }

    @property
    def plant_part_id_cols(self):
        """Get the primary keys for a plant-part."""
        return (pudl.analysis.plant_parts_eia.PLANT_PARTS
                [self.plant_part]['id_cols']
                + pudl.analysis.plant_parts_eia.IDX_TO_ADD
                + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD)

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
          generators but duplicated. (Step 2 and 3 are grouped into one method)
        * STEP 3: This is the scaling step (only allocaiton is currently
          implemented). Here we take the dataset records that have been
          duplicated across their mulitple generator components and distribute
          portions of the data columns based on another weighting column (ex:
          if there are 2 generators associated with a dataset record and one
          has a 100 MW capacity while the second has a 200 MW capacity, 1/3 of
          the data column would be allocated to the first generator while the
          remaining 2/3 would be allocated to the second generator). At the end
          of this step, we have generator records with data columns allocated.
        * STEP 4: Aggregate the generator based records. At this step we
          sometimes have multiple records representing the same generator. This
          happens when we have two seperate records reporting overlapping
          peices of infrastructure (ex: a plant's coal ash pound in one
          depreciation record and a unit in another). We are assuming here that
          the records do not contain duplicate data - which we know isn't
          always a perfect bet.

        Args:
            df_to_scale: the input data table that you want to scale.
            ppl:

        """
        # extract the records that are NOT connected to the EIA plant-part list
        # Note: Right now we are just dropping the non-connected
        # not_connected = df_to_scale[df_to_scale.record_id_eia.isnull()]
        connected_to_scale = df_to_scale[~df_to_scale.record_id_eia.isnull()]
        # STEP 1
        # Aggregate when there is more than one source record associated with
        # the same EIA plant-part.
        to_scale = self.aggregate_duplicate_eia(connected_to_scale, ppl)
        # STEP 2 & 3
        allocated = self.allocate(to_scale, ppl)
        # STEP 4
        # second aggregation of the duplicate EIA records.
        scaled_df_post_agg = self.aggregate_duplicate_eia(
            connected_to_scale=allocated.reset_index(),
            ppl=ppl
        )
        # set the index to be the main EIA plant-part index columns
        scaled_df_post_agg = scaled_df_post_agg.set_index(
            self.plant_part_id_cols + ['record_id_eia']
        )
        return scaled_df_post_agg

    def allocate(
            self,
            to_allocate: pd.DataFrame,
            ppl: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Allocate records w/ larger granularities to sub-component plant-parts.

        Implement both steps 2 and 3 as described in
        :meth:`PlantPartScaler.execute`.

        Args:
            to_allocate: the dataframe to be allocated.
            ppl: EIA plant-part list generated from
                ``pudl.analysis.plant_parts_eia``
        """
        # STEP 2
        merged_df = self.many_merge_on_scale_part(
            to_scale=to_allocate,
            ppl=ppl.reset_index(),
            cols_to_keep=list(self.treatments)
        )
        # at this stage we have plant-part records with the data columns from
        # the original dataset broadcast across multiple records: read the data
        # is duplicated!

        # STEP 3
        # grab all of the ppl columns, plus data set's id column(s)
        # this enables us to have a unique index
        idx_cols = (
            self.plant_part_id_cols
            + self.data_set_id_cols
        )
        allocated = merged_df.set_index(idx_cols)
        for allocate_col, allocator_cols in self.allocator_cols_dict.items():
            allocated.loc[:, allocate_col] = _allocate_col(
                to_allocate=allocated,
                by=self.data_set_id_cols,
                allocate_col=allocate_col,
                allocator_cols=allocator_cols
            )
        return allocated

    def aggregate_duplicate_eia(self, connected_to_scale, ppl):
        """Aggregate duplicate EIA plant-part records."""
        dupe_mask = connected_to_scale.duplicated(
            subset=self.ppl_id, keep=False
        )
        # two dfs
        dupes = connected_to_scale[dupe_mask]
        non_dupes = connected_to_scale[~dupe_mask]
        # If there are no duplicate records, then the following aggs will fail
        # bc there is nothing to merge. So we're making a new df to output that
        # is these non_dupes. If ther are dupes, we'll aggregate them!
        if dupes.empty:
            df_out = non_dupes
        else:
            logger.info(
                f"Aggergating {len(dupes)} duplicate records "
                f"({len(dupes)/len(connected_to_scale):.1%})")

            # sum and weighted average!
            de_duped = pudl.helpers.sum_and_weighted_average_agg(
                df_in=dupes,
                by=self.ppl_id,
                sum_cols=self._get_treatment_cols('scale'),
                wtavg_dict=self.wtavg_dict
            )
            # add in the string columns
            # TODO: add a test to ensure that the str-squish character doesn't
            # show up in the original data columns
            de_duped = de_duped.merge(
                (
                    dupes.groupby(self.ppl_id, as_index=False)
                    .agg({k: str_concat for k
                          in self._get_treatment_cols('str_concat')})
                ),
                on=self.ppl_id,
                validate='1:1',
                how='left'
            ).pipe(pudl.helpers.convert_cols_dtypes, 'eia')

            # merge back in the ppl idx columns
            de_duped = (
                de_duped.set_index('record_id_eia')
                .merge(
                    ppl,
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
            df_out = pd.concat([non_dupes, de_duped], join='inner')
        return df_out

    def many_merge_on_scale_part(
            self,
            to_scale: pd.DataFrame,
            cols_to_keep: list,
            ppl: pd.DataFrame) -> pd.DataFrame:
        """
        Merge a particular EIA plant-part list plant-part onto a dataframe.

        Note: CG is not sure if we will need this method to aggregate or not.
        So we are keeping this method with the more generic "scale" verbage.

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


def _allocate_col(
        to_allocate: pd.DataFrame,
        by: list,
        allocate_col: str,
        allocator_cols: list[str]) -> pd.Series:
    """
    Allocate larger dataset records porportionally by EIA plant-part columns.

    Args:
        to_allocate: table of data that has been merged with the EIA plant-part
            list records of the scale that you want the output to be in.
        allocate_col: name of the data column to scale. The data in this column
            has been broadcast across multiple records in ``df_to_scale``.
        by: columns to group by.
        allocator_cols: ordered list of columns to allocate porportionally
            based on. Ordered based on priority: if non-null result from
            frist column, result will include first column result, then
            second and so on.
    Returns:
        a series of the ``allocate_col`` scaled to the plant-part level.

    """
    # add a total column for all of the allocate cols. This will enable us to
    # determine each records' proportion of the
    to_allocate.loc[:, [f"{c}_total" for c in allocator_cols]] = (
        to_allocate.loc[:, allocator_cols]
        .groupby(by=by, dropna=False)
        .transform(sum, min_count=1)
        .add_suffix('_total')
    )
    # for each of the columns we want to allocate the frc data by
    # generate the % of the total group, so we can allocate the data_col
    allocated_col = f"{allocate_col}_allocated"
    to_allocate[allocated_col] = pd.NA
    for allocate_col in allocator_cols:
        to_allocate[f"{allocate_col}_proportion"] = (
            to_allocate[allocate_col] / to_allocate[f"{allocate_col}_total"])
        # choose the first non-null option. The order of the allocate_cols will
        # determine which allocate_col will be used
        to_allocate[allocated_col] = (
            to_allocate[allocated_col].fillna(
                to_allocate[allocate_col]
                * to_allocate[f"{allocate_col}_proportion"])
        )
    to_allocate = (
        to_allocate.drop(columns=allocate_col)
        .rename(columns={allocated_col: allocate_col})
    )
    return to_allocate.loc[:, [allocate_col]]
