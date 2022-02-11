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
* PK: This module uses PK to denote primary keys of data tables. They are not
  true primary keys in the database sense, but rather the set of columns that
  would constitue a composite primary key. You can think of them as index
  columns, but they are not always found as the index of the DataFrame and thus
  PK seems like a more apt description.

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
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import pudl
from pydantic import BaseModel, validator

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
    'capex_annual_addition':
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
            that correspond to portions of plants from generators to fuel
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
            ppl_pk=['record_id_eia'],
            data_set_pk_cols=['record_id_ferc1'],
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
            ppl_pk=['record_id_eia', 'data_source'],
            data_set_pk_cols=['line_id'],
            plant_part='plant_gen'
        )
        .execute(
            df_to_scale=deprish_eia,
            ppl=plant_parts_eia)
    )

    # both of these scaled dfs have ppl columns. we are going to drop all of
    # the ppl columns before merging and then merge the ppl back in as oppose
    # to try to reconcile the ppl columns from the scaled dfs
    ferc_deprish_eia = (
        pd.merge(
            scaled_de.drop(
                columns=[c for c in scaled_de if c in plant_parts_eia]),
            scaled_fe.drop(
                columns=[c for c in scaled_fe if c in plant_parts_eia]),
            right_index=True,
            left_index=True,
            how='outer',
        )
        .merge(
            plant_parts_eia,
            right_index=True, left_index=True,
            how='left'
        )
    )
    test_consistency_of_data_stages(df1=deprish_eia, df2=ferc_deprish_eia)
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
        ppl_pk: Identifing columns for the EIA plant-part list.
        data_set_pk_cols: Identifing columns for dataset to scale.
        plant_part: name of EIA plant-part to scale to. Current implementation
            only allows for ``plant_gen``.

    """

    treatments: Dict[str, FieldTreatment]
    ppl_pk: List[str] = ['record_id_eia']
    data_set_pk_cols: List[str]
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
    def plant_part_pk_cols(self):
        """Get the primary keys for a plant-part."""
        return (pudl.analysis.plant_parts_eia.PLANT_PARTS
                [self.plant_part]['id_cols']
                + pudl.analysis.plant_parts_eia.IDX_TO_ADD
                + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD)

    def execute(
        self,
        df_to_scale: pd.DataFrame,
        ppl: pd.DataFrame
    ) -> pd.DataFrame:
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
            ppl: the EIA plant-part list.

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
        scaled_df_post_agg = scaled_df_post_agg.set_index(['record_id_eia'])
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
        merged_df = self.broadcast_merge_to_plant_part(
            data_to_scale=to_allocate,
            ppl=ppl.reset_index(),
            cols_to_keep=list(self.treatments)
        )
        # at this stage we have plant-part records with the data columns from
        # the original dataset broadcast across multiple records: read the data
        # is duplicated!

        # STEP 3
        # grab all of the ppl columns, plus data set's id column(s)
        # this enables us to have a unique index
        pk_cols = (
            self.plant_part_pk_cols
            + self.data_set_pk_cols
        )
        allocated = merged_df.set_index(pk_cols)
        for allocate_col, allocator_cols in self.allocator_cols_dict.items():
            allocated.loc[:, allocate_col] = _allocate_col(
                to_allocate=allocated,
                by=self.data_set_pk_cols,
                allocate_col=allocate_col,
                allocator_cols=allocator_cols
            )
        return allocated

    def aggregate_duplicate_eia(self, connected_to_scale, ppl):
        """Aggregate duplicate EIA plant-part records."""
        dupe_mask = connected_to_scale.duplicated(
            subset=self.ppl_pk, keep=False
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
                by=self.ppl_pk,
                sum_cols=self._get_treatment_cols('scale'),
                wtavg_dict=self.wtavg_dict
            ).pipe(pudl.metadata.fields.apply_pudl_dtypes)
            # add in the string columns
            # TODO: add a test to ensure that the str-squish character doesn't
            # show up in the original data columns
            de_duped = de_duped.merge(
                (
                    dupes.groupby(self.ppl_pk, as_index=False)
                    .agg({k: str_concat for k
                          in self._get_treatment_cols('str_concat')})
                ),
                on=self.ppl_pk,
                validate='1:1',
                how='left'
            ).pipe(pudl.metadata.fields.apply_pudl_dtypes)

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

    def broadcast_merge_to_plant_part(
            self,
            data_to_scale: pd.DataFrame,
            cols_to_keep: List[str],
            ppl: pd.DataFrame) -> pd.DataFrame:
        """
        Broadcast data with a variety of granularities to a single plant-part.

        This method merges an input dataframe (``data_to_scale``) containing
        data that has a heterogeneous set of plant-part granularities with a
        subset of the EIA plant-part list that has a single granularity.
        (Currently this single granularity must be generators). In general this
        will be a one-to-many merge in which values from single records in the
        input data end up associated with several records from the plant part
        list.

        First, we select a subset of the full EIA plant-part list corresponding
        to the plant-part of the :class:`PlantPartScaler` instance.  (specified
        by its :attr:`plant_part`). In theory this could be the plant,
        generator, fuel type, etc. Currently only generators are supported.

        Then, we iterate over all the possible plant parts, selecting the
        subset of records in ``data_to_scale`` that have that granularity, and
        merge the homogeneous subset of the plant part list that we selected
        above onto that subset of the input data. Each iteration uses a
        different set of columns to merge on -- the columns which define the
        primary key for the plant part being merged. Each iteration creates a
        separate dataframe, corresponding to a particular plant part, and at
        the end they are all concatenated together and returned.

        This method is implementing Step 2 enumerated in :meth:`execute`.

        Note: :user:`cmgosnell` thinks this method might apply to both
        aggretation and allocation, so it is using the more generic "scale"
        verb.

        Args:
            data_to_scale: a data table where all records have been linked to
                EIA plant-part list but they may be heterogeneous in its
                plant-part granularities (i.e. some records could be of 'plant'
                plant-part type while others are 'plant_gen' or
                'plant_prime_mover').  All of the plant-part list columns need
                to be present in this table.
            cols_to_keep: columns from the original data ``data_to_scale`` that
                you want to show up in the output. These should not be columns
                that show up in the ``ppl``.
            ppl: the EIA plant-part list.

        Returns:
            A dataframe in which records correspond to :attr:`plant_part` (in
            the current implementation: the records all correspond to EIA
            generators!). This is an intermediate table that cannot be used
            directly for analysis because the data columns from the original
            dataset are duplicated and still need to be scaled up/down.

        """
        # select only the plant-part records that we are trying to scale to
        ppl_part_df = ppl[ppl.plant_part == self.plant_part]
        # convert the date to year start - this is necessary because the
        # depreciation data is often reported as EOY and the ppl is always SOY
        data_to_scale.loc[:, 'report_date'] = (
            pd.to_datetime(data_to_scale.report_date.dt.year, format='%Y')
        )
        out_dfs = []
        for merge_part in pudl.analysis.plant_parts_eia.PLANT_PARTS_ORDERED:
            pk_cols = (
                pudl.analysis.plant_parts_eia.PLANT_PARTS
                [merge_part]['id_cols']
                + pudl.analysis.plant_parts_eia.IDX_TO_ADD
                + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
            )
            part_df = pd.merge(
                (
                    # select just the records that correspond to merge_part
                    data_to_scale[data_to_scale.plant_part == merge_part]
                    [pk_cols + ['record_id_eia'] + cols_to_keep]
                ),
                ppl_part_df,
                on=pk_cols,
                how='left',
                # this unfortunately needs to be a m:m bc sometimes the df
                # data_to_scale has multiple record associated with the same
                # record_id_eia but are unique records and are not aggregated
                # in aggregate_duplicate_eia. For instance, the depreciation
                # data has both PUC and FERC studies.
                validate='m:m',
                suffixes=('_og', '')
            )
            out_dfs.append(part_df)
        out_df = pd.concat(out_dfs)
        return out_df


def str_concat(x):
    """Concatenate list of strings with a semicolon-space delimiter."""
    return '; '.join(list(map(str, [x for x in x.unique() if x is not pd.NA])))


def _allocate_col(
        to_allocate: pd.DataFrame,
        by: list,
        allocate_col: str,
        allocator_cols: List[str]) -> pd.Series:
    """
    Allocate larger dataset records porportionally by EIA plant-part columns.

    Args:
        to_allocate: table of data that has been merged with the EIA plant-part
            list records of the scale that you want the output to be in.
        allocate_col: name of the data column to scale. The data in this column
            has been broadcast across multiple records in
            :meth:`broadcast_merge_to_plant_part`.
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
    output_col = f"{allocate_col}_allocated"
    to_allocate[output_col] = pd.NA
    for allocator_col in allocator_cols:
        to_allocate[f"{allocator_col}_proportion"] = (
            to_allocate[allocator_col] / to_allocate[f"{allocator_col}_total"])
        # choose the first non-null option. The order of the allocate_cols will
        # determine which allocate_col will be used
        to_allocate[output_col] = (
            to_allocate[output_col].fillna(
                to_allocate[allocate_col]
                * to_allocate[f"{allocator_col}_proportion"])
        )
    to_allocate = (
        to_allocate.drop(columns=allocate_col)
        .rename(columns={output_col: allocate_col})
    )
    return to_allocate.loc[:, [allocate_col]]


##################
# Validation Tests
##################

# This whole section should pprrrrooobably be moved into a validation layer of
# CI tests for this repo.

def test_consistency_of_data_stages(df1, df2):
    """
    Test the consistency of two stages of the depreciation data processing.

    The data that is processed in this repo goes along multiple stages of its
    journey. This function right now is hard coded to test two the depreication
    data's main data columns. Right now, this is hard coded to fail when there
    are more inconsitent plants and utilities than are currently known.

    TODO: I think :func:``data_col_test`` is a perfect candidate for the
    decorators that let you test multiple inputs (different stages of the data
    and different data columns) as well as output values (in this case the
    number of known bad utilities and plants).

    Args:
        df1: One dataframe to sum and check consistency with ``df2``. Both
            dfs should have columns ``data_col`` as well as identifying
            columns: ``['report_date', 'data_source', 'utility_id_pudl',
            'plant_id_eia']``
        df2: Other dataframe to sum and check consistency against ``df1``.

    """
    util_bad, plant_bad = data_col_test(
        df1=df1, df2=df2, data_col='plant_balance_w_common')

    # known baddies. We need to track these down
    assert(len(plant_bad) <= 53)
    assert(len(util_bad) <= 88)

    util_bad, plant_bad = data_col_test(
        df1=df1, df2=df2, data_col='unaccrued_balance_w_common')

    # known baddies. We need to track these down
    assert(len(plant_bad) <= 35)
    assert(len(util_bad) <= 70)


def data_col_test(df1, df2, data_col: str) -> pd.DataFrame:
    """
    Check consistency of column at the plant and utility level in two inputs.

    Args:
        df1: One dataframe to sum and check consistency with ``df2``. Both
            dfs should have columns ``data_col`` as well as identifying
            columns: ``['report_date', 'data_source', 'utility_id_pudl',
            'plant_id_eia']``
        df2: Other dataframe to sum and check consistency against ``df1``.
        data_col: data column to check. Column must be in both ``df1`` and
            ``df2``.

    """
    util_test = gb_test(
        df1,
        df2,
        data_col=data_col,
        by=['report_year', 'data_source', 'utility_id_pudl']
    )

    logger.info(
        f"Duke utility level data is consistent for {data_col}!! Go forth and "
        "close them coal plants lil tables")

    plant_test = gb_test(
        df1,
        df2,
        data_col=data_col,
        by=['report_year', 'data_source', 'utility_id_pudl', 'plant_id_eia']
    )

    util_bad = util_test[~util_test.match & util_test.match.notnull()]
    plant_bad = plant_test[~plant_test.match & plant_test.match.notnull()]
    logger.warning(
        f"We have {len(util_bad)} utilities and {len(plant_bad)} "
        f"plants who's {data_col} doesn't match the input."
    )

    # we know the duke utilities should be correct!
    assert(util_test.loc[2018, 'FERC', 90].match)
    assert(util_test.loc[2018, 'FERC', 97].match)
    return util_bad, plant_bad


def gb_test(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    data_col: str,
    by: List[str]
) -> pd.DataFrame:
    """
    Merge two grouped input tables to determine if summed data column matches.

    Args:
        df1: One dataframe to sum and check consistency with ``df2``.
        df2: Other dataframe to sum and check consistency against ``df1``.
        data_col: data column to check. Column must be in both ``df1`` and
            ``df2``.
    """
    return (
        pd.merge(
            _group_sum_col(
                df1, data_col=data_col, by=by),
            _group_sum_col(
                df2, data_col=data_col, by=by),
            right_index=True, left_index=True,
            suffixes=('_1', '_2'),
            how='outer'
        )
        .assign(
            match=lambda x: np.where(
                x[f"{data_col}_1"].notnull() & x[f"{data_col}_2"].notnull(),
                np.isclose(x[f"{data_col}_1"], x[f"{data_col}_2"]),
                pd.NA
            ),
            diff=lambda x: x[f"{data_col}_1"] / x[f"{data_col}_2"],
        )
        .sort_index()
    )


def _group_sum_col(df, data_col: str, by: List[str]) -> pd.DataFrame:
    """Groupby sum a specific table's data col."""
    return (
        df  # convert date to year bc many of the og depish studies are EOY
        .assign(report_year=lambda x: x.report_date.dt.year)
        [df.plant_id_eia.notnull()]  # only plant associated reocrds
        .groupby(by=by, dropna=True)
        [[data_col]]
        .sum(min_count=1)
    )
