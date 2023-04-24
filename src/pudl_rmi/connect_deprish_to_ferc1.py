"""
Connect the depreciation data with FERC1 plant records from the combined FERC table.

This module attempts to connect the depreciation data with FERC1 records
through the EIA plant-part list. Both the depreciation records and FERC1
all plants tables have been connected to the EIA plant-part list, which
is a compilation of various possible combinations of generator records.

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

import pandas as pd
import pudl
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

SCALE_CAP_GEN_COST: "FieldTreatment" = {
    "treatment_type": "scale",
    "allocator_cols": ["capacity_mw", "net_generation_mwh", "total_fuel_cost"],
}


META_DEPRISH_EIA: Dict[str, "FieldTreatment"] = {
    "line_id": {"treatment_type": "str_concat"},
    "plant_balance_w_common": SCALE_CAP_GEN_COST,
    "book_reserve_w_common": SCALE_CAP_GEN_COST,
    "unaccrued_balance_w_common": SCALE_CAP_GEN_COST,
    "net_plant_balance_w_common": SCALE_CAP_GEN_COST,
    "net_salvage_w_common": SCALE_CAP_GEN_COST,
    "depreciation_annual_epxns_w_common": SCALE_CAP_GEN_COST,
    "depreciation_annual_rate": {
        "treatment_type": "wtavg",
        "wtavg_col": "unaccrued_balance_w_common",
    },
    "remaining_life_avg": {
        "treatment_type": "wtavg",
        "wtavg_col": "unaccrued_balance_w_common",
    },
    "utility_name_ferc1": {"treatment_type": "str_concat"},
    "data_source": {"treatment_type": "str_concat"},
}


META_FERC1_EIA: Dict[str, "FieldTreatment"] = {
    "record_id_ferc1": {"treatment_type": "str_concat"},
    "capex_total": SCALE_CAP_GEN_COST,
    "capex_annual_addition": SCALE_CAP_GEN_COST,
    "opex_allowances": SCALE_CAP_GEN_COST,
    "opex_boiler": SCALE_CAP_GEN_COST,
    "opex_coolants": SCALE_CAP_GEN_COST,
    "opex_electric": SCALE_CAP_GEN_COST,
    "opex_engineering": SCALE_CAP_GEN_COST,
    "opex_fuel": SCALE_CAP_GEN_COST,
    "opex_misc_power": SCALE_CAP_GEN_COST,
    "opex_misc_steam": SCALE_CAP_GEN_COST,
    "opex_total_nonfuel": SCALE_CAP_GEN_COST,
    "opex_operations": SCALE_CAP_GEN_COST,
    "opex_plant": SCALE_CAP_GEN_COST,
    "opex_production_total": SCALE_CAP_GEN_COST,
    "opex_rents": SCALE_CAP_GEN_COST,
    "opex_steam": SCALE_CAP_GEN_COST,
    "opex_steam_other": SCALE_CAP_GEN_COST,
    "opex_structures": SCALE_CAP_GEN_COST,
    "opex_transfer": SCALE_CAP_GEN_COST,
    "opex_maintenance": SCALE_CAP_GEN_COST,
    "opex_total": SCALE_CAP_GEN_COST,
    "opex_dams": SCALE_CAP_GEN_COST,
    "opex_generation_misc": SCALE_CAP_GEN_COST,
    "opex_hydraulic": SCALE_CAP_GEN_COST,
    "opex_misc_plant": SCALE_CAP_GEN_COST,
    "opex_water_for_power": SCALE_CAP_GEN_COST,
    "opex_production_before_pumping": SCALE_CAP_GEN_COST,
    "opex_pumped_storage": SCALE_CAP_GEN_COST,
    "opex_pumping": SCALE_CAP_GEN_COST,
    "capacity_mw_ferc1": SCALE_CAP_GEN_COST,
    "avg_num_employees": {"treatment_type": "wtavg", "wtavg_col": "capacity_mw_ferc1"},
}


def execute(plant_parts_eia, deprish_eia, ferc1_eia):
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
    scaled_ferc1_eia = PlantPartScaler(
        treatments=META_FERC1_EIA,
        ppe_pk=["record_id_eia"],
        data_set_pk_cols=["record_id_ferc1"],
        plant_part="plant_gen",
    ).execute(df_to_scale=ferc1_eia, plant_parts_eia=plant_parts_eia)

    logger.info("Scaling Depreciation-EIA to the generator level.")
    scaled_deprish_eia = PlantPartScaler(
        treatments=META_DEPRISH_EIA,
        ppe_pk=["record_id_eia", "data_source"],
        data_set_pk_cols=["line_id"],
        plant_part="plant_gen",
    ).execute(df_to_scale=deprish_eia, plant_parts_eia=plant_parts_eia)

    # scale the FERC-EIA records that we can't match to Deprish-EIA records due
    # to ownership
    scaled_ferc1_eia = scale_to_fraction_owned(
        scaled_deprish_eia=scaled_deprish_eia,
        scaled_ferc1_eia=scaled_ferc1_eia,
        plant_parts_eia=plant_parts_eia,
    )
    # both of these scaled dfs have ppe columns. we are going to drop all of
    # the ppe columns before merging and then merge the ppe back in as oppose
    # to try to reconcile the ppe columns from the scaled dfs
    ferc_deprish_eia = pd.merge(
        scaled_deprish_eia.drop(
            columns=[c for c in scaled_deprish_eia if c in plant_parts_eia]
        ),
        scaled_ferc1_eia.drop(
            columns=[c for c in scaled_ferc1_eia if c in plant_parts_eia]
        ),
        right_index=True,
        left_index=True,
        how="outer",
    ).merge(plant_parts_eia, right_index=True, left_index=True, how="left")
    return ferc_deprish_eia


def scale_to_fraction_owned(
    scaled_deprish_eia: pd.DataFrame,
    scaled_ferc1_eia: pd.DataFrame,
    plant_parts_eia: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize by ownership.

    When we merge the FERC-EIA data (scaled to generators) with the Deprish-EIA
    data (scaled to generators), we are merging on the EIA plant-part list's
    ``record_id_eia``. A majority of the time is only one owner for each plant,
    but when plants do have multiple owners this complicates this merge. When
    mapping the depreciation records, we assume all of the records are "owned"
    portions of each plant - which are often synonymous with the "total"
    records.

    Find the FERC records which have a different ownership % than its deprish
    counterparts owership & the convert them.

    Args:
        scaled_deprish_eia
        scaled_ferc1_eia
        plant_parts_eia
    """
    # convert EIA ppe ownership dupes. Regardless of what a record was
    # matched with, default to the "total" ownership record if the "total"
    # and "owned" records are the same (i.e. there is only one owner)
    # this step could be done at basically any step before merging two
    # scaled_dfs together
    scaled_deprish_eia = pudl.analysis.plant_parts_eia.reassign_id_ownership_dupes(
        scaled_deprish_eia
    )
    scaled_ferc1_eia = pudl.analysis.plant_parts_eia.reassign_id_ownership_dupes(
        scaled_ferc1_eia
    )

    # first we must find the records that are connected
    # to the same EIA ppe record
    own_df = (
        pd.merge(
            _make_record_id_eia_wo_ownership(scaled_deprish_eia),
            _make_record_id_eia_wo_ownership(scaled_ferc1_eia),
            right_index=True,
            left_index=True,
            suffixes=("_de", "_fe"),
        ).assign(
            ownership_off=lambda x: x.record_id_eia_de != x.record_id_eia_fe
        )  # bc sometimes there are two of the same record_id_eia in scaled_de,
        # but with different data sources we need to drop duplicates so we
        # don't end up with duplicates in the end.
        .drop_duplicates()
    )
    # find matches where the ownership if off but it is the deprish
    # side that has the totals
    # de_bad_totals = own_df[
    #     own_df.ownership_off
    #     & (own_df.record_id_eia_de.str.contains("_total_"))
    # ]
    # if not de_bad_totals.empty:
    #     raise AssertionError(
    #         "The depreciation data in these records are connected to "
    #         "total ownership EIA plant-part records and should "
    #         "probably be hooked up to the owned slice of EIA plant-part"
    #         f"records: {de_bad_totals}"
    # )

    # convert the ppe columns from the ferc-eia data
    # seperate the 'good' records (that can be merged w/o dealing with ownership)
    fe_own_good = scaled_ferc1_eia.drop(
        index=own_df[own_df.ownership_off].record_id_eia_fe
    )
    # grab the record ids for the records that need to be scaled
    fe_own_off = scaled_ferc1_eia.loc[own_df[own_df.ownership_off].record_id_eia_fe]

    # convert the index (which is the record_id_eia)
    fe_own_off.index = fe_own_off.index.str.replace("_total_", "_owned_")
    # replace the columns from the ppe
    ppe_cols = [c for c in fe_own_off if c in plant_parts_eia]
    fe_own_off.loc[:, ppe_cols] = plant_parts_eia.loc[fe_own_off.index, ppe_cols]

    # the columns that need to be scaled are the same allocator cols
    # from the PlantPartScaler
    scale_cols = PlantPartScaler(
        treatments=META_FERC1_EIA,
        ppe_pk=["record_id_eia"],
        data_set_pk_cols=["record_id_ferc1"],
        plant_part="plant_gen",
    ).allocator_cols_dict.keys()

    # actually scale the columns!!
    fe_own_off.loc[:, scale_cols] = fe_own_off.loc[:, scale_cols].multiply(
        fe_own_off.loc[:, "fraction_owned"], axis="index"
    )

    # squish the goodies and the cleaned baddies back together
    scaled_fe_cleaned = pd.concat([fe_own_good, fe_own_off]).sort_index()
    # the output should be exactly the same len
    if len(scaled_fe_cleaned) != len(scaled_ferc1_eia):
        raise AssertionError("The scaled output should be the same length as the input")
    return scaled_fe_cleaned


def _make_record_id_eia_wo_ownership(scaled_df):
    """Make a record_id_eia col.. w/o ownership."""
    scaled_df = (
        scaled_df.assign(
            record_id_eia_wo_ownership=lambda x: x.index.str.replace(
                "owned_", ""
            ).str.replace("total_", ""),
        )
        .reset_index()
        .set_index(["record_id_eia_wo_ownership"])[["record_id_eia", "ownership_dupe"]]
    )
    return scaled_df


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

    treatment_type: Literal["scale", "str_concat", "wtavg"]

    @validator("treatment_type")
    def check_treatments(cls, value, values):  # noqa: N805
        """Check treatments that need additional info."""
        if value == "scale" and not values.get("allocator_cols"):
            raise AssertionError("Scale column treatment needs allocator_cols")
        if value == "wtavg" and not values.get("wtavg_col"):
            raise AssertionError("Weighted Average column treatment needs wtavg_col")
        return value


class PlantPartScaler(BaseModel):
    """
    Scale a table to a plant-part.

    Args:
        treatments: a dictionary of column name (keys) with field treatments
            (values)
        ppe_pk: Identifing columns for the EIA plant-part list.
        data_set_pk_cols: Identifing columns for dataset to scale.
        plant_part: name of EIA plant-part to scale to. Current implementation
            only allows for ``plant_gen``.

    """

    treatments: Dict[str, FieldTreatment]
    ppe_pk: List[str] = ["record_id_eia"]
    data_set_pk_cols: List[str]
    plant_part: Literal["plant_gen"]

    def _get_treatment_cols(self, treatment_type: str) -> List[str]:
        """Grab the columns which need a specific treatment type."""
        return [
            col
            for (col, treat) in self.treatments.items()
            if treat.treatment_type == treatment_type
        ]

    @property
    def wtavg_dict(self) -> Dict:
        """Grab the dict of columns that get a weighted average treatment."""
        return {
            wtavg_col: self.treatments[wtavg_col].wtavg_col
            for wtavg_col in self._get_treatment_cols("wtavg")
        }

    @property
    def allocator_cols_dict(self) -> Dict:
        """Grab the columns from the metadata which need to be allocated."""
        return {
            allocate_col: self.treatments[allocate_col].allocator_cols
            for allocate_col in self._get_treatment_cols("scale")
        }

    @property
    def plant_part_pk_cols(self):
        """Get the primary keys for a plant-part."""
        return (
            pudl.analysis.plant_parts_eia.PLANT_PARTS[self.plant_part]["id_cols"]
            + pudl.analysis.plant_parts_eia.IDX_TO_ADD
            + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
        )

    def execute(
        self, df_to_scale: pd.DataFrame, plant_parts_eia: pd.DataFrame
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
            ppe: the EIA plant-parts.

        """
        # extract the records that are NOT connected to the EIA plant-parts
        # Note: Right now we are just dropping the non-connected
        # not_connected = df_to_scale[df_to_scale.record_id_eia.isnull()]
        connected_to_scale = df_to_scale[~df_to_scale.record_id_eia.isnull()]
        # STEP 1
        # Aggregate when there is more than one source record associated with
        # the same EIA plant-part.
        to_scale = self.aggregate_duplicate_eia(connected_to_scale, plant_parts_eia)
        # STEP 2 & 3
        allocated = self.allocate(to_scale, plant_parts_eia)
        # STEP 4
        # second aggregation of the duplicate EIA records.
        scaled_df_post_agg = self.aggregate_duplicate_eia(
            connected_to_scale=allocated.reset_index(), plant_parts_eia=plant_parts_eia
        )
        # set the index to be the main EIA plant-part index columns
        scaled_df_post_agg = scaled_df_post_agg.set_index(["record_id_eia"])
        return scaled_df_post_agg

    def allocate(
        self, to_allocate: pd.DataFrame, plant_parts_eia: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Allocate records w/ larger granularities to sub-component plant-parts.

        Implement both steps 2 and 3 as described in
        :meth:`PlantPartScaler.execute`.

        Args:
            to_allocate: the dataframe to be allocated.
            plant_parts_eia: EIA plant-parts generated from
                ``pudl.analysis.plant_parts_eia``
        """
        # STEP 2
        merged_df = self.broadcast_merge_to_plant_part(
            data_to_scale=to_allocate,
            ppe=plant_parts_eia.reset_index(),
            cols_to_keep=list(self.treatments),
        )
        # at this stage we have plant-part records with the data columns from
        # the original dataset broadcast across multiple records: read the data
        # is duplicated!

        # STEP 3
        # grab all of the ppe columns, plus data set's id column(s)
        # this enables us to have a unique index
        # pk_cols = self.plant_part_pk_cols + self.data_set_pk_cols
        allocated = merged_df.copy()  # .set_index(pk_cols)
        allocated = allocate_cols(
            to_allocate=allocated,
            by=self.data_set_pk_cols,
            data_and_allocator_cols=self.allocator_cols_dict,
        )
        return allocated

    def aggregate_duplicate_eia(self, connected_to_scale, plant_parts_eia):
        """Aggregate duplicate EIA plant-part records."""
        connected_to_scale = pudl.analysis.plant_parts_eia.reassign_id_ownership_dupes(
            connected_to_scale
        )
        dupe_mask = connected_to_scale.duplicated(subset=self.ppe_pk, keep=False)
        # two dfs
        dupes = connected_to_scale.loc[dupe_mask]
        non_dupes = connected_to_scale.loc[~dupe_mask]
        # If there are no duplicate records, then the following aggs will fail
        # bc there is nothing to merge. So we're making a new df to output that
        # is these non_dupes. If ther are dupes, we'll aggregate them!
        if dupes.empty:
            df_out = non_dupes
        else:
            logger.info(
                f"Aggergating {len(dupes)} duplicate records "
                f"({len(dupes)/len(connected_to_scale):.1%})"
            )

            # sum and weighted average!
            de_duped = pudl.helpers.sum_and_weighted_average_agg(
                df_in=dupes,
                by=self.ppe_pk,
                sum_cols=self._get_treatment_cols("scale"),
                wtavg_dict=self.wtavg_dict,
            ).pipe(pudl.metadata.fields.apply_pudl_dtypes)
            # add in the string columns
            # TODO: add a test to ensure that the str-squish character doesn't
            # show up in the original data columns
            de_duped = de_duped.merge(
                (
                    dupes.groupby(self.ppe_pk, as_index=False).agg(
                        {k: str_concat for k in self._get_treatment_cols("str_concat")}
                    )
                ),
                on=self.ppe_pk,
                validate="1:1",
                how="left",
            ).pipe(pudl.metadata.fields.apply_pudl_dtypes)

            # merge back in the ppe idx columns
            de_duped = (
                de_duped.set_index("record_id_eia")
                .merge(
                    plant_parts_eia,
                    left_index=True,
                    right_index=True,
                    how="left",
                    validate="m:1",
                )
                .reset_index()
            )
            # merge the non-dupes and de-duplicated records
            # we're doing an inner merge here bc we don't want columns with
            # partially null values
            df_out = pd.concat([non_dupes, de_duped], join="inner")
        return df_out

    def broadcast_merge_to_plant_part(
        self, data_to_scale: pd.DataFrame, cols_to_keep: List[str], ppe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Broadcast data with a variety of granularities to a single plant-part.

        This method merges an input dataframe (``data_to_scale``) containing
        data that has a heterogeneous set of plant-part granularities with a
        subset of the EIA plant-parts that has a single granularity. (Currently
        this single granularity must be generators). In general this will be a
        one-to-many merge in which values from single records in the input data
        end up associated with several records from the EIA plant parts.

        First, we select a subset of the full EIA plant-parts corresponding to
        the plant-part of the :class:`PlantPartScaler` instance.  (specified
        by its :attr:`plant_part`). In theory this could be the plant,
        generator, fuel type, etc. Currently only generators are supported.

        Then, we iterate over all the possible plant parts, selecting the
        subset of records in ``data_to_scale`` that have that granularity, and
        merge the homogeneous subset of the EIA plant parts that we selected
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
                EIA plant-parts but they may be heterogeneous in its
                plant-part granularities (i.e. some records could be of 'plant'
                plant-part type while others are 'plant_gen' or
                'plant_prime_mover').  All of the plant-parts columns need
                to be present in this table.
            cols_to_keep: columns from the original data ``data_to_scale`` that
                you want to show up in the output. These should not be columns
                that show up in the ``ppe``.
            ppe: the EIA plant-parts.

        Returns:
            A dataframe in which records correspond to :attr:`plant_part` (in
            the current implementation: the records all correspond to EIA
            generators!). This is an intermediate table that cannot be used
            directly for analysis because the data columns from the original
            dataset are duplicated and still need to be scaled up/down.

        """
        # select only the plant-part records that we are trying to scale to
        ppe_part_df = ppe[ppe.plant_part == self.plant_part]
        # convert the date to year start - this is necessary because the
        # depreciation data is often reported as EOY and the ppe is always SOY
        data_to_scale.loc[:, "report_date"] = pd.to_datetime(
            data_to_scale.report_date.dt.year, format="%Y"
        )
        out_dfs = []
        for merge_part in pudl.analysis.plant_parts_eia.PLANT_PARTS:
            pk_cols = (
                pudl.analysis.plant_parts_eia.PLANT_PARTS[merge_part]["id_cols"]
                + pudl.analysis.plant_parts_eia.IDX_TO_ADD
                + pudl.analysis.plant_parts_eia.IDX_OWN_TO_ADD
            )
            part_df = pd.merge(
                (
                    # select just the records that correspond to merge_part
                    data_to_scale[data_to_scale.plant_part == merge_part][
                        pk_cols + ["record_id_eia"] + cols_to_keep
                    ]
                ),
                ppe_part_df,
                on=pk_cols,
                how="left",
                # this unfortunately needs to be a m:m bc sometimes the df
                # data_to_scale has multiple record associated with the same
                # record_id_eia but are unique records and are not aggregated
                # in aggregate_duplicate_eia. For instance, the depreciation
                # data has both PUC and FERC studies.
                validate="m:m",
                suffixes=("_og", ""),
            )
            out_dfs.append(part_df)
        out_df = pd.concat(out_dfs)
        return out_df


def str_concat(x):
    """Concatenate list of strings with a semicolon-space delimiter."""
    return "; ".join(list(map(str, [x for x in x.unique() if x is not pd.NA])))


def allocate_cols(
    to_allocate: pd.DataFrame, by: list, data_and_allocator_cols: dict
) -> pd.DataFrame:
    """
    Allocate larger dataset records porportionally by EIA plant-part columns.

    Args:
        to_allocate: table of data that has been merged with the EIA plant-parts
            records of the scale that you want the output to be in.
        by: columns to group by.
        data_and_allocator_cols: dict of data columns that you want to allocate (keys)
            and ordered list of columns to allocate porportionally based on. Values
            ordered based on priority: if non-null result from frist column, result
            will include first column result, then second and so on.

    Returns:
        an augmented version of ``to_allocate`` with the data columns (keys in
        ``data_and_allocator_cols``) allocated proportionally.

    """
    # add a total column for all of the allocate cols.
    all_allocator_cols = list(set(sum(data_and_allocator_cols.values(), [])))
    to_allocate.loc[:, [f"{c}_total" for c in all_allocator_cols]] = (
        to_allocate.groupby(by=by, dropna=False)[all_allocator_cols]
        .transform(sum, min_count=1)
        .add_suffix("_total")
    )
    # for each of the columns we want to allocate the frc data by
    # generate the % of the total group, so we can allocate the data_col
    to_allocate = to_allocate.assign(
        **{
            f"{col}_proportion": to_allocate[col] / to_allocate[f"{col}_total"]
            for col in all_allocator_cols
        }
    )
    # do the allocation for each of the data columns
    for data_col in data_and_allocator_cols:
        output_col = f"{data_col}_allocated"
        to_allocate.loc[:, output_col] = pd.NA
        # choose the first non-null option. The order of the allocate_cols will
        # determine which allocate_col will be used
        for allocator_col in data_and_allocator_cols[data_col]:
            to_allocate[output_col] = to_allocate[output_col].fillna(
                to_allocate[data_col] * to_allocate[f"{allocator_col}_proportion"]
            )
    # drop and rename all the columns in the data_and_allocator_cols dict keys and
    # return these columns in the dataframe
    to_allocate = (
        to_allocate.drop(columns=list(data_and_allocator_cols.keys()))
        .rename(
            columns={
                f"{data_col}_allocated": data_col
                for data_col in data_and_allocator_cols
            }
        )
        .drop(
            columns=list(to_allocate.filter(like="_proportion").columns)
            + [f"{c}_total" for c in all_allocator_cols]
        )
    )
    return to_allocate
