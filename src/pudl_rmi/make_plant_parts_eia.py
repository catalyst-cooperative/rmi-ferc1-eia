"""
Aggregate plant parts to make an EIA master unit list.

The EIA data about power plants (from EIA 923 and 860) is reported in tables
with records that correspond to mostly generators and plants. Practically
speaking, a plant is a collection of generator(s). There are many attributes
of generators (i.e. prime mover, primary fuel source, technology type). We can
use these generator attributes to group generator records into larger aggregate
records which we call "plant parts. A "plant part" is a record which
corresponds to a particular collection of generators that all share an
identical attribute. E.g. all of the generators with unit_id=2, or all of the
generators with coal as their primary fuel source.

Because generators are often owned by multiple utilities, another dimention of
the master unit list involves generating two records for each owner: one of the
portion of the plant part they own and one for the plant part as a whole. The
portion records are labeled in the `ownership` column as "owned" and the total
records are labeled as "total".

This module refers to "true granularies". Many plant parts we cobble together
here in the master unit list refer to the same collection of infrastructure as
other master unit list records. For example, if we have a "plant_prime_mover"
plant part record and a "plant_unit" plant part record which were both cobbled
together from the same two generators. We want to be able to reduce the master
unit list to only unique collections of generators, so we label the first
unique granularity as a true granularity and label the subsequent records as
false granularities with the `true_gran` column. In order to choose which plant
part to keep in these instances, we assigned a `plant_parts_ordered` and
effectively keep the first instance of a unique granularity.

Overview of flow for generating the master unit list:

There are two classes in here - one which compiles input tables (CompileTables)
and one which compiles the master unit list (CompilePlantParts). The method
that rules the show here is `generate_master_unit_list`, which is a method of
CompilePlantParts.

`PLANT_PARTS` is basically the main recipe book for how each of the plant parts
need to be compiled. CompilePlantParts eats this recipe book and follows the
recipe to make the master unit list.

All of the plant parts are compiled from generators. So we first generate a
big dataframe of generators with any columns we'll need. This is where we add
records regarding utility ownership slices. Then we use that generator
dataframe and information stored in `PLANT_PARTS` to know how to aggregate each
of the plant parts. Then we have plant part dataframes with the columns which
identify the plant part and all of the data columns aggregated to the level of
the plant part.

With that compiled plant part dataframe we also add in qualifier columns with
`get_qualifiers()`. A qualifer column is a column which contain data that is
not endemic to the plant part record (it is not one of the identifying columns
or aggregated data columns) but the data is still useful data that is
attributable to each of the plant part records. For more detail on what a
qualifier column is, see the `get_qualifiers()` method.
"""


import logging
import pathlib
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd

import pudl

logger = logging.getLogger(__name__)


PLANT_PARTS = {
    'plant': {
        'id_cols': ['plant_id_eia'],
        'false_grans': None
    },
    'plant_gen': {
        'id_cols': ['plant_id_eia', 'generator_id'],
        'false_grans': ['plant', 'plant_unit']
    },
    'plant_unit': {
        'id_cols': ['plant_id_eia', 'unit_id_pudl'],
        'false_grans': ['plant']
    },
    'plant_technology': {
        'id_cols': ['plant_id_eia', 'technology_description'],
        'false_grans': ['plant_prime_mover', 'plant_gen', 'plant_unit', 'plant'
                        ]
    },
    'plant_prime_fuel': {
        'id_cols': ['plant_id_eia', 'energy_source_code_1'],
        'false_grans': ['plant_technology', 'plant_prime_mover', 'plant_gen',
                        'plant_unit', 'plant']
    },
    'plant_prime_mover': {
        'id_cols': ['plant_id_eia', 'prime_mover_code'],
        'false_grans': ['plant_ferc_acct', 'plant_gen', 'plant_unit', 'plant']
    },
    'plant_ferc_acct': {
        'id_cols': ['plant_id_eia', 'ferc_acct_name'],
        'false_grans': ['plant_gen', 'plant_unit', 'plant']
    },
    #    'plant_install_year': {
    #        'id_cols': ['plant_id_eia', 'installation_year'],
    #        'false_grans': ['plant_gen', 'plant_unit', 'plant'],
    #    },
}
"""
dict: this dictionary contains a key for each of the 'plant parts' that should
end up in the mater unit list. The top-level value for each key is another
dictionary, which contains seven keys:
    * id_cols (the primary key type id columns for this plant part),
    * false_grans (the list of other plant parts to check against for whether
    or not the records are in this plant part are false granularities)
"""

SUM_COLS = [
    'total_fuel_cost',
    'net_generation_mwh',
    'capacity_mw',
    'total_mmbtu',
]
"""
iterable: list of columns to sum when aggregating a table.
"""

WTAVG_DICT = {
    'fuel_cost_per_mwh': 'capacity_mw',
    'heat_rate_mmbtu_mwh': 'capacity_mw',
    'fuel_cost_per_mmbtu': 'capacity_mw',
}
"""
dict: a dictionary of columns (keys) to perform weighted averages on and
the weight column (values)"""


QUAL_RECORDS = [
    'fuel_type_code_pudl',
    'operational_status',
    'planned_retirement_date',
    'generator_id',
    'unit_id_pudl',
    'technology_description',
    'energy_source_code_1',
    'prime_mover_code',
    'ferc_acct_name',
    # 'installation_year'
]
"""
dict: a dictionary of qualifier column name (key) and original table (value).
"""

MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]

DTYPES_MUL = {
    "plant_id_eia": "int64",
    "report_date": "datetime64[ns]",
    "plant_part": "object",
    "generator_id": "object",
    "unit_id_pudl": "object",
    "prime_mover_code": "object",
    "energy_source_code_1": "object",
    "technology_description": "object",
    "ferc_acct_name": "object",
    "utility_id_eia": "object",
    "true_gran": "bool",
    "appro_part_label": "object",
    "appro_record_id_eia": "object",
    "capacity_factor": "float64",
    "capacity_mw": "float64",
    "fraction_owned": "float64",
    "fuel_cost_per_mmbtu": "float64",
    "fuel_cost_per_mwh": "float64",
    "heat_rate_mmbtu_mwh": "float64",
    "installation_year": "Int64",
    "net_generation_mwh": "float64",
    "ownership": "category",
    "plant_id_pudl": "Int64",
    "plant_name_eia": "string",
    "total_fuel_cost": "float64",
    "total_mmbtu": "float64",
    "utility_id_pudl": "Int64",
    "utility_name_eia": "string",
    "report_year": "int64",
    "plant_id_report_year": "object",
    "plant_name_new": "string"
}

FIRST_COLS = ['plant_id_eia', 'report_date', 'plant_part', 'generator_id',
              'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
              'technology_description', 'ferc_acct_name',
              'utility_id_eia', 'true_gran', 'appro_part_label']


class CompilePlantParts(object):
    """Compile plant parts."""

    def __init__(self, pudl_out):
        """
        Compile the plant parts for the master unit list.

        Args:
            pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
                the tables for EIA and FERC Form 1 analysis.
            clobber (bool) : if True, you will clobber plant_parts_df (the
                master unit list)

        """
        self.pudl_out = pudl_out
        self.freq = self.pudl_out.freq

        self.plant_gen_df = None
        self.part_true_gran_labels = None
        self.plant_parts_df = None
        self.plant_parts_ordered = [
            'plant', 'plant_unit', 'plant_prime_mover', 'plant_technology',
            'plant_prime_fuel', 'plant_ferc_acct', 'plant_gen'
        ]
        self.parts_to_parent_parts = self.get_parts_to_parent_parts()
        # make a dictionary with the main id column (key) corresponding to the
        # plant part (values)
        self.ids_to_parts = {}
        for part, part_dict in PLANT_PARTS.items():
            self.ids_to_parts[PLANT_PARTS[part]['id_cols'][-1]] = part

        self.parts_to_ids = {v: k for k, v
                             in self.ids_to_parts.items()}
        # get a list of all of the id columns that constitue the primary keys
        # for all of the plant parts
        self.id_cols_list = (
            ['report_date'] +
            dedupe_n_flatten_list_of_lists(
                [x['id_cols'] for x in PLANT_PARTS.values()])
        )

    def generate_master_plant_parts(self):
        """
        Aggreate and slice data points by each plant part.

        This method generates a master list of different "plant-parts", which
        are various collections of generators - i.e. units, fuel-types, whole
        plants, etc. - as well as various ownership arrangements. Each
        plant-part is included in the master plant-part table associated with
        each of the plant-part's owner twice - once with the data scaled to the
        fraction of each owners' ownership and another for a total plant-part
        for each owner.

        This master plant parts table is generated by first creating a complete
        generators table - with all of the data columns we will be aggregating
        to different plant-part's and sliced and scaled by ownership. Then we
        make a label for each plant-part record which indicates whether or not
        the record is a unique grouping of generator records. Then we use the
        complete generator table to aggregate by each of the plant-part
        categories.

        Returns:
            pandas.DataFrame:
        """
        # make the master generator table
        self.plant_gen_df = self.prep_plant_gen_df()
        # generate the true granularity labels
        self.part_true_gran_labels = self.label_true_granularities()
        # 3) aggreate everything by each plant part
        plant_parts_df = pd.DataFrame()
        for part_name in self.plant_parts_ordered:
            part_df = self.get_part_df(part_name)
            # add in the qualifier records
            for qual_record in QUAL_RECORDS:
                part_df = self.get_qualifiers(
                    part_df, part_name, qual_record
                )
            plant_parts_df = plant_parts_df.append(part_df, sort=True)
        # clean up, add additional columns
        self.plant_parts_df = (
            self.add_additonal_cols(plant_parts_df)
            .pipe(pudl.helpers.organize_cols, FIRST_COLS)
            .pipe(self._clean_plant_parts)
        )
        self.test_ownership_for_owned_records(self.plant_parts_df)
        return self.plant_parts_df

    def make_id_cols_dict(self):
        """Make a dict of part to corresponding peer part id columns."""
        id_cols_dict = {}
        for part, i in zip(self.plant_parts_ordered,
                           range(1, len(self.plant_parts_ordered) + 1)):
            logger.debug(part)
            ids = set({'report_date'})
            for peer in self.plant_parts_ordered[i:]:
                for id_col in PLANT_PARTS[peer]['id_cols']:
                    logger.debug(f'   {id_col}')
                    ids.add(id_col)
            for part_id_col in PLANT_PARTS[part]['id_cols']:
                logger.debug(f'   {part_id_col}')
                ids.add(part_id_col)
            id_cols_dict[part] = list(ids)
        return id_cols_dict

    def get_parts_to_parent_parts(self):
        """
        Make a dictionary of each plant-part's parent parts.

        We have imposed a hierarchy on the plant-parts with the
        ``plant_parts_ordered`` and this method generates a dictionary of each
        plant-part's (key) parent-parts (value).
        """
        parts_to_parent_parts = {}
        n = 0
        for part_name in self.plant_parts_ordered:
            parts_to_parent_parts[part_name] = self.plant_parts_ordered[:n]
            n = n + 1
        return parts_to_parent_parts

    def make_fake_totals(self, plant_gen_df):
        """Generate total versions of generation-owner records."""
        # make new records for generators to replicate the total generator
        fake_totals = plant_gen_df[[
            'plant_id_eia', 'report_date', 'utility_id_eia',
            'owner_utility_id_eia']].drop_duplicates()
        # asign 1 to all of the fraction_owned column
        fake_totals = fake_totals.assign(fraction_owned=1,
                                         ownership='total')
        fake_totals = pd.merge(
            plant_gen_df.drop(
                columns=['ownership', 'utility_id_eia',
                         'owner_utility_id_eia', 'fraction_owned']),
            fake_totals)
        return fake_totals

    def slice_by_ownership(self, plant_gen_df):
        """Generate proportional data by ownership %s."""
        own860 = (
            self.pudl_out.own_eia860()
            [['plant_id_eia', 'generator_id', 'report_date',
              'fraction_owned', 'owner_utility_id_eia']]
            .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        )

        logger.debug(f'# of generators before munging: {len(plant_gen_df)}')
        plant_gen_df = plant_gen_df.merge(
            own860,
            how='outer',
            on=['plant_id_eia', 'generator_id', 'report_date'],
            indicator=True
        )

        # if there are records that don't show up in the ownership table (and
        # have a 'left_only' merge indicator and NaN for 'fraction_owned'),
        # we're going to assume the ownership % is 1 for the reporting utility
        if len(plant_gen_df[(plant_gen_df['_merge'] == 'right_only') &
                            (plant_gen_df['fraction_owned'].isnull())]) != 0:
            # if there are records with null 'fraction_owned' then we've done
            # something wrong.
            raise AssertionError(
                'merge error: ownership and gens produced with null records')
        # clean the remaining nulls
        # assign 100% ownership for records not in the ownership table
        plant_gen_df = plant_gen_df.assign(
            fraction_owned=plant_gen_df.fraction_owned.fillna(value=1),
            # assign the operator id as the owner if null
            owner_utility_id_eia=plant_gen_df.owner_utility_id_eia.fillna(
                plant_gen_df.utility_id_eia),
            ownership='owned'
        )

        fake_totals = self.make_fake_totals(plant_gen_df)

        plant_gen_df = plant_gen_df.append(fake_totals, sort=False)
        logger.debug(f'# of generators post-fakes:     {len(plant_gen_df)}')
        plant_gen_df = (
            plant_gen_df.drop(columns=['_merge', 'utility_id_eia']).
            rename(columns={'owner_utility_id_eia': 'utility_id_eia'}).
            drop_duplicates())

        cols_to_cast = ['net_generation_mwh', 'capacity_mw', 'total_fuel_cost']
        plant_gen_df[cols_to_cast] = (plant_gen_df[cols_to_cast].
                                      multiply(plant_gen_df['fraction_owned'],
                                               axis='index'))
        if (len(plant_gen_df[plant_gen_df.ownership == 'owned']) >
                len(plant_gen_df[plant_gen_df.ownership == 'total'])):
            warnings.warn(
                'There should be more records labeled as total.')
        return plant_gen_df

    def _ag_fraction_owned(self, part_ag, id_cols):
        """
        Calculate the fraction owned for a plant-part df.

        This method takes a dataframe of records that are aggregated to the
        level of a plant-part (with certain `id_cols`) and appends a
        fraction_owned column, which indicates the % ownership that a
        particular utility owner has for each aggreated plant-part record.

        For partial owner records (ownership == "owned"), fraction_owned is
        calcuated based on the portion of the capacity and the total capacity
        of the plant. For total owner records (ownership == "total"), the
        fraction_owned is always 1.

        This method is meant to be run within `ag_part_by_own_slice()`.

        Args:
            part_ag (pandas.DataFrame):
            id_cols (list): list of identifying columns
                (stored as: `PLANT_PARTS[part_name]['id_cols']`)
        """
        # we must first get the total capacity of the full plant
        # Note: we could simply not include the ownership == "total" records
        # We are automatically assign fraction_owned == 1 to them, but it seems
        # cleaner to run the full df through this same grouby
        frac_owned = (
            part_ag.groupby(by=id_cols + ['ownership', 'report_date'])
            [['capacity_mw']].sum(min_count=1)
        )
        # then merge the total capacity with the plant-part capacity to use to
        # calculate the fraction_owned
        part_frac = (
            pd.merge(part_ag,
                     frac_owned,
                     right_index=True,
                     left_on=frac_owned.index.names,
                     suffixes=("", "_total")
                     )
            .assign(fraction_owned=lambda x:
                    np.where(x.ownership == 'owned',
                             x.capacity_mw / x.capacity_mw_total,
                             1
                             ))
            .drop(columns=['capacity_mw_total'])
        )
        return part_frac

    def ag_part_by_own_slice(self, part_name):
        """
        Aggregate the plant part by seperating ownership types.

        There are total records and owned records in this master unit list.
        Those records need to be aggregated differently to scale. The total
        owned slice is now grouped and aggregated as a single version of the
        full plant and then the utilities are merged back. The owned slice is
        grouped and aggregated with the utility_id_eia, so the portions of
        generators created by slice_by_ownership will be appropriately
        aggregated to each plant part level.

        Args:
            part_name (str): the name of the part to aggregate to. Names can be
                only those in `PLANT_PARTS`

        Returns:
            pandas.DataFrame : dataframe aggregated to the level of the
                part_name
        """
        plant_part = PLANT_PARTS[part_name]
        logger.info(f'begin aggregation for: {part_name}')
        id_cols = plant_part['id_cols']
        # split up the 'owned' slices from the 'total' slices.
        # this is because the aggregations are different
        plant_gen_df = self.prep_plant_gen_df()
        part_own = plant_gen_df[plant_gen_df.ownership == 'owned']
        part_tot = plant_gen_df[plant_gen_df.ownership == 'total']
        if len(plant_gen_df) != len(part_own) + len(part_tot):
            raise AssertionError(
                "Error occured in breaking apart ownership types."
                "The total and owned slices should equal the total records."
                "Check for nulls in the ownership column."
            )
        dedup_cols = list(part_tot.columns)
        dedup_cols.remove('utility_id_eia')
        dedup_cols.remove('unit_id_pudl')
        part_tot = part_tot.drop_duplicates(subset=dedup_cols)
        part_own = agg_cols(
            df_in=part_own,
            id_cols=id_cols + ['utility_id_eia', 'ownership'],
            sum_cols=SUM_COLS,
            wtavg_dict=WTAVG_DICT,
            freq=self.freq
        )
        # still need to re-calc the fraction owned for the part
        part_tot = (
            agg_cols(
                df_in=part_tot,
                id_cols=id_cols,
                sum_cols=SUM_COLS,
                wtavg_dict=WTAVG_DICT,
                freq=self.freq
            )
            .merge(plant_gen_df[id_cols + ['report_date', 'utility_id_eia']]
                   .dropna()
                   .drop_duplicates())
            .assign(ownership='total')
        )
        part_ag = (
            part_own.append(part_tot, sort=False)
            .pipe(self._ag_fraction_owned, id_cols=id_cols)
        )

        return part_ag

    def add_additonal_cols(self, plant_parts_df):
        """
        Add additonal data and id columns.

        capacity_factor +
        utility_id_pudl +
        plant_id_pudl +
        ownership_dupe (boolean): indicates whether the "owned" record has a
        corresponding "total" duplicate.
        """
        plant_parts_df = (
            calc_capacity_factor(plant_parts_df, -0.5, 1.5, self.freq)
            .merge(
                self.pudl_out.plants_eia860()
                [['plant_id_eia', 'plant_id_pudl',
                  'utility_id_eia', 'utility_id_pudl']]
                .drop_duplicates(),
                how='left',
                on=['plant_id_eia', 'utility_id_eia']
            )
            .assign(ownership_dupe=lambda x: np.where(
                (x.ownership == 'owned') & (x.fraction_owned == 1),
                True, False)
            )
        )
        return plant_parts_df

    def add_record_id(self, part_df, id_cols, plant_part_col='plant_part'):
        """
        Add a record id to a compiled part df.

        We need a standardized way to refer to these compiled records that
        contains enough information in the id itself that in theory we could
        deconstruct the id and determine which plant id and plant part id
        columns are associated with this record.
        """
        ids = deepcopy(id_cols)
        # we want the plant id first... mostly just bc it'll be easier to read
        part_df = part_df.assign(record_id_eia=part_df.plant_id_eia.map(str))
        ids.remove('plant_id_eia')
        for col in ids:
            part_df = part_df.assign(
                record_id_eia=part_df.record_id_eia + "_" +
                part_df[col].astype(str))
        part_df = (
            part_df.assign(
                record_id_eia=part_df.record_id_eia + "_" +
                part_df.report_date.dt.year.astype(str) + "_" +
                part_df[plant_part_col] + "_" +
                part_df.ownership.astype(str) + "_" +
                part_df.utility_id_eia.astype('Int64').astype(str))
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']))
        return part_df

    def get_consistent_qualifiers(self, record_df, base_cols, record_name):
        """
        Get fully consistent qualifier records.

        When data is a qualifer column is identical for every record in a
        plant part, we associate this data point with the record. If the data
        points for the related generator records are not identical, then
        nothing is associated with the record.

        Args:
            record_df (pandas.DataFrame): the dataframe with the record
            base_cols (list) : list of identifying columns.
            record_name (string) : name of qualitative record
        """
        # TODO: determine if we can move this up the chain so we can do this
        # once per plant-part, not once per plant-part * qualifer record
        entity_count_df = (
            pudl.helpers.count_records(
                record_df, base_cols, 'entity_occurences')
            .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        )
        record_count_df = (
            pudl.helpers.count_records(
                record_df, base_cols + [record_name], 'record_occurences')
            . pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        )
        re_count = (
            record_df[base_cols + [record_name]]
            .merge(entity_count_df, how='left', on=base_cols)
            .merge(record_count_df, how='left', on=base_cols + [record_name])
        )
        # find all of the matching records..
        consistent_records = (
            re_count[
                re_count['entity_occurences'] == re_count['record_occurences']]
            .drop(columns=['entity_occurences', 'record_occurences'])
            .drop_duplicates())
        return consistent_records

    def dedup_on_category(self, dedup_df, base_cols, category_name, sorter):
        """
        Deduplicate a df using a sorted category to retain prefered values.

        Use a sorted category column to retain your prefered values when a
        dataframe is deduplicated.

        Args:
            dedup_df (pandas.DataFrame): the dataframe with the record
            base_cols (list) : list of identifying columns
            category_name (string) : name of qualitative record
            sorter (list): sorted list of category options
        """
        dedup_df[category_name] = dedup_df[category_name].astype("category")
        dedup_df[category_name].cat.set_categories(sorter, inplace=True)
        dedup_df = dedup_df.sort_values(category_name)
        return dedup_df.drop_duplicates(subset=base_cols, keep='first')

    def get_qualifiers(self, part_df, part_name, record_name):
        """
        Get qualifier records.

        For an individual dataframe of one plant part (e.g. only
        "plant_prime_mover" plant part records), we typically have identifying
        columns and aggregated data columns. The identifying columns for a
        given plant part are only those columns which are required to uniquely
        specify a record of that type of plant part. For example, to uniquely
        specify a plant_unit record, we need both plant_id_eia and the
        unit_id_pudl, and nothing else. In other words, the identifying columns
        for a given plant part would make up a natural composite primary key
        for a table composed entirely of that type of plant part. Every plant
        part is cobbled together from generator records, so each record in
        each part_df can be thought of as a collection of generators.

        Identifier and qualifier columns are the same columns; whether a column
        is an identifier or a qualifier is a function of the plant part you're
        considering. All the other columns which could be identifiers in the
        context of other plant parrts (but aren't for this plant part) are
        qualifiers.

        This method takes a part_df and goes and checks whether or not the data
        we are trying to grab from the record_name column is consistent across
        every component genertor from each record.

        When record_name is "operational_status", we are not going to check for
        consistency; instead we will grab the highest level of operational
        status that is associated with each records' component generators. The
        order of operational status is defined within the method as:
        'existing', 'proposed', then 'retired'. For example if a plant_unit is
        composed of two generators, and one of them is "existing" and another
        is "retired" the entire plant_unit will be considered "existing".

        Args:
            part_df (pandas.DataFrame): dataframe containing records associated
                with one plant part.
            part_name (string): name of plant-part.

        """
        if record_name in part_df.columns:
            logger.debug(f'{record_name} already here.. ')
            return part_df

        record_df = self.prep_plant_gen_df().copy()

        # the base columns will be the id columns, plus the other two main ids
        id_cols = PLANT_PARTS[part_name]['id_cols']
        base_cols = id_cols + ['ownership', 'report_date']

        if record_name != 'operational_status':
            consistent_records = self.get_consistent_qualifiers(
                record_df, base_cols, record_name)
        if record_name == 'operational_status':
            logger.debug(f'getting max {record_name}')
            sorter = ['existing', 'proposed', 'retired']
            # restric the number of columns in here to only include the ones we
            # need, unlike get_consistent_qualifiers, dedup_on_category
            # preserves all of the columns from record_df
            record_df = record_df[base_cols + [record_name]]
            consistent_records = self.dedup_on_category(
                record_df, base_cols, record_name, sorter
            )
        non_nulls = consistent_records[
            consistent_records[record_name].notnull()]
        logger.debug(
            f'merging in consistent {record_name}: {len(non_nulls)}')
        return part_df.merge(consistent_records, how='left')

    def label_true_granularities(self, drop_extra_cols=True, clobber=False):
        """
        Prep the table that denotes true_gran for all generators.

        This method will generate a dataframe based on ``self.plant_gen_df``
        that has boolean columns that denotes whether each plant-part is a true
        or false granularity.

        There are four main steps in this process:
          * For every combinations of plant-parts, count the number of unique
            types of peer plant-parts (see ``make_all_the_counts()`` for more
            details).
          * Convert those counts to boolean values if there is more or less
            than one unique type parent or child plant-part (see
            ``make_all_the_bools()`` for more details).
          * Using the boolean values label each plant-part as a True or False
            granularies if both the boolean for the parent-to-child and
            child-to-parent (see ``label_true_grans_by_part()`` for more
            details).
          * For each plant-part, label it with its the appropriate plant-part
            counterpart - if it is a True granularity, the appropriate label is
            itself (see ``label_true_id_by_part()`` for more details).

        Args:
            drop_extra_cols (boolean): if True, the extra columns used to
                generate the true_gran columns. Default is True.
            clobber (boolean)

        """
        if self.part_true_gran_labels is None or clobber:
            self.part_true_gran_labels = (
                self.make_all_the_counts()
                .pipe(self.make_all_the_bools)
                .pipe(self.label_true_grans_by_part)
                .pipe(self.label_true_id_by_part)
            )

            if drop_extra_cols:
                for drop_cols in ['_v_', '_has_only_one_', 'count_per']:
                    self.part_true_gran_labels = (
                        self.part_true_gran_labels.drop(
                            columns=self.part_true_gran_labels
                            .filter(like=drop_cols)
                        )
                    )
        return self.part_true_gran_labels

    def count_child_and_parent_parts(self, part_name, count_ids):
        """
        Count the child- and parent-parts contained within a plant-part.

        Args:
            part_name (string): name of plant-part
            count_ids (pandas.DataFrame): a table of generator records with

        Returns:
            pandas.DataFrame: an agumented version of the ``plant_gen_df``
            dataframe with new columns for each of the child and parent
            plant-parts with counts of unique instances of those parts. The
            columns will be named in the following format:
            {child/parent_part_name}_count_per_{part_name}

        """
        part_cols = PLANT_PARTS[part_name]['id_cols'] + ['report_date']
        # because the plant_id_eia is always a part of the groupby columns
        # and we want to count the plants as well, we need to make a duplicate
        # plant_id_eia column to count on
        df_count = (
            count_ids.assign(plant_id_eia_temp=lambda x: x.plant_id_eia)
            .groupby(by=part_cols, dropna=False).nunique()
            .rename(columns={'plant_id_eia_temp': 'plant_id_eia'})
            .rename(columns=self.ids_to_parts)
            .add_suffix(f'_count_per_{part_name}')
        )
        # merge back into the og df
        df_w_count = count_ids.merge(
            df_count,
            how='left',
            right_index=True,
            left_on=part_cols,
        )
        return df_w_count

    def make_all_the_counts(self):
        """
        For each plant-part, count the unique child and parent parts.

        Returns:
            pandas.DataFrame: an agumented version of the ``plant_gen_df``
            dataframe with new columns for each of the child and parent
            plant-parts with counts of unique instances of those parts. The
            columns will be named in the following format:
            {child/parent_part_name}_count_per_{part_name}
        """
        # grab the plant-part id columns from the generator table
        count_ids = (
            self.prep_plant_gen_df()
            .loc[:, self.id_cols_list]
            .drop_duplicates()
        )
        # we want to compile the count results on a copy of the generator table
        all_the_counts = self.prep_plant_gen_df().copy()
        for part_name in self.plant_parts_ordered:
            logger.debug(f"making the counts for: {part_name}")
            all_the_counts = all_the_counts.merge(
                self.count_child_and_parent_parts(part_name, count_ids),
                how='left')

        # check the expected # of columns
        # id columns minus the report_date column
        pp_l = len(self.id_cols_list) - 1
        expected_col_len = (
            len(self.prep_plant_gen_df().columns) +  # the plant_gen_df colums
            pp_l * (pp_l - 1) + 1  # the count columns (we add one bc we get a
            # stragger plant_count_per_plant column bc we make that
            # plant_id_eia_temp column)
        )
        if expected_col_len != len(all_the_counts.columns):
            raise AssertionError(
                f"We got {len(all_the_counts.columns)} columns from "
                f"all_the_counts when we should have gotten {expected_col_len}"
            )
        return all_the_counts

    def make_all_the_bools(self, counts):
        """
        Count consistency of records and convert that to bools.

        Args:
            all_the_counts (pandas.DataFrame): result of
                ``make_all_the_counts()``

        Returns:
            pandas.DataFrame: a table with generator records where we have new
            boolean columns which indicated whether or not the plant-part
            has more than one child/parent-part. These columns are formated
            as: {child/parent_part_name}_has_only_one_{part_name}

        """
        counts.loc[:, counts.filter(like='_count_per_').columns] = (
            counts.loc[:, counts.filter(like='_count_per_').columns]
            .astype(pd.Int64Dtype())
        )

        # convert the count columns to bool columns
        for col in counts.filter(like='_count_per_').columns:
            bool_col = col.replace("_count_per_", "_has_only_one_")
            counts.loc[counts[col].notnull(), bool_col] = counts[col] == 1
        # force the nullable bool type for all our count cols
        counts.loc[:, counts.filter(like='_has_only_one_').columns] = (
            counts.filter(like='_has_only_one_').astype(pd.BooleanDtype())
        )
        return counts

    def label_true_grans_by_part(self, part_bools):
        """
        Label the true/false granularies for each part/parent-part combo.

        This method uses the indicator columns which let us know whether or not
        there are more than one unique value for both the parent and child
        plant-part ids to generate an additional indicator column that let's us
        know whether the child plant-part is a true or false granularity when
        compared to the parent plant-part. With all of the indicator columns
        from each plant-part's parent plant-parts, if all of those determined
        that the plant-part is a true granularity, then this method will label
        the plant-part as being a true granulary and vice versa.

        Because we have forced a hierarchy within the ``plant_parts_ordered``,
        the process for labeling true or false granularities must investigate
        bi-directionally. This is because all of the plant-parts besides
        'plant' and 'plant_gen' are not necessarily bigger of smaller than
        their parent plant-part and thus there is overlap. Because of this,
        this method uses the checks in both directions (from partent to child
        and from child to parent).

        Args:
            part_bools (pandas.DataFrame): result of ``make_all_the_bools()``
        """
        # assign a bool for the true gran only if all
        for part_name, parent_parts in self.parts_to_parent_parts.items():
            for parent_part_name in parent_parts:
                # let's save the input boolean columns
                bool_cols = [f'{part_name}_has_only_one_{parent_part_name}',
                             f'{parent_part_name}_has_only_one_{part_name}']
                false_gran_col = f'false_gran_{part_name}_v_{parent_part_name}'
                # the long awaited ALL.. label them as
                part_bools[false_gran_col] = (
                    part_bools[bool_cols].all(axis='columns'))
                part_bools = part_bools.astype(
                    {false_gran_col: pd.BooleanDtype()})
                # create the inverse column as true_grans
                part_bools[f'true_gran_{part_name}_v_{parent_part_name}'] = (
                    ~part_bools[false_gran_col])
            # if all of the true_gran part v parent part columns are false,
            # than this part is a false gran. if they are all true, then wahoo
            # the record is truly unique
            part_bools[f'true_gran_{part_name}'] = (
                part_bools.filter(like=f'true_gran_{part_name}')
                .all(axis='columns'))
            trues_found = (
                part_bools[part_bools[f'true_gran_{part_name}']]
                .drop_duplicates(subset=[self.parts_to_ids[part_name],
                                         'plant_id_eia', 'report_date']))
            logger.info(
                f'true grans found for {part_name}: {len(trues_found)}'
            )
        part_trues = part_bools.drop(
            columns=part_bools.filter(like='false_gran').columns)
        return part_trues

    def label_true_id_by_part(self, part_trues):
        """
        Label the appropriate plant-part.

        For each plant-part, we need to make a label which indicates what the
        "true" unique plant-part is.. if a gen vs a unit is a non-unique set a
        records, we only want to label one of them as false granularities. We
        are going to use the ``parts_to_parent_parts`` dictionary to help us
        with this. We want to "save" the biggest parent plant-part as true
        granularity.

        Because we have columns in ``part_trues`` that indicate whether a
        plant-part is a true gran vs each parent part, we can cycle through
        the possible parent-parts from biggest to smallest and the first time
        we find that a plant-part is a false gran, we label it's true id as
        that parent-part.
        """
        for part_name, parent_parts in self.parts_to_parent_parts.items():
            # make column which will indicate which part is the true/unique
            # plant-part...
            appro_part_col = f"appro_part_label_{part_name}"
            # make this col null so we can fill in
            part_trues[appro_part_col] = pd.NA
            for parent_part_name in parent_parts:
                # find the reords where the true gran col is false, and label
                # the appropriate part column name with that parent part
                mask_loc = (
                    ~part_trues[f'true_gran_{part_name}_v_{parent_part_name}'],
                    appro_part_col
                )
                part_trues.loc[mask_loc] = part_trues.loc[mask_loc].fillna(
                    parent_part_name)
            # for all of the plant-part records which we didn't find any false
            # gran's the appropriate label is itself! it is a unique snowflake
            part_trues[appro_part_col] = part_trues[appro_part_col].fillna(
                part_name)
            part_trues = (
                self.assign_record_id_eia(part_trues,
                                          plant_part_col=appro_part_col)
                .rename(columns={'record_id_eia': f'record_id_eia_{part_name}'}
                        )
            )
            # do a little check
            if not all(part_trues[part_trues[f'true_gran_{part_name}']]
                       [appro_part_col] == part_name):
                raise AssertionError(
                    f'eeeeEEeEe... the if true_gran_{part_name} is true, the '
                    f'{appro_part_col} should {part_name}.'
                )
        return part_trues

    def assign_record_id_eia(self, test_df, plant_part_col='plant_part'):
        """
        Assign record ids to a df with a mix of plant parts.

        Args:
            test_df (pandas.DataFrame)
            plant_part_col (string)

        """
        test_df_ids = pd.DataFrame()
        for part in PLANT_PARTS:
            test_df_ids = pd.concat(
                [test_df_ids,
                 self.add_record_id(
                     part_df=test_df[test_df[plant_part_col] == part],
                     id_cols=PLANT_PARTS[part]['id_cols'],
                     plant_part_col=plant_part_col
                 )])
        return test_df_ids

    def assign_true_gran(self, part_df, part_name):
        """
        Merge the true granularity labels into the plant part df.

        Args:
            part_df (pandas.DataFrame)
            part_name (string)

        """
        bool_df = self.label_true_granularities()
        # get only the columns you need for this part and drop duplicates
        bool_df = (
            bool_df[
                PLANT_PARTS[part_name]['id_cols'] +
                ['report_date',
                 f'true_gran_{part_name}',
                 f'appro_part_label_{part_name}',
                 f'record_id_eia_{part_name}',
                 'utility_id_eia',
                 'ownership']]
            .drop_duplicates()
        )

        prop_true_len1 = len(
            bool_df[bool_df[f'true_gran_{part_name}']]) / len(bool_df)
        logger.debug(f'proportion of trues: {prop_true_len1:.02}')
        logger.debug(f'number of records pre-merge:  {len(part_df)}')

        part_df = (part_df.
                   merge(bool_df, how='left').
                   rename(columns={
                       f'true_gran_{part_name}': 'true_gran',
                       f'appro_part_label_{part_name}': 'appro_part_label',
                       f'record_id_eia_{part_name}': 'appro_record_id_eia'
                   }))

        prop_true_len2 = len(part_df[part_df.true_gran]) / len(part_df)
        logger.debug(f'proportion of trues: {prop_true_len2:.02}')
        logger.debug(f'number of records post-merge: {len(part_df)}')
        return part_df

    def _clean_plant_parts(self, plant_parts_df):
        return (
            plant_parts_df.
            assign(report_year=lambda x: x.report_date.dt.year,
                   plant_id_report_year=lambda x: x.plant_id_pudl.astype(str)
                   + "_" + x.report_year.astype(str)).
            # pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']).
            # we'll eventually take this out... once Issue #20
            drop_duplicates(subset=['record_id_eia']).
            set_index('record_id_eia'))

    def add_new_plant_name(self, part_df, id_cols, part_name):
        """
        Add plants names into the compiled plant part df.

        Args:
            part_df (pandas.DataFrame)
            part_name (string)
        """
        part_df = (
            pd.merge(
                part_df,
                self.prep_plant_gen_df()
                [id_cols + ['plant_name_eia']].drop_duplicates(),
                on=id_cols,
                how='left'
            )
            .assign(plant_name_new=lambda x: x.plant_name_eia))
        # we don't want the plant_id_eia to be part of the plant name, but all
        # of the other parts should have their id column in the new plant name
        if part_name != 'plant':
            col = self.parts_to_ids[part_name]
            part_df.loc[part_df[col].notnull(), 'plant_name_new'] = (
                part_df['plant_name_new'] + " " + part_df[col].astype(str))
        return part_df

    def add_record_count(self, part_df):
        """
        Add a record count for each set of plant part records in each plant.

        Args:
            part_df (pandas.DataFrame): dataframe containing records associated
                with one plant part.
        """
        group_cols = ['plant_id_eia', 'utility_id_eia',
                      'report_date', 'ownership']
        # count unique records per plant
        part_count = (part_df.groupby(group_cols)
                      [['record_id_eia']].count()
                      .rename(columns={'record_id_eia': 'record_count'})
                      .reset_index()
                      )
        part_df = pd.merge(part_df, part_count,
                           on=group_cols, how='left'
                           )
        return part_df

    def prep_plant_gen_df(self, clobber=False):
        """
        Prepare plant gen dataframe.

        Get a table of all of the generators with all of their id columns and
        data columns, sliced by ownership which makes "total" and "owned"
        records for each generator owner.
        """
        if self.plant_gen_df is None or clobber:
            logger.info(
                'Generating the mega generator table with ownership.')
            self.plant_gen_df = (
                self.get_mega_gens_table()
                .pipe(self.slice_by_ownership)
            )
        return self.plant_gen_df

    def get_mega_gens_table(self):
        """
        Compile the main generators table that will be used as base of MUL.

        Get a table of all of the generators there ever were and all of the
        data PUDL has to offer about those generators. This generator table
        will be used to compile all of the "plant-parts", so we need to ensure
        that any of the id columns from the other plant-parts are in this
        generator table as well as all of the data columns that we are going to
        aggregate to the various plant-parts.

        Returns:
            pandas.DataFrame
        """
        # pull in the main two tables
        gens = self.pudl_out.gens_eia860()
        mcoe = self.pudl_out.mcoe()

        # because lots of these input dfs include same info columns, this
        # generates drop columnss for fuel_cost. This avoids needing to hard
        # code columns.
        merge_cols = ['plant_id_eia', 'generator_id', 'report_date']
        drop_cols = [x for x in mcoe if x in gens and x not in merge_cols]

        all_gens = (
            pd.merge(
                gens.pipe(pudl.helpers.convert_cols_dtypes, 'eia'),
                mcoe.drop(drop_cols, axis=1)
                .pipe(pudl.helpers.convert_cols_dtypes, 'eia'),
                on=merge_cols,
                validate='1:1',
                how='left'
            )
            .merge(
                get_eia_ferc_acct_map(),
                on=['technology_description', 'prime_mover_code'],
                validate='m:1',
                how='left'
            )
            .assign(installation_year=lambda x: x.operating_date.dt.year)
            .astype({'installation_year': 'Int64', })
        )

        # check to see if the master gens table has all of the columns we want
        # extract columns from PLANT_PARTS + a few extra
        other_cols = [
            'plant_name_eia',
            'installation_year',
            'utility_id_eia',
            'fuel_type_code_pudl',
            'operational_status',
            'planned_retirement_date'
        ]
        all_cols = (
            self.id_cols_list + SUM_COLS + list(WTAVG_DICT.keys()) + other_cols
        )

        missing = [c for c in all_cols if c not in all_gens]
        if missing:
            raise AssertionError(
                f'The main generator table is missing {missing}'
            )
        # bb test to ensure that we are getting all of the possible records
        # w/ net generation
        generation = self.pudl_out.gen_eia923()
        assert (
            len(generation[generation.net_generation_mwh.notnull()]) ==
            len(all_gens[all_gens.net_generation_mwh.notnull()]
                .drop_duplicates(
                    subset=['plant_id_eia', 'report_date', 'generator_id']
            ))
        )
        return all_gens[all_cols]

    def get_part_df(self, part_name):
        """
        Get a table of data aggregated by a specific plant-part.

        This method will use ``plant_gen_df`` (or generate if it doesn't
        exist yet) to aggregate the generator records to the level of the
        plant-part. This is mostly done via ``ag_part_by_own_slice()``. Then
        several additional columns are added and the records are labeled as
        true or false granularities.

        Args:
            part_name (string): name of plant-part
        """
        plant_part = PLANT_PARTS[part_name]
        id_cols = plant_part['id_cols']

        part_df = (
            self.ag_part_by_own_slice(part_name)
            .assign(plant_part=part_name)
            .pipe(self.add_install_year, id_cols)
            .pipe(self.assign_true_gran, part_name)
            .pipe(self.add_record_id, id_cols, plant_part_col='plant_part')
            .pipe(self.add_new_plant_name, id_cols, part_name)
            .pipe(self.add_record_count)
        )
        return part_df

    def add_install_year(self, part_df, id_cols):
        """Add the install year from the entities table to your plant part."""
        logger.debug(f'pre count of part DataFrame: {len(part_df)}')
        # we want to sort to have the most recent on top
        install = (
            self.plant_gen_df[id_cols + ['installation_year']]
            .sort_values('installation_year', ascending=False)
            .drop_duplicates(subset=id_cols, keep='first')
            .dropna(subset=id_cols)
        )
        part_df = part_df.merge(
            install, how='left',
            on=id_cols, validate='m:1')
        logger.debug(
            f'count of install years for part: {len(install)} \n'
            f'post count of part DataFrame: {len(part_df)}'
        )
        return part_df

    def test_ownership_for_owned_records(self, plant_parts_df):
        """
        Test ownership - fraction owned for owned records.

        This test can be run at the end of or with the result of
        `generate_master_unit_list()`. It tests a few aspects of the the
        fraction_owned column and raises assertions if the tests fail.
        """
        test_own_df = (
            plant_parts_df.groupby(
                by=self.id_cols_list + ['plant_part', 'ownership'],
                dropna=False
            )
            [['fraction_owned', 'capacity_mw']].sum(min_count=1).reset_index())

        owned_one_frac = test_own_df[
            (~np.isclose(test_own_df.fraction_owned, 1))
            & (test_own_df.capacity_mw != 0)
            & (test_own_df.capacity_mw.notnull())
            & (test_own_df.ownership == 'owned')]

        if not owned_one_frac.empty:
            self.test_own_df = test_own_df
            self.owned_one_frac = owned_one_frac
            raise AssertionError(
                "Hello friend, you did bad. It happens... Error with the "
                "fraction_owned col/slice_by_ownership(). There are "
                f"{len(owned_one_frac)} rows where fraction_owned != 1 for "
                "owned records. Check cached `owned_one_frac` & `test_own_df`"
            )

        no_frac_n_cap = test_own_df[
            (test_own_df.capacity_mw == 0)
            & (test_own_df.fraction_owned == 0)
        ]
        if len(no_frac_n_cap) > 60:
            self.no_frac_n_cap = no_frac_n_cap
            warnings.warn(
                f"""Too many nothings, you nothing. There shouldn't been much
                more than 60 instances of records with zero capacity_mw (and
                therefor zero fraction_owned) and you got {len(no_frac_n_cap)}.
                """
            )

    def _test_prep_merge(self, part_name):
        """Run the test groupby and merge with the aggregations."""
        id_cols = PLANT_PARTS[part_name]['id_cols']
        plant_cap = (
            self.prep_plant_gen_df()
            .groupby(
                by=id_cols + ['report_date', 'utility_id_eia', 'ownership'])
            [SUM_COLS]
            .sum(min_count=1)
            .reset_index()
        )
        plant_cap = self.dedup_on_category(
            plant_cap,
            category_name='ownership',
            base_cols=[x for x in plant_cap.columns if x not in [
                'record_id_eia', 'ownership', 'appro_record_id_eia', ]],
            sorter=['owned', 'total']
        )

        test_merge = pd.merge(
            self.plant_parts_df[self.plant_parts_df.plant_part == part_name],
            plant_cap,
            on=id_cols + ['report_date', 'utility_id_eia', 'ownership'],
            how='outer',
            indicator=True,
            suffixes=('', '_test'))
        return test_merge

    def _test_col_bool(self, test_merge, test_col):
        """
        Check if the test aggregation is the same as the preped aggreation.

        Apply a boolean column to the test df.
        """
        test_merge[f'test_{test_col}'] = (
            (test_merge[f'{test_col}_test'] == test_merge[f'{test_col}'])
            & test_merge[f'{test_col}_test'].notnull()
            & test_merge[f'{test_col}'].notnull()
        )
        result = list(test_merge[f'test_{test_col}'].unique())
        logger.info(f'  Results for {test_col}: {result}')
        return test_merge

    def test_sum_cols(self, part_name):
        """
        For a compiled plant-part df, re-run groubys and check similarity.

        Args:
            part_name (string)
        """
        test_merge = self._test_prep_merge(part_name)
        for test_col in SUM_COLS:
            test_merge = self._test_col_bool(test_merge, test_col)
        return test_merge

    def test_run_aggregations(self):
        """
        Run a test of the aggregated columns.

        This test will used the self.plant_parts_df, re-run groubys and check
        similarity. This should only be run after running
        generate_master_unit_list().
        """
        for part_name in self.plant_parts_ordered:
            logger.info(f'Begining tests for {part_name}:')
            self.test_sum_cols(part_name)


def calc_capacity_factor(df, min_cap_fact, max_cap_fact, freq):
    """
    Calculate capacity factor.

    TODO: Move this into pudl.helpers and incorporate into mcoe.capacity_factor
    """
    # get a unique set of dates to generate the number of hours
    dates = df['report_date'].drop_duplicates()

    # merge in the hours for the calculation
    df = df.merge(pd.DataFrame(
        data={'report_date': dates,
              'hours': dates.apply(
                  lambda d: (
                      pd.date_range(d, periods=2, freq=freq)[1] -
                      pd.date_range(d, periods=2, freq=freq)[0]) /
                  pd.Timedelta(hours=1))}), on=['report_date'])

    # actually calculate capacity factor wooo!
    df['capacity_factor'] = df['net_generation_mwh'] / \
        (df['capacity_mw'] * df['hours'])

    # Replace unrealistic capacity factors with NaN
    df.loc[df['capacity_factor'] < min_cap_fact, 'capacity_factor'] = np.nan
    df.loc[df['capacity_factor'] >= max_cap_fact, 'capacity_factor'] = np.nan

    # drop the hours column, cause we don't need it anymore
    df.drop(['hours'], axis=1, inplace=True)
    return df


def weighted_average(df, data_col, weight_col, by_col):
    """Generate a weighted average."""
    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = (
        g['_data_times_weight'].sum(min_count=1)
        / g['_weight_where_notnull'].sum(min_count=1)
    )
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result.to_frame(name=data_col).reset_index()


def agg_cols(df_in, id_cols, sum_cols, wtavg_dict, freq):
    """
    Aggregate dataframe by summing and using weighted averages.

    Args:
        df_in (pandas.DataFrame): table to aggregate. Must have columns

    """
    cols_to_grab = id_cols
    cols_to_grab = list(
        set([x for x in cols_to_grab if x in list(df_in.columns)]))
    if cols_to_grab != cols_to_grab:
        warnings.warn(f"um {[x for x in id_cols if x not in cols_to_grab]}")
    # Not totally sure if this freq functionally still works.. haven't used it
    # in a while.
    df_in = df_in.astype({'report_date': 'datetime64[ns]'})
    if 'report_date' in list(df_in.columns):
        if len(df_in[df_in['report_date'].dt.month > 2]) > 0:
            cols_to_grab = cols_to_grab + [pd.Grouper(freq=freq)]
            df_in = df_in.set_index(pd.DatetimeIndex(df_in.report_date))
        else:
            cols_to_grab = cols_to_grab + ['report_date']
    logger.debug('aggregate the parts')
    logger.debug(f'     grouping by on {cols_to_grab}')

    df_out = (
        df_in.groupby(by=cols_to_grab, as_index=False)
        [sum_cols]
        .sum(min_count=1)
    )

    for data_col, weight_col in wtavg_dict.items():
        df_out = weighted_average(
            df_in,
            data_col=data_col,
            weight_col=weight_col,
            by_col=cols_to_grab
        ).merge(df_out, how='outer', on=cols_to_grab)
    return df_out


def get_eia_ferc_acct_map():
    """
    Get map of EIA technology_description/pm codes <> ferc accounts.

    We must refactor this with a better path dependency. Or just store this
    map as a dataframe or dictionary.
    """
    file_path = pathlib.Path().cwd().parent / 'inputs' / \
        'ferc_acct_to_pm_tech_map.csv'

    eia_ferc_acct_map = (
        pd.read_csv(file_path)
        [['technology_description', 'prime_mover_code', 'ferc_acct_name']]
        .drop_duplicates()
    )
    return eia_ferc_acct_map


def reassign_id_ownership_dupes(plant_parts_df):
    """
    Reassign the record_id for the records that are labeled ownership_dupe.

    Args:
        plant_parts_df (pandas.DataFrame): master unit list. Result of
            ``generate_master_unit_list()`` or ``get_master_unit_list_eia()``.
            Must have boolean column ``ownership_dupe`` and string column or
            index of ``record_id_eia``.

    """
    # the record_id_eia's need to be a column to mess with it and record_id_eia
    # is typically the index of plant_parts_df, so we are going to reset index
    # if record_id_eia is the index
    og_index = False
    if plant_parts_df.index.name == "record_id_eia":
        plant_parts_df = plant_parts_df.reset_index()
        og_index = True

    plant_parts_df = plant_parts_df.assign(record_id_eia=lambda x: np.where(
        x.ownership_dupe,
        x.record_id_eia.str.replace("owned", "total"),
        x.record_id_eia))
    # then we reset the index so we return the dataframe in the same structure
    # as we got it.
    if og_index:
        plant_parts_df = plant_parts_df.set_index("record_id_eia")
    return plant_parts_df


def get_master_unit_list_eia(file_path_mul, pudl_out, clobber=False):
    """
    Get the master unit list; generate it or get if from a file.

    If you generate the MUL, it will be saved at the file path given.

    Args:
        file_path_mul (pathlib.Path): where you want the master unit list to
            live. Must be a compressed pickle file ('.pkl.gz').
        clobber (boolean): True if you want to regenerate the master unit list
            whether or not it is saved at the file_path_mul
    """
    if '.pkl' not in file_path_mul.suffixes:
        raise AssertionError(f"{file_path_mul} must be a pickle file")
    if not file_path_mul.is_file() or clobber:
        logger.info(
            f"Master unit list not found {file_path_mul}"
            "Generating a new master unit list. This should take ~10 minutes."
        )

        parts_compilers = CompilePlantParts(pudl_out)
        plant_parts_df = parts_compilers.generate_master_plant_parts()
        plant_parts_df.to_csv(file_path_mul, compression='gzip')

    elif file_path_mul.is_file():
        logger.info(f"Reading the master unit list from {file_path_mul}")
        plant_parts_df = pd.read_pickle(file_path_mul, compression='gzip')
    return plant_parts_df


def dedupe_n_flatten_list_of_lists(mega_list):
    """Flatten a list of lists and remove duplicates."""
    return list(set(
        [item for sublist in mega_list for item in sublist]))
