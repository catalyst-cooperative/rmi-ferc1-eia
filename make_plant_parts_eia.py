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
import sqlalchemy as sa

import pudl
import pudl.constants as pc

logger = logging.getLogger(__name__)

PLANT_PARTS = {
    'plant': {
        'id_cols': ['plant_id_eia'],
        'denorm_table': None,
        'denorm_cols': None,
        'install_table': None,
        'false_grans': None,
        'ag_cols': {
            'total_fuel_cost': 'sum',
            'net_generation_mwh': 'sum',
            'capacity_mw': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
    'plant_gen': {
        'id_cols': ['plant_id_eia', 'generator_id'],
        # unit_id_pudl are associated with plant_ids & plant_ids/generator_ids
        'denorm_table': None,
        'denorm_cols': None,
        'install_table': None,
        'false_grans': ['plant', 'plant_unit'],
        'ag_cols': {
            'capacity_mw': pudl.helpers.sum_na,
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
        'ag_tables': {
            'generation_eia923': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'net_generation_mwh': 'sum',
                },
                'wtavg_cols': None,
            },
            'generators_eia860': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'capacity_mw': 'sum',
                },
                'wtavg_cols': None,
            },
            'mcoe': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'total_fuel_cost': 'sum',
                    'total_mmbtu': 'sum'
                },
                'wtavg_cols': {
                    'fuel_cost_per_mwh': 'capacity_mw',  # 'wtavg_mwh',
                    'heat_rate_mmbtu_mwh': 'capacity_mw',  # 'wtavg_mwh',
                    'fuel_cost_per_mmbtu': 'capacity_mw',  # 'wtavg_mwh',
                },

            }
        },
    },
    'plant_unit': {
        'id_cols': ['plant_id_eia', 'unit_id_pudl'],
        # unit_id_pudl are associated with plant_ids & plant_ids/generator_ids
        'denorm_table': 'boiler_generator_assn_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'boiler_generator_assn_eia860',
        'false_grans': ['plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
    'plant_technology': {
        'id_cols': ['plant_id_eia', 'technology_description'],
        'denorm_table': 'generators_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'generators_eia860',
        'false_grans': ['plant_prime_mover', 'plant_gen', 'plant_unit', 'plant'
                        ],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
    'plant_prime_fuel': {
        'id_cols': ['plant_id_eia', 'energy_source_code_1'],
        'denorm_table': 'generators_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'generators_eia860',
        'false_grans': ['plant_technology', 'plant_prime_mover', 'plant_gen',
                        'plant_unit', 'plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
    'plant_prime_mover': {
        'id_cols': ['plant_id_eia', 'prime_mover_code'],
        'denorm_table': 'generators_entity_eia',
        'denorm_cols': ['plant_id_eia', 'generator_id'],
        'install_table': None,
        'false_grans': ['plant_ferc_acct', 'plant_gen', 'plant_unit', 'plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
    'plant_ferc_acct': {
        'id_cols': ['plant_id_eia', 'ferc_acct_name'],
        'denorm_table': None,  # 'ferc_acct_rmi',
        # ['technology_description', 'prime_mover_code', 'ferc_acct_name'],
        'denorm_cols':  None,
        'install_table': 'ferc_acct_rmi',
        'false_grans': ['plant_gen', 'plant_unit', 'plant'],
        'ag_cols': {
            'total_fuel_cost': 'sum',
            'net_generation_mwh': 'sum',
            'capacity_mw': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
            'fraction_owned': 'capacity_mw',
        },
    },
}
"""
dict: this dictionary contains a key for each of the 'plant parts' that should
end up in the mater unit list. The top-level value for each key is another
dictionary, which contains seven keys:
    * id_cols (the primary key type id columns for this plant part),
    * denorm_table (the table needed to merge into the generator table to get
    the associated id_cols, if not neccesary then None),
    * denorm_cols (the columns needed to merge in the denorm table),
    * install_table (the table needed to merge in the installation year),
    * false_grans (the list of other plant parts to check against for whether
    or not the records are in this plant part are false granularities),
    * ag_cols (a dictionary of the columns to aggregate on with a groupby),
    * wtavg_cols (a dictionary of columns to perform weighted averages on and
    the weight column)
The plant_gen part has an additional item called ag_tables. This is for use in
generating plant_gen_df; it contains tables to merge together to compile all
the neccessary components, how to denormalize, aggregate and perform weighted
averages at the generator level.
"""

FREQ_AG_COLS = {
    'generation_eia923': {
        'id_cols': ['plant_id_eia', 'generator_id'],
        'ag_cols': {'net_generation_mwh': 'sum', },
        'wtavg_cols': None
    },
    'generation_fuel_eia923': {
        'id_cols': ['plant_id_eia', 'nuclear_unit_id',
                    'fuel_type', 'fuel_type_code_pudl',
                    'fuel_type_code_aer', 'prime_mover_code'],
        'ag_cols': {'net_generation_mwh': 'sum', },
        'wtavg_cols': ['fuel_consumed_mmbtu']
    },
    'fuel_receipts_costs_eia923': {
        'id_cols': ['plant_id_eia', 'energy_source_code',
                    'fuel_type_code_pudl', 'fuel_group_code',
                    'fuel_group_code_simple', 'contract_type_code'],
        'ag_cols': None,
        'wtavg_cols': ['fuel_cost_per_mmbtu']
    },
    'generators_eia860': None,
    'boiler_generator_assn_eia860': None,
    'ownership_eia860': None,
    'generators_entity_eia': None,
    'utilities_eia': None,
    'plants_eia': None,
    'energy_source_eia923': None,


}


QUAL_RECORD_TABLES = {
    'fuel_type_code_pudl': 'generators_eia860',
    'operational_status': 'generators_eia860',
    'planned_retirement_date': 'generators_eia860',
    'generator_id': 'generators_eia860',
    'unit_id_pudl': 'generators_eia860',
    'technology_description': 'generators_eia860',
    'energy_source_code_1': 'generators_eia860',
    'prime_mover_code': 'generators_eia860',
    'ferc_acct_name': 'generators_eia860',

}
"""
dict: a dictionary of qualifier column name (key) and original table (value).
"""

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


class CompileTables(object):
    """Compile tables from sqlite db or pudl output object."""

    def __init__(self, pudl_engine, freq=None, start_date=None, end_date=None,
                 roll=True, fill=True):
        """
        Initialize a table compiler.

        gets the stuff....
        """
        self.pudl_engine = pudl_engine
        self.freq = freq

        if start_date is None:
            self.start_date = \
                pd.to_datetime(
                    '{}-01-01'.format(min(pc.working_years['eia923'])))
        else:
            # Make sure it's a date... and not a string.
            self.start_date = pd.to_datetime(start_date)

        if end_date is None:
            self.end_date = \
                pd.to_datetime(
                    '{}-12-31'.format(max(pc.working_years['eia923'])))
        else:
            # Make sure it's a date... and not a string.
            self.end_date = pd.to_datetime(end_date)

        if not pudl_engine:
            raise AssertionError('PudlTabl object needs a pudl_engine')
        self.pudl_engine = pudl_engine

        # geting the metadata object for the sqlite db
        self.pt = pudl.output.pudltabl.get_table_meta(self.pudl_engine)

        self.pudl_out = pudl.output.pudltabl.PudlTabl(
            pudl_engine=pudl_engine, freq=self.freq, roll=roll, fill=fill)

        self._dfs = {
            # pudl sqlite tables
            'generation_fuel_eia923': None,
            'fuel_receipts_costs_eia923': None,
            'generators_eia860': None,
            'boiler_generator_assn_eia860': None,
            'generation_eia923': None,
            'ownership_eia860': None,
            'generators_entity_eia': None,
            'utilities_eia': None,
            'plants_eia': None,
            'energy_source_eia923': None,
            # pudl_out tables
            'fuel_cost': None,
            'mcoe': None,
            'plants_steam_ferc1': None,

            'ferc_acct_rmi': None,
        }

    def get_the_table(self, table):
        """Get a dataframe from the sqlite tables or the output object."""
        if table is None:
            return

        if self._dfs[table] is None:
            # this is going to try to see if the table is in the db
            # if pt[table] is not None:
            try:
                tbl = self.pt[table]
                logger.info(f'getting {table} from the sqlite db')
                select = sa.sql.select([tbl, ])
                # for the non-date tables...
                if 'report_date' not in tbl.columns:
                    logger.debug('getting a non-date table')
                    df = pd.read_sql(select, self.pudl_engine)
                else:
                    if self.start_date is not None:
                        select = select.where(
                            tbl.c.report_date >= self.start_date)
                    if self.end_date is not None:
                        select = select.where(
                            tbl.c.report_date <= self.end_date)
                    df = pd.read_sql(select, self.pudl_engine, parse_dates=[
                                     'report_date'], index_col=['id'])

                # if we have a freq and a table is reported annually..aggregate
                if self.freq is not None and FREQ_AG_COLS[table] is not None:
                    df = self.agg_cols(id_cols=FREQ_AG_COLS[table]['id_cols'] +
                                       ['utility_id_eia', 'fraction_owned'],
                                       ag_cols=FREQ_AG_COLS[table]['ag_cols'],
                                       wtavg_cols=None,
                                       df_in=df)

            # if is it not a database table, it is an output function
            # elif hasattr(pudl_out_eia, table):
            except KeyError:
                # getattr turns the string of the table into an attribute
                # of the object, so this runs the output function
                logger.info(f'getting {table} from the output object')
                # the pudl_out.mcoe function has defaults set for data purity.
                # these defaults were removing ~20% of the fuel cost records.
                # mcoe is the only pudl_out function that is used in the MUL -
                # this if/else enables getting other pudl_out tables w/o args
                if table == 'mcoe':
                    df = self.pudl_out.mcoe(min_heat_rate=None,
                                            min_fuel_cost_per_mwh=None,
                                            min_cap_fact=None,
                                            max_cap_fact=None)
                elif table == 'ferc_acct_rmi':
                    df = get_eia_ferc_acct_map()
                else:
                    df = getattr(self.pudl_out, table,)()
            self._dfs[table] = pudl.helpers.convert_cols_dtypes(df,
                                                                'eia',
                                                                name=table)
        return self._dfs[table]

    def agg_cols(self, id_cols, ag_cols, wtavg_cols, df_in):
        """Aggregate dataframe."""
        cols_to_grab = id_cols
        cols_to_grab = list(set(
            [x for x in cols_to_grab if x in list(df_in.columns)]))
        if 'report_date' in list(df_in.columns):
            if len(df_in[df_in['report_date'].dt.month > 2]) > 0:
                cols_to_grab = cols_to_grab + [pd.Grouper(freq=self.freq)]
                df_in = df_in.set_index(pd.DatetimeIndex(df_in.report_date))
            else:
                cols_to_grab = cols_to_grab + ['report_date']
        logger.debug('aggregate the parts')
        logger.debug(f'     grouping by on {cols_to_grab}')
        logger.debug(f'     agg-ing on by on {ag_cols}')
        # logger.debug(f'     cols in df are {df_in.columns}')
        df_in = df_in.astype({'report_date': 'datetime64[ns]'})
        df_out = (df_in.groupby(by=cols_to_grab).
                  # use the groupby object to aggregate on the ag_cols
                  # this runs whatever function we've defined in the
                  # ag_cols dictionary
                  agg(ag_cols).
                  # reset the index because the output of the agg
                  reset_index())
        if wtavg_cols:
            for data_col, weight_col in wtavg_cols.items():
                df_out = weighted_average(
                    df_in,
                    data_col=data_col,
                    weight_col=weight_col,
                    by_col=cols_to_grab
                ).merge(df_out, how='outer', on=cols_to_grab)
        return df_out


class CompilePlantParts(object):
    """Compile plant parts."""

    def __init__(self, table_compiler, clobber=False):
        """
        Compile the plant parts for the master unit list.

        Args:
            plant_parts (dict): a dictionary of information required to
                aggregate each plant part.
            table_compiler (object)
            clobber (bool) : if True, you will clobber plant_parts_df (the
                master unit list)

        """
        self.table_compiler = table_compiler
        self.freq = table_compiler.freq
        self.plant_parts = PLANT_PARTS
        self.plant_gen_df = None
        self.part_true_gran_labels = None
        self.plant_parts_df = None
        self.clobber = clobber
        self.plant_parts_ordered = ['plant', 'plant_unit',
                                    'plant_prime_mover', 'plant_technology',
                                    'plant_prime_fuel', 'plant_ferc_acct',
                                    'plant_gen',
                                    ]
        self.parts_to_parent_parts = self.get_parts_to_parent_parts()
        self.gen_util_ids = ['plant_id_eia', 'generator_id',
                             'report_date', 'utility_id_eia']
        # make a dictionary with the main id column (key) corresponding to the
        # plant part (values)
        self.ids_to_parts = {}
        for part, part_dict in self.plant_parts.items():
            self.ids_to_parts[self.plant_parts[part]['id_cols'][-1]] = part

        self.id_cols_dict = self.make_id_cols_dict()
        self.id_cols_list = self.get_id_cols_list()
        self.parts_to_ids = {v: k for k, v
                             in self.ids_to_parts.items()}

    def make_id_cols_dict(self):
        """Make a dict of part to corresponding peer part id columns."""
        id_cols_dict = {}
        for part, i in zip(self.plant_parts_ordered,
                           range(1, len(self.plant_parts_ordered) + 1)):
            logger.debug(part)
            ids = set({'report_date'})
            for peer in self.plant_parts_ordered[i:]:
                for id_col in self.plant_parts[peer]['id_cols']:
                    logger.debug(f'   {id_col}')
                    ids.add(id_col)
            for part_id_col in self.plant_parts[part]['id_cols']:
                logger.debug(f'   {part_id_col}')
                ids.add(part_id_col)
            id_cols_dict[part] = list(ids)
        return id_cols_dict

    def get_id_cols_list(self):
        """
        Get all the identifying columns for the plant parts.

        We sometimes need the list of the identifying colums from all of the
        plant parts in order to have quick access to all of the column names.

        Returns:
            list: list of identifying columns for all of the plant parts.
        """
        id_cols_list = []
        for id_cols in self.ids_to_parts.keys():
            id_cols_list = id_cols_list + [id_cols]
        return id_cols_list

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

    def get_denormalize_table(self,
                              table,
                              denorm_table=None,
                              denorm_cols=None,
                              id_cols=None,
                              indicator=False):
        """
        Get and denormalize the table.

        get the table that you want, and merge it with another table based
        on the 'denorm_cols'.

        Args:
            table (string): a table name
            denorm_table (string): the name of the table you want to merge in
            denorm_cols (list): the columns to use to merge the tables
            indicator (bool): True of False for whether or not you want to
                include an indicator column in your merge that notes where
                each row came from.
            id_cols (list): the list of columns that identify the plant
                granularity.
        Returns:
            pandas.Dataframe

        """
        table_df = self.table_compiler.get_the_table(table)
        if denorm_table:
            logger.info(f'denormalizing {table}')
            table_df = self.denoramlize_table(
                table_df, denorm_table, denorm_cols)
        return table_df

    def denoramlize_table(self,
                          table_df,
                          id_cols,
                          denorm_table,
                          denorm_cols,
                          indicator=False):
        """Merge data table with additional table to get additional colums."""
        # denormalize the plant granularity
        table_df = table_df.merge(
            self.table_compiler.get_the_table(denorm_table)
            [list(set(id_cols + denorm_cols))].drop_duplicates(),
            on=denorm_cols,
            how='outer',
            indicator=indicator)
        return(table_df)

    def get_ownership(self):
        """Get the ownership table and create total rows."""
        # get the ownership table and do some munging
        own860 = (self.table_compiler.get_the_table('ownership_eia860')
                  [['plant_id_eia', 'generator_id', 'report_date',
                    'utility_id_eia', 'fraction_owned',
                    'owner_utility_id_eia']])
        return own860

    def aggregate_plant_part(self, plant_part_details):
        """Generate dataframe of aggregated plant part."""
        cols_to_grab = plant_part_details['id_cols'] + ['report_date']
        plant_part_df = pd.DataFrame(columns=cols_to_grab)
        for table_name, table_deets in plant_part_details['ag_tables'].items():
            logger.info(f'beginning the aggregation for {table_name}')
            # get the table
            table = self.get_denormalize_table(
                table_name,
                denorm_table=table_deets['denorm_table'],
                denorm_cols=table_deets['denorm_cols'],
                id_cols=plant_part_details['id_cols'],
            )

            plant_part_df = (self.table_compiler.agg_cols(
                id_cols=plant_part_details['id_cols'] +
                ['utility_id_eia', 'fraction_owned', 'ownership'],
                ag_cols=table_deets['ag_cols'],
                wtavg_cols=table_deets['wtavg_cols'],
                df_in=table)
                .merge(plant_part_df, how='outer')
                .dropna(subset=plant_part_details['id_cols'])
            )
        # if 'utility_id_eia' in plant_part_df.columns:
        plant_part_df = plant_part_df.dropna(subset=['utility_id_eia'])
        return plant_part_df

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
        own860 = self.get_ownership()
        logger.debug(f'# of generators before munging: {len(plant_gen_df)}')
        plant_gen_df = plant_gen_df.merge(
            own860,
            how='outer',
            on=self.gen_util_ids,
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
            raise AssertionError(
                'There should be more records labeled as total.')
        return plant_gen_df

    def denorm_plant_gen(self, qual_records):
        """Denromalize the plant_gen_df."""
        og_len = len(self.plant_gen_df)
        # compile all of the denorm table info in one place
        denorm_tables = {}
        for part in self.plant_parts:
            denorm_table = self.plant_parts[part]['denorm_table']
            if denorm_table:
                if denorm_table in denorm_tables.keys():
                    denorm_tables[denorm_table]['id_cols'] = list(set(
                        denorm_tables[denorm_table]['id_cols'] +
                        self.plant_parts[part]['id_cols']))
                    denorm_tables[denorm_table]['denorm_cols'] = list(set(
                        denorm_tables[denorm_table]['denorm_cols'] +
                        self.plant_parts[part]['denorm_cols']))
                else:
                    denorm_tables[denorm_table] = {
                        'id_cols': self.plant_parts[part]['id_cols'],
                        'denorm_cols': self.plant_parts[part]['denorm_cols']
                    }
        # get the demorn tables and squish the id cols in with the gens
        for k, v in denorm_tables.items():
            if not set(v['id_cols']).issubset(set(self.plant_gen_df.columns)):
                logger.debug(f'no id cols from {k}')
                denorm_df = self.table_compiler.get_the_table(
                    k)[list(set(v['id_cols']
                                + v['denorm_cols']))].drop_duplicates()
                self.plant_gen_df = self.plant_gen_df.merge(
                    denorm_df, how='left')
        if 'plant_ferc_acct' in self.plant_parts.keys():
            self.plant_gen_df = self.add_ferc_acct(self.plant_gen_df)
        # if
        if qual_records:
            # add all the missing qualifiers
            qual_records_missing = [
                x for x in QUAL_RECORD_TABLES.keys()
                if x not in self.plant_gen_df.columns
            ]
            idx_cols_gen = ['plant_id_eia', 'generator_id', 'report_date']
            self.plant_gen_df = pd.merge(
                self.plant_gen_df,
                self.table_compiler.get_the_table('generators_eia860')[
                    idx_cols_gen + qual_records_missing],
                on=idx_cols_gen,
                how='left'
            )
        if og_len != len(self.plant_gen_df):
            warnings.warn(
                'ahh merge error! when adding denorm colunms to '
                'plant_gen_df we must get the same number of records'
                f'og # of records: {og_len} vs  end state #: '
                f'{len(self.plant_gen_df)}'
            )
        return self.plant_gen_df

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
                only those in `self.plant_parts`

        Returns:
            pandas.DataFrame : dataframe aggregated to the level of the
                part_name
        """
        plant_part = self.plant_parts[part_name]
        logger.info(f'begin aggregation for: {part_name}')
        id_cols = plant_part['id_cols']
        ag_cols = plant_part['ag_cols']
        wtavg_cols = plant_part['wtavg_cols']
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
        part_own = self.table_compiler.agg_cols(
            id_cols=id_cols +
            ['utility_id_eia',
             'ownership'],
            ag_cols=ag_cols,
            wtavg_cols=wtavg_cols,
            df_in=part_own)
        # still need to re-calc the fraction owned for the part
        part_tot = (
            self.table_compiler.agg_cols(
                id_cols=id_cols,
                ag_cols=ag_cols,
                wtavg_cols=wtavg_cols,
                df_in=part_tot)
            .merge(plant_gen_df[id_cols + ['report_date', 'utility_id_eia']]
                   .dropna()
                   .drop_duplicates())
            .assign(ownership='total')
        )
        return part_own.append(part_tot, sort=False)

    def add_additonal_cols(self, plant_parts_df):
        """
        Add additonal data and id columns.

        capacity_factor +
        utility_id_pudl +
        plant_id_pudl +

        """
        plant_parts_df = (
            calc_capacity_factor(plant_parts_df, -0.5, 1.5, self.freq).
            merge(self.table_compiler.get_the_table('utilities_eia'),
                  how='left').
            merge(self.table_compiler.get_the_table('plants_eia'), how='left')
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
        part_df = part_df.assign(
            record_id_eia=part_df.record_id_eia + "_" +
            part_df.report_date.dt.year.astype(str) + "_" +
            part_df[plant_part_col] + "_" +
            part_df.ownership.astype(str) + "_" +
            part_df.utility_id_eia.astype('Int64').astype(str))
        return part_df

    def add_install_year(self, part_df, id_cols, install_table):
        """Add the install year from the entities table to your plant part."""
        logger.debug(f'pre count of part DataFrame: {len(part_df)}')
        gen_ent = self.table_compiler.get_the_table('generators_entity_eia')
        install = (gen_ent.
                   assign(installation_year=gen_ent['operating_date'].dt.year).
                   astype({'installation_year': 'Int64', }).
                   # we want to sort to have the most recent on top
                   sort_values('installation_year', ascending=False))
        if not install_table:
            # then the install table has everything we need
            part_install = (install[id_cols + ['installation_year']].
                            drop_duplicates(subset=id_cols, keep='first'))

        elif install_table == 'ferc_acct_rmi':
            part_install = (install[['plant_id_eia', 'generator_id',
                                     'installation_year']].
                            merge(self.prep_plant_gen_df())
                            [id_cols + ['installation_year']].
                            drop_duplicates(subset=id_cols, keep='first'))
        else:
            part_install = (install[['plant_id_eia', 'generator_id',
                                     'installation_year']].
                            merge(self.table_compiler.get_the_table(
                                install_table))
                            [id_cols + ['installation_year']].
                            drop_duplicates(subset=id_cols, keep='first'))
        part_df = part_df.merge(part_install, how='left')
        logger.debug(f'count of install years for part: {len(part_install)}')
        logger.debug(f'post count of part DataFrame: {len(part_df)}')
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
        entity_count_df = (pudl.helpers.count_records(record_df,
                                                      base_cols,
                                                      'entity_occurences').
                           pipe(pudl.helpers.convert_cols_dtypes, 'eia'))
        record_count_df = (pudl.helpers.count_records(record_df,
                                                      base_cols +
                                                      [record_name],
                                                      'record_occurences').
                           pipe(pudl.helpers.convert_cols_dtypes, 'eia'))

        something = (
            record_df[base_cols + [record_name]].
            merge(entity_count_df, how='left', on=base_cols).
            merge(record_count_df, how='left', on=base_cols + [record_name])
        )
        # find all of the matching records..
        consistent_records = (
            something[something['entity_occurences'] ==
                      something['record_occurences']].
            drop(columns=['entity_occurences', 'record_occurences']).
            drop_duplicates())
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

    def get_qualifiers(self,
                       part_df,
                       part_name,
                       record_name,
                       ):
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
        id_cols = self.plant_parts[part_name]['id_cols']
        base_cols = id_cols + ['ownership', 'report_date']

        if record_name != 'operational_status':
            logger.debug(f'getting consistent {record_name}s')
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
        non_nulls = consistent_records[consistent_records[record_name].notnull(
        )]
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
        part_cols = self.plant_parts[part_name]['id_cols'] + ['report_date']
        # because the plant_id_eia is always a part of the groupby columns
        # and we want to count the plants as well, we need to make a duplicate
        # plant_id_eia column to count on
        df_count = (count_ids
                    .assign(plant_id_eia_temp=lambda x: x.plant_id_eia)
                    .groupby(by=part_cols, dropna=False)
                    .nunique()
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
            self.prep_plant_gen_df(
            ).loc[:, self.id_cols_list + ['report_date']]
            .drop_duplicates()
        )
        # we want to compile the count results on a copy of the generator table
        all_the_counts = self.prep_plant_gen_df().copy()
        for part_name in self.plant_parts_ordered:
            logger.info(f"making the counts for: {part_name}")
            all_the_counts = all_the_counts.merge(
                self.count_child_and_parent_parts(part_name, count_ids),
                how='left')

        # check the expected # of columns
        pp_l = len(self.id_cols_list)
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
        for part in self.plant_parts:
            test_df_ids = pd.concat(
                [test_df_ids,
                 self.add_record_id(
                     part_df=test_df[test_df[plant_part_col] == part],
                     id_cols=self.plant_parts[part]['id_cols'],
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
        bool_df = (bool_df[self.plant_parts[part_name]['id_cols'] +
                           ['report_date', f'true_gran_{part_name}',
                            f'appro_part_label_{part_name}',
                            f'record_id_eia_{part_name}',
                            # 'fraction_owned',
                            'utility_id_eia',
                            'ownership'
                            ]].
                   drop_duplicates())

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
            pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia']).
            # we'll eventually take this out... once Issue #20
            drop_duplicates(subset=['record_id_eia']).
            set_index('record_id_eia'))

    def add_new_plant_name(self, part_df, part_name):
        """
        Add plants names into the compiled plant part df.

        Args:
            part_df (pandas.DataFrame)
            part_name (string)
        """
        part_df = (pd.merge(
            part_df, self.table_compiler.get_the_table('plants_eia'))
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

    def add_ferc_acct(self, plant_gen_df):
        """Merge the plant_gen_df with the ferc_acct map."""
        return pd.merge(plant_gen_df, get_eia_ferc_acct_map(), how='left')

    def prep_plant_gen_df(self, qual_records=True, clobber=False):
        """Prepare plant gen dataframe."""
        if self.plant_gen_df is None or clobber:
            logger.info(
                'Generating the master generator table with ownership.')
            self.plant_gen_df = (
                self.aggregate_plant_part(self.plant_parts['plant_gen'])
                .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
                .pipe(self.slice_by_ownership)
            )
            self.plant_gen_df = self.denorm_plant_gen(qual_records)
        return self.plant_gen_df

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
        plant_part = self.plant_parts[part_name]
        id_cols = plant_part['id_cols']

        part_df = (
            self.ag_part_by_own_slice(part_name)
            .assign(plant_part=part_name)
            .pipe(self.add_install_year, id_cols, plant_part['install_table'])
            .pipe(self.assign_true_gran, part_name)
            .pipe(self.add_record_id, id_cols, plant_part_col='plant_part')
            .pipe(self.add_new_plant_name, part_name)
            .pipe(self.add_record_count)
        )
        return part_df

    def generate_master_unit_list(self, qual_records=True, clobber=False):
        """
        Aggreate and slice data points by each plant part.

        1) aggregate the the data points by generator
        2) generating proportional data by ownership %s
        3) aggreate everything by each plant part

        Args:
            qual_records (bool): if True, the master unit list will be
                generated with all consistent qualifer records in
                `QUAL_RECORD_TABLES`. See `get_qualifiers()` for more details.
                Default is True.
            clobber (boolean):

        Returns:
            pandas.DataFrame
        """
        if self.plant_parts_df is not None and not clobber:
            return self.plant_parts_df
        # make the master generator table
        self.plant_gen_df = self.prep_plant_gen_df(qual_records)
        # generate the true granularity labels
        self.part_true_gran_labels = self.label_true_granularities()

        # 3) aggreate everything by each plant part
        plant_parts_df = pd.DataFrame()
        for part_name in self.plant_parts_ordered:
            part_df = self.get_part_df(part_name)
            if qual_records:
                # add in the qualifier records
                for qual_record in QUAL_RECORD_TABLES:
                    logger.debug(
                        f'get consistent {qual_record} for {part_name}')
                    part_df = self.get_qualifiers(
                        part_df, part_name, qual_record
                    )
            plant_parts_df = plant_parts_df.append(part_df, sort=True)
        # clean up, add additional columns, and drop duplicates
        self.plant_parts_df = (
            self.add_additonal_cols(plant_parts_df)
            .pipe(pudl.helpers.organize_cols, FIRST_COLS)
            .pipe(self._clean_plant_parts)
            .pipe(self.dedup_on_category,
                  category_name='ownership',
                  base_cols=[x for x in plant_parts_df.columns if x not in [
                      'record_id_eia', 'ownership', 'appro_record_id_eia']],
                  sorter=['owned', 'total'])
        )
        return self.plant_parts_df

    def _test_prep_merge(self, part_name):
        """Run the test groupby and merge with the aggregations."""
        id_cols = self.plant_parts[part_name]['id_cols']
        plant_cap = (self.prep_plant_gen_df().groupby(
            by=id_cols + ['report_date', 'utility_id_eia', 'ownership'])
            .agg(self.plant_parts[part_name]['ag_cols'])
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
            test_merge[f'{test_col}_test'] == test_merge[f'{test_col}']
        )
        result = list(test_merge[f'test_{test_col}'].unique())
        logger.info(f'  Results for {test_col}: {result}')
        return test_merge

    def test_ag_cols(self, part_name):
        """
        For a compiled plant-part df, re-run groubys and check similarity.

        Args:
            part_name (string)
        """
        test_merge = self._test_prep_merge(part_name)
        for test_col in self.plant_parts[part_name]['ag_cols'].keys():
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
            self.test_ag_cols(part_name)


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
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result.to_frame(name=data_col).reset_index()


def get_eia_ferc_acct_map(file_name='depreciation_rmi.xlsx'):
    """
    Get map of EIA technology_description/pm codes <> ferc accounts.

    We must refactor this with a better path dependency. Or just store this
    map as a dataframe or dictionary.
    """
    file_path = pathlib.Path.cwd().parent / file_name
    eia_ferc_acct_map = (
        pd.read_excel(file_path, skiprows=0, sheet_name=3)
        [['technology_description', 'prime_mover_code', 'ferc_acct_name']]
        .drop_duplicates()
    )
    return eia_ferc_acct_map


def get_master_unit_list_eia(file_path_mul, clobber=False):
    """
    Get the master unit list; generate it or get if from a file.

    Args:
        file_path_mul (pathlib.Path)
        clobber (boolean): True if you want to regenerate the master unit list
            whether or not it is saved at the file_path_mul
    """
    if not file_path_mul.is_file() or clobber:
        logger.info(
            f"Master unit list not found {file_path_mul}"
            "Generating a new master unit list. This should take ~10 minutes."
        )

        table_compiler = CompileTables(
            pudl_engine=sa.create_engine(
                pudl.workspace.setup.get_defaults()["pudl_db"]),
            freq='AS', rolling=True)
        parts_compilers = CompilePlantParts(table_compiler)
        plant_parts_df = parts_compilers.generate_master_unit_list()
        plant_parts_df.to_csv(file_path_mul, compression='gzip')

    elif file_path_mul.is_file():
        logger.info(f"Reading the master unit list from {file_path_mul}")
        plant_parts_df = pd.read_pickle(file_path_mul, compression='gzip')
    return plant_parts_df
