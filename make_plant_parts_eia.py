"""Aggregate plant parts."""


import logging
import pathlib
from copy import deepcopy

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


class CompileTables(object):
    """Compile tables from sqlite db or pudl output object."""

    def __init__(self, pudl_engine, freq=None, start_date=None, end_date=None,
                 rolling=False):
        """
        Initialize a table compiler.

        grabs the stuff....
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

        # grabing the metadata object for the sqlite db
        self.pt = pudl.output.pudltabl.get_table_meta(self.pudl_engine)

        self.pudl_out = pudl.output.pudltabl.PudlTabl(
            pudl_engine=pudl_engine, freq=self.freq, roll=rolling)

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

    def grab_the_table(self, table):
        """Get a dataframe from the sqlite tables or the output object."""
        if table is None:
            return

        if self._dfs[table] is None:
            # this is going to try to see if the table is in the db
            # if pt[table] is not None:
            try:
                tbl = self.pt[table]
                logger.info(f'grabbing {table} from the sqlite db')
                select = sa.sql.select([tbl, ])
                # for the non-date tables...
                if 'report_date' not in tbl.columns:
                    logger.debug('grabbing a non-date table')
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
                logger.info(f'grabbing {table} from the output object')
                # the pudl_out.mcoe function has defaults set for data purity.
                # these defaults were removing ~20% of the fuel cost records.
                # mcoe is the only pudl_out function that is used in the MUL -
                # this if/else enables grabbing other pudl_out tables w/o args
                if table == 'mcoe':
                    df = self.pudl_out.mcoe(min_heat_rate=None,
                                            min_fuel_cost_per_mwh=None,
                                            min_cap_fact=None,
                                            max_cap_fact=None)
                elif table == 'ferc_acct_rmi':
                    df = grab_eia_ferc_acct_map()
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
        self.plant_parts_df = None
        self.clobber = clobber
        self.plant_parts_ordered = ['plant', 'plant_unit',
                                    'plant_prime_mover', 'plant_technology',
                                    'plant_prime_fuel', 'plant_gen',
                                    'plant_ferc_acct']
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

    def grab_denormalize_table(self,
                               table,
                               denorm_table=None,
                               denorm_cols=None,
                               id_cols=None,
                               indicator=False):
        """
        Grab and denormalize the table.

        Grab the table that you want, and merge it with another table based
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
        table_df = self.table_compiler.grab_the_table(table)
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
        """Merge data table with additional table to grab additional colums."""
        # denormalize the plant granularity
        table_df = table_df.merge(
            self.table_compiler.grab_the_table(denorm_table)
            [list(set(id_cols + denorm_cols))].drop_duplicates(),
            on=denorm_cols,
            how='outer',
            indicator=indicator)
        return(table_df)

    def grab_ownership(self):
        """Grab the ownership table and create total rows."""
        # grab the ownership table and do some munging
        own860 = (self.table_compiler.grab_the_table('ownership_eia860')
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
            # grab the table
            table = self.grab_denormalize_table(
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
        own860 = self.grab_ownership()
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

    def denorm_plant_gen(self):
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
        # grab the demorn tables and squish the id cols in with the gens
        for k, v in denorm_tables.items():
            if not set(v['id_cols']).issubset(set(self.plant_gen_df.columns)):
                logger.debug(f'no id cols from {k}')
                denorm_df = self.table_compiler.grab_the_table(
                    k)[list(set(v['id_cols']
                                + v['denorm_cols']))].drop_duplicates()
                self.plant_gen_df = self.plant_gen_df.merge(
                    denorm_df, how='left')
                if og_len != len(self.plant_gen_df):
                    raise AssertionError(
                        'ahh merge error! when adding denorm colunms to'
                        'plant_gen_df we must get the same number of records')
        if 'plant_ferc_acct' in self.plant_parts.keys():
            self.plant_gen_df = self.add_ferc_acct(self.plant_gen_df)
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
        part_own = self.plant_gen_df[self.plant_gen_df.ownership == 'owned']
        part_tot = self.plant_gen_df[self.plant_gen_df.ownership == 'total']
        if len(self.plant_gen_df) != len(part_own) + len(part_tot):
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
        part_tot = (self.table_compiler.agg_cols(
            id_cols=id_cols,
            ag_cols=ag_cols,
            wtavg_cols=wtavg_cols,
            df_in=part_tot).
            merge(self.plant_gen_df[id_cols +
                                    ['report_date', 'utility_id_eia']].
                  dropna().
                  drop_duplicates()).
            assign(ownership='total')
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
            merge(self.table_compiler.grab_the_table('utilities_eia'),
                  how='left').
            merge(self.table_compiler.grab_the_table('plants_eia'), how='left')
        )
        return plant_parts_df

    def add_record_id(self, part_df, id_cols, plant_part_col='plant_part'):
        """Add a record id to a compiled part df."""
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
        gen_ent = self.table_compiler.grab_the_table('generators_entity_eia')
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
                            merge(self.plant_gen_df)
                            [id_cols + ['installation_year']].
                            drop_duplicates(subset=id_cols, keep='first'))
        else:
            part_install = (install[['plant_id_eia', 'generator_id',
                                     'installation_year']].
                            merge(self.table_compiler.grab_the_table(
                                install_table))
                            [id_cols + ['installation_year']].
                            drop_duplicates(subset=id_cols, keep='first'))
        part_df = part_df.merge(part_install, how='left')
        logger.debug(f'count of install years for part: {len(part_install)}')
        logger.debug(f'post count of part DataFrame: {len(part_df)}')
        return part_df

    def grab_consistent_qualifiers(self, record_df, base_cols, record_name):
        """
        Grab fully consistent qualifier records.

        When qualitative data is consistent for every record in a plant part,
        we grab these catagoricals. If the records are not consistent, then
        nothing is compiled.

        Args:
            record_df (pandas.DataFrame): the dataframe with the record
            base_cols (list) : list of identifying columns.
            record_name (string) : name of qualitative record
        """
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

    def grab_max_op_status(self, record_df, base_cols, record_name, sorter):
        """
        Grab the max operating status.

        We want to find the most relevant record  as defined by the sorter. In
        the example case of the operating status means that if any related
        generator is operable, than we'll label the whole plant as operable.

        Args:
            record_df (pandas.DataFrame): the dataframe with the record
            base_cols (list) : list of identifying columns.
            record_name (string) : name of qualitative record
            sorter (list): sorted list of category options
        """
        record_df[record_name] = record_df[record_name].astype("category")
        record_df[record_name].cat.set_categories(sorter, inplace=True)
        record_df = record_df.sort_values(record_name)
        return (record_df[base_cols + [record_name]]
                .drop_duplicates(keep='first'))

    def dedup_on_category(self, dedup_df, base_cols, category_name, sorter):
        """
        Drop duplicates based on category.

        Args:
            dedup_df (pandas.DataFrame): the dataframe with the record
            base_cols (list) : list of identifying columns.
            category_name (string) : name of qualitative record
            sorter (list): sorted list of category options
        """
        dedup_df[category_name] = dedup_df[category_name].astype("category")
        dedup_df[category_name].cat.set_categories(sorter, inplace=True)
        dedup_df = dedup_df.sort_values(category_name)
        return dedup_df.drop_duplicates(subset=base_cols, keep='first')

    def grab_qualifiers(self,
                        part_df,
                        record_name,
                        id_cols,
                        denorm_table,
                        denorm_cols
                        ):
        """
        Grab qualifier records.

        For an individual compiled dataframe for each of the plant parts, we
        want to add back in some non-data columns (qualifier columns). When
        qualitative data is consistent for every record in a plant part, we
        assign these catagoricals. If the records are not consistent, then
        nothing is added.

        Args:
            part_df (pandas.DataFrame)
            record_name (string) : name of qualitative record
            id_cols (list) : list of identifying columns.
            denorm_table (string) : name of table needed to denormalize
            denorm_cols (list): list of columns to denormalize on

        """
        if record_name in part_df.columns:
            logger.info(f'{record_name} already here.. ')
            return part_df

        record_df = pd.merge(
            self.plant_gen_df,
            self.table_compiler.grab_the_table(QUAL_RECORD_TABLES[record_name])
        )

        base_cols = id_cols
        if 'report_date' in record_df.columns:
            base_cols = base_cols + ['report_date']

        if record_name != 'operational_status':
            logger.debug(f'grabbing consistent {record_name}s')
            consistent_records = self.grab_consistent_qualifiers(
                record_df, base_cols, record_name)
        if record_name == 'operational_status':
            logger.debug(f'grabbing max {record_name}')
            sorter = ['existing', 'proposed', 'retired']
            # restric the number of columns in here to only include the ones we
            # need, unlike grab_consistent_qualifiers, dedup_on_category
            # preserves all of the columns from record_df
            record_df = record_df[
                [c for c in record_df.columns
                 if c in base_cols + list(part_df.columns) + [record_name]]]
            consistent_records = self.dedup_on_category(
                record_df, base_cols, record_name, sorter
            )
        non_nulls = consistent_records[consistent_records[record_name].notnull(
        )]
        logger.info(
            f'merging in consistent {record_name}: {len(non_nulls)}')
        return part_df.merge(consistent_records, how='left')

    def count_peer_parts(self, part_name):
        """Count the number of records from peer parts."""
        df = self.plant_gen_df[self.id_cols_dict[part_name]].drop_duplicates()
        part_cols = self.plant_parts[part_name]['id_cols'] + ['report_date']
        # create the count
        df_count = (df.dropna(subset=part_cols).
                    groupby(by=part_cols).nunique().
                    drop(columns=[self.plant_parts[part_name]['id_cols'][-1],
                                  'plant_id_eia',
                                  'report_date']).
                    rename(columns=self.ids_to_parts).
                    add_suffix(f'_count_{part_name}'))
        # merge back into the og df
        df_w_count = df.merge(
            df_count,
            how='left',
            right_index=True,
            left_on=part_cols,
        )
        return df_w_count

    def make_all_the_counts(self):
        """All of them."""
        all_the_counts = deepcopy(self.plant_gen_df)
        for part in self.plant_parts_ordered:
            all_the_counts = all_the_counts.merge(
                self.count_peer_parts(part), how='left')
        return all_the_counts

    def make_all_the_bools(self):
        """Count consistency of records and convert that to bools."""
        all_the_counts = self.make_all_the_counts()
        # remove the data columns... just for ease (maybe remove this later)
        counts = all_the_counts.drop(
            columns=['fuel_cost_per_mmbtu', 'heat_rate_mmbtu_mwh',
                     'fuel_cost_per_mwh', 'total_fuel_cost',
                     'total_mmbtu', 'net_generation_mwh', 'capacity_mw', ])

        # convert the count columns to bool columns
        for col in counts.filter(like='_count_').columns:
            counts.loc[counts[col].notnull(), col] = counts[col] > 1
            # counts = counts.astype({col: 'bool'})

        # TODO: turn on this astype when we convert over to pandas 1.0
        # counts = counts.filter(like='_count_').astype(pd.BooleanDtype())
        # assign a bool for the true gran only if all of the subparts are true
        for part in self.plant_parts.keys():
            counts[f'true_gran_{part}'] = (
                counts.filter(like=part + '_count_').
                all(axis='columns'))
            # counts = counts.astype({f'true_gran_{part}': pd.BooleanDtype()})

        return counts

    def assign_record_id_eia(self, test_df, plant_part_col='plant_part'):
        """
        Assign record ids to a df with a mix of plant parts.

        Args:
            test_df (pandas.DataFrame)
            plant_part_col (string)

        """
        test_df_ids = pd.DataFrame()
        for part in self.plant_parts.keys():
            test_df_ids = pd.concat(
                [test_df_ids,
                 self.add_record_id(test_df[test_df[plant_part_col] == part],
                                    self.plant_parts[part]['id_cols'],
                                    plant_part_col=plant_part_col
                                    )])
        return test_df_ids

    def label_true_id_by_part(self, part_name, bools):
        """
        Label the false granularities with their true parts.

        Args:
            part_name (string)
            bools (pandas.DataFrame)

        """
        # another way to do this would be to construct the list of colums based
        # on self.plant_parts_ordered
        cols = list(bools.filter(like=part_name + '_count_').columns)
        # reserve the columns because we want the biggest part to be the last
        # assigned
        assign_col = "appro_part_label_" + part_name
        bools[assign_col] = part_name
        cols.reverse()
        for col in cols:
            # TODO: go back to ~bools[col] when we convert over to pandas 1.0
            # bools.loc[~bools[col], assign_col] = col.split(sep="_count_")[1]
            bools.loc[bools[col] == False, assign_col] = \
                col.split(sep="_count_")[1]
        bools = (
            self.assign_record_id_eia(bools, assign_col).
            rename(columns={'record_id_eia': f'record_id_eia_{part_name}'}))
        return bools

    def assign_true_gran(self, part_df, part_name):
        """
        Merge the true granularity labels into the plant part df.

        Args:
            part_df (pandas.DataFrame)
            part_name (string)

        """
        bool_df = deepcopy(self.part_bools)
        # grab only the columns you need for this part and drop duplicates
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
        """Add plants names into the compiled plant part df."""
        part_df = (pd.merge(
            part_df, self.table_compiler.grab_the_table('plants_eia'))
            .assign(plant_name_new=lambda x: x.plant_name_eia))
        col = self.parts_to_ids[part_name]
        part_df.loc[part_df[col].notnull(), 'plant_name_new'] = (
            part_df['plant_name_new'] + " " + part_df[col].astype(str))
        return part_df

    def add_ferc_acct(self, plant_gen_df):
        """Merge the plant_gen_df with the ferc_acct map."""
        return pd.merge(plant_gen_df, grab_eia_ferc_acct_map(), how='left')

    def prep_plant_gen_df(self):
        """Prepare plant gen dataframe."""
        logger.info('Generating the master generator table with ownership.')
        self.plant_gen_df = (
            self.aggregate_plant_part(self.plant_parts['plant_gen'])
            .astype({'utility_id_eia': 'Int64'}).
            pipe(self.slice_by_ownership)
            .astype({'utility_id_eia': 'Int64'})
        )
        self.plant_gen_df = self.denorm_plant_gen()
        return self.plant_gen_df

    def prep_part_bools(self):
        """Prep the part_bools df that denotes true_gran for all generators."""
        self.part_bools = self.make_all_the_bools()
        for part_name1 in self.plant_parts.keys():
            self.part_bools = self.label_true_id_by_part(part_name1,
                                                         self.part_bools)
        return self.part_bools

    def generate_master_unit_list(self, qual_records):
        """
        Aggreate and slice data points by each plant part.

        1) aggregate the the data points by generator
        2) generating proportional data by ownership %s
        3) aggreate everything by each plant part

        Args:
            plant_parts (dict): a dictionary of information required to
                aggregate each plant part.
            relabel (bool): if True, the one owner plants will be labeled as
                "owned" in the ownership column.

        """
        if self.plant_parts_df is not None and self.clobber is False:
            return self.plant_parts_df
        if self.plant_gen_df is None:
            self.plant_gen_df = self.prep_plant_gen_df()
        if self.part_bools is None:
            self.part_bools = self.prep_part_bools()

        # 3) aggreate everything by each plant part
        plant_parts_df = pd.DataFrame()
        plant_parts_ordered = [
            'plant_ferc_acct', 'plant_prime_fuel', 'plant_technology',
            'plant_prime_mover', 'plant_gen', 'plant_unit', 'plant']
        # could use reversed(self.plant_parts_ordered)?
        for part_name in plant_parts_ordered:
            plant_part = self.plant_parts[part_name]
            id_cols = plant_part['id_cols']

            thing = (
                self.ag_part_by_own_slice(part_name)
                .pipe(self.add_install_year, id_cols,
                      plant_part['install_table'])
                .assign(plant_part=part_name)
                .pipe(self.assign_true_gran, part_name)
                .pipe(self.add_record_id, id_cols, plant_part_col='plant_part')
                .pipe(self.add_new_plant_name, part_name)
            )
            logger.info('plant_name_eia' in thing.columns)
            if qual_records:
                # add in the qualifier records
                for qual_record in QUAL_RECORD_TABLES:
                    logger.debug(
                        f'grab consistent {qual_record} for {part_name}')
                    thing = self.grab_qualifiers(
                        thing,
                        qual_record,
                        id_cols,
                        plant_part['denorm_table'],
                        plant_part['denorm_cols'])
            logger.info('plant_name_eia' in thing.columns)
            plant_parts_df = plant_parts_df.append(thing, sort=True)
        # clean up, add additional columns
        plant_parts_df = (
            self.add_additonal_cols(plant_parts_df)
            .pipe(pudl.helpers.organize_cols,
                  ['plant_id_eia', 'report_date', 'plant_part', 'generator_id',
                   'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
                   'technology_description', 'ferc_acct_name',
                   'utility_id_eia', 'true_gran', 'appro_part_label'])
            .pipe(self._clean_plant_parts)
        )

        self.plant_parts_full_df = plant_parts_df

        self.plant_parts_df = self.dedup_on_category(
            plant_parts_df,
            category_name='ownership',
            base_cols=[x for x in plant_parts_df.columns if x not in [
                'record_id_eia', 'ownership', 'appro_record_id_eia', ]],
            sorter=['owned', 'total']
        )
        return self.plant_parts_df

    def _test_prep_merge(self, part_name):
        """Run the test groupby and merge with the aggregations."""
        id_cols = self.plant_parts[part_name]['id_cols']
        plant_cap = (self.plant_gen_df.groupby(
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


def grab_eia_ferc_acct_map(file_name='depreciation_rmi.xlsx'):
    """
    Grab map of EIA technology_description/pm codes <> ferc accounts.

    We must refactor this with a better path dependency. Or just store this
    map as a dataframe or dictionary.
    """
    file_path = pathlib.Path.cwd().parent / file_name
    eia_ferc_acct_map = pd.read_excel(file_path, skiprows=0, sheet_name=3)[
        ['technology_description', 'prime_mover_code', 'ferc_acct_name']]
    return eia_ferc_acct_map


def get_master_unit_list_eia(file_path_mul, clobber=False):
    """
    Get the master unit list; generate it or grab if from a file.

    Args:
        file_path_mul (pathlib.Path)
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
        plant_parts_df = parts_compilers.generate_master_unit_list(
            qual_records=False)
        plant_parts_df.to_csv(file_path_mul, compression='gzip')

    elif file_path_mul.is_file():
        logger.info(f"Reading the master unit list from {file_path_mul}")
        plant_parts_df = pd.read_pickle(file_path_mul, compression='gzip')
    return plant_parts_df
