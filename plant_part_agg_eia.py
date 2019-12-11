"""Aggregate plant parts."""


import logging

# import numpy as np
import pandas as pd
import sqlalchemy as sa

import pudl
import pudl.constants as pc

logger = logging.getLogger(__name__)


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
            pudl_engine=pudl_engine, freq=self.freq, rolling=rolling)

        self._dfs = {
            'generation_fuel_eia923': None,
            'fuel_receipts_costs_eia923': None,
            'generators_eia860': None,
            'boiler_generator_assn_eia860': None,
            'generation_eia923': None,
            'ownership_eia860': None,
            'generators_entity_eia': None,

            'fuel_cost': None,
            'mcoe': None,
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
                print(f'grabbing {table} from the sqlite db')
                select = sa.sql.select([tbl, ])

                if table == 'generators_entity_eia':
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

                # bga table has no sumable data cols and is reported annually
                if self.freq is not None and freq_ag_cols[table] is not None:
                    df = self.agg_cols(id_cols=freq_ag_cols[table]['id_cols'],
                                       ag_cols=freq_ag_cols[table]['ag_cols'],
                                       wtavg_cols=None,
                                       df_in=df)

            # if is it not a database table, it is an output function
            # elif hasattr(pudl_out_eia, table):
            except KeyError:
                # getattr turns the string of the table into an attribute
                # of the object, so this runs the output function
                print(f'grabbing {table} from the output object')
                df = getattr(self.pudl_out, table)()
            self._dfs[table] = pudl.helpers.convert_cols_dtypes(df,
                                                                'eia',
                                                                name=table)
        return self._dfs[table]

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
        table_df = self.grab_the_table(table)
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
        table_df = table_df.merge(self.grab_the_table(denorm_table)
                                  [list(set(id_cols + denorm_cols))]
                                  .drop_duplicates(),
                                  on=denorm_cols,
                                  how='outer',
                                  indicator=indicator)
        return(table_df)

    def grab_ownership(self):
        """Grab the ownership table and create total rows."""
        # grab the ownership table and do some munging
        own860 = (self.grab_the_table('ownership_eia860')
                  [['plant_id_eia', 'generator_id', 'report_date',
                    'utility_id_eia', 'fraction_owned',
                    'owner_utility_id_eia']])
        # make new records for generators to replicate the total generator
        own860_fake_totals = own860[own860['fraction_owned'] != 1][[
            'plant_id_eia', 'generator_id', 'report_date', 'utility_id_eia',
            'owner_utility_id_eia']].drop_duplicates()
        # asign 1 to all of the fraction_owned column
        # we'll be able to tell later if it is a total by the fraction owned
        own860_fake_totals['fraction_owned'] = 1
        # squish that back into the ownership table
        own860 = own860.append(own860_fake_totals, sort=True)

        # generate records in which we assume the owner is the operator
        # fake_operator_own860 = own860[['plant_id_eia',
        #                               'report_date',
        #                               'utility_id_eia',
        #                               'generator_id']].drop_duplicates()
        # fake_operator_own860['fraction_owned'] = 1.0
        # fake_operator_own860['owner_utility_id_eia'] = \
        #    fake_operator_own860['utility_id_eia']
        # squish that back into the ownership table
        # and drop dupes bc we may have added some dupes in this round
        # own860 = own860.append(fake_operator_own860,
        #                       sort=True).drop_duplicates()

        return own860

    def aggregate_plant_part(self, plant_part):
        """Generate dataframe of aggregated plant part."""
        cols_to_grab = plant_part['id_cols'] + ['report_date']
        plant_part_df = pd.DataFrame(columns=cols_to_grab)
        for table_name, table_details in plant_part['ag_tables'].items():
            # grab the table
            logger.info(f'beginning the aggregation for {table_name}')

            # grab the table
            table = self.grab_denormalize_table(
                table_name,
                denorm_table=table_details['denorm_table'],
                denorm_cols=table_details['denorm_cols'],
                id_cols=plant_part['id_cols'],
            )

            plant_part_df = self.agg_cols(
                id_cols=plant_part['id_cols'],
                ag_cols=table_details['ag_cols'],
                wtavg_cols=table_details['wtavg_cols'],
                df_in=table).merge(plant_part_df,
                                   how='outer')
        return plant_part_df

    def slice_by_ownership(self, plant_gen_df):
        """Generate proportional data by ownership %s."""
        own860 = self.grab_ownership()
        plant_gen_df = plant_gen_df.merge(
            own860,
            how='outer',
            on=['plant_id_eia', 'generator_id',
                'report_date', 'utility_id_eia'],
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

        # assign 100% ownership for records not in the ownership table
        plant_gen_df['fraction_owned'] = plant_gen_df['fraction_owned'].fillna(
            value=1)
        # assign the operator id as the owner if null
        plant_gen_df['owner_utility_id_eia'] = \
            plant_gen_df['owner_utility_id_eia'].fillna(
            plant_gen_df['utility_id_eia'])

        plant_gen_df = (
            plant_gen_df.drop(columns=['_merge', 'utility_id_eia']).
            rename(columns={'owner_utility_id_eia': 'utility_id_eia'}).
            drop_duplicates())

        cols_to_cast = ['net_generation_mwh', 'capacity_mw', 'total_fuel_cost']
        plant_gen_df[cols_to_cast] = (plant_gen_df[cols_to_cast].
                                      multiply(plant_gen_df['fraction_owned'],
                                               axis='index'))
        plant_gen_df.drop_duplicates()
        if (len(plant_gen_df[plant_gen_df['fraction_owned'] == 1].
                drop_duplicates()) !=
            len(plant_gen_df.drop_duplicates(
                subset=['plant_id_eia', 'generator_id',
                        'report_date', 'utility_id_eia']))):
            raise AssertionError('something')
        return plant_gen_df

    def agg_cols(self, id_cols, ag_cols, wtavg_cols, df_in):
        """Aggregate dataframe."""
        cols_to_grab = id_cols + ['utility_id_eia', 'fraction_owned']
        cols_to_grab = list(set(
            [x for x in cols_to_grab if x in list(df_in.columns)]))
        if 'report_date' in list(df_in.columns):
            if len(df_in[df_in['report_date'].dt.month > 2]) > 0:
                cols_to_grab = cols_to_grab + [pd.Grouper(freq=self.freq)]
                df_in = df_in.set_index(pd.DatetimeIndex(df_in.report_date))
            else:
                cols_to_grab = cols_to_grab + ['report_date']
        logger.info('aggregate the parts')
        logger.debug(f'     grouping by on {cols_to_grab}')
        logger.debug(f'     agg-ing on by on {ag_cols}')
        logger.debug(f'     cols in df are {df_in.columns}')
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

    def generate_master_unit_list(self, plant_parts):
        """
        Aggreate and slice data points by each plant part.

        1) aggregate the the data points by generator
        2) generating proportional data by ownership %s
        3) aggreate everything by each plant part

        Args:
            plant_parts (dict): a dictionary of information required to
                aggregate each plant part.

        """
        # 1) aggregate the data points by generator
        plant_gen_df = (self.aggregate_plant_part(plant_parts['plant_gen']).
                        astype({'utility_id_eia': 'Int64'}))
        # 2) generating proportional data by ownership %s
        plant_gen_df = (self.slice_by_ownership(plant_gen_df).
                        astype({'utility_id_eia': 'Int64'}))
        # 3) aggreate everything by each plant part
        compiled_dfs = {}
        for part_name, plant_part in plant_parts.items():
            logger.info(part_name)
            id_cols = plant_part['id_cols']
            ag_cols = plant_part['ag_cols']
            wtavg_cols = plant_part['wtavg_cols']

            if plant_part['denorm_table']:
                logger.info(f'denormalize {part_name}')
                compiled_dfs[part_name] = self.agg_cols(
                    id_cols=id_cols,
                    ag_cols=ag_cols,
                    wtavg_cols=wtavg_cols,
                    df_in=self.denoramlize_table(plant_gen_df,
                                                 id_cols,
                                                 plant_part['denorm_table'],
                                                 plant_part['denorm_cols'],
                                                 ))
            else:
                compiled_dfs[part_name] = self.agg_cols(
                    id_cols=id_cols,
                    ag_cols=ag_cols,
                    wtavg_cols=wtavg_cols,
                    df_in=plant_gen_df)
        return compiled_dfs


freq_ag_cols = {
    'generation_eia923': {
        'id_cols': ['plant_id_eia', 'generator_id'],
        'ag_cols': {'net_generation_mwh': 'sum', },
        'wtavg_cols': None
    },
    'generation_fuel_eia923': ['plant_id_eia', 'nuclear_unit_id',
                               'fuel_type', 'fuel_type_code_pudl',
                               'fuel_type_code_aer', 'prime_mover_code'],
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
}


def weighted_average(df, data_col, weight_col, by_col):
    """Generate a weighted average."""
    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result.to_frame(name=data_col).reset_index()
