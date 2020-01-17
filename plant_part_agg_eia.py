"""Aggregate plant parts."""


import logging
from copy import deepcopy

import numpy as np
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
            'utilities_eia': None,
            'plants_eia': None,
            'energy_source_eia923': None,

            'fuel_cost': None,
            'mcoe': None,
            'plants_steam_ferc1': None,
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
                logger.info(f'grabbing {table} from the output object')
                df = getattr(self.pudl_out, table)()
            self._dfs[table] = pudl.helpers.convert_cols_dtypes(df,
                                                                'eia',
                                                                name=table)
        return self._dfs[table]

    def agg_cols(self, id_cols, ag_cols, wtavg_cols, df_in):
        """Aggregate dataframe."""
        cols_to_grab = id_cols + \
            ['utility_id_eia', 'fraction_owned', 'ownership']
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


class CompilePlantParts(object):
    """Compile plant parts."""

    def __init__(self, table_compiler, clobber=False):
        """idk."""
        self.table_compiler = table_compiler
        self.freq = table_compiler.freq
        self.plant_gen_df = None
        self.plant_parts_df = None
        self.clobber = clobber

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
        # make new records for generators to replicate the total generator
        own860_fake_totals = own860[own860['fraction_owned'] != 1][[
            'plant_id_eia', 'generator_id', 'report_date', 'utility_id_eia',
            'owner_utility_id_eia']].drop_duplicates()
        # asign 1 to all of the fraction_owned column
        # we'll be able to tell later if it is a total by the fraction owned
        own860_fake_totals['fraction_owned'] = 1
        # squish that back into the ownership table
        own860 = own860.append(own860_fake_totals, sort=True)
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

            plant_part_df = self.table_compiler.agg_cols(
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
            drop_duplicates().
            assign(ownership=plant_gen_df['fraction_owned'].
                   apply(lambda x: 'total' if x == 1 else 'owned')))

        cols_to_cast = ['net_generation_mwh', 'capacity_mw', 'total_fuel_cost']
        plant_gen_df[cols_to_cast] = (plant_gen_df[cols_to_cast].
                                      multiply(plant_gen_df['fraction_owned'],
                                               axis='index'))
        if (len(plant_gen_df[plant_gen_df['fraction_owned'] == 1].
                drop_duplicates()) !=
            len(plant_gen_df.drop_duplicates(
                subset=['plant_id_eia', 'generator_id',
                        'report_date', 'utility_id_eia']))):
            raise AssertionError('something')
        return plant_gen_df

    def _find_false_grans(self, part_df, part_peer_df,
                          id_cols, id_cols_peer, peer_part):
        # logger.debug(f'part_df cols: {list(part_df.columns)}')
        # logger.debug(f'part_peer_df cols: {list(part_peer_df.columns)}')
        logger.debug(
            f'id cols:{id_cols} & peer id cols {id_cols_peer}')
        if 'count' in part_df.columns:
            part_df = part_df.drop(columns=['count'])

        return (
            part_peer_df.drop_duplicates(
                subset=set(id_cols_peer + id_cols)).
            pipe(pudl.helpers.count_records, id_cols, 'count').
            merge(part_df, how='right').
            # here we are assigning True
            assign(false_gran=lambda x: x.apply(
                lambda x: True if x['count'] == 1
                else x['false_gran'], axis=1),
                peer_part=lambda x: x.apply(
                    lambda x: peer_part if x['count'] == 1 else x['peer_part'],
                    axis=1)).
            drop(columns=['count'])
        )

    def prep_peer_part(self, part_df, plant_parts, peer_part, part_name,
                       id_cols, id_cols_peer):
        """
        Prepare a dataframe with info on the peer plant part.

        Args:
            part_df (pandas.DataFrame): dataframe containing the compiled/
                aggregated information for this plant part.
            plant_parts (dict): a dictionary of information required to
                aggregate each plant part.
            peer_part (string): name of the peer plant part
            part_name (string): name of the plant part
            id_cols (list): list of id columns (from plant_parts) for plant
                part
            id_cols_peer (list): list of id columns (from plant_parts) for peer
                plant part
        Returns:
            pandas.DataFrame

        """
        # prepare a df for counting
        if plant_parts[peer_part]['denorm_table']:
            logger.debug('grabbing denorm_table')
            part_peer_df = self.table_compiler.grab_the_table(
                plant_parts[peer_part]['denorm_table'])
        else:
            part_peer_df = part_df
        if (not set(id_cols_peer).issubset(list(part_peer_df.columns))
                and peer_part == 'plant_gen'):
            logger.debug('using the plant_gen_df for the part_peer_df')
            part_peer_df = self.plant_gen_df
        if not set(id_cols).issubset(list(part_peer_df.columns)):
            logger.debug('merging the id_cols into the peer table')
            part_peer_df = part_peer_df.merge(
                self.table_compiler.grab_the_table(
                    plant_parts[part_name]['denorm_table'])
                [id_cols + ['generator_id']].drop_duplicates())
        return part_peer_df

    def remove_false_gran(self, part_df, id_cols, plant_parts, part_name):
        """
        Remove the plant parts that are false granularities.

        We are removing all instances of records where there is only one
        type of a record for each plant. We could later determine that we want
        to remove these granularities based on each part's true parent (unit is
        the parent of a generator), but for now this generalization is
        sufficent in removing false granularities.
        """
        false_gran = plant_parts[part_name]['false_grans']
        if false_gran:
            if 'false_gran' not in part_df.columns:
                part_df['false_gran'] = np.nan
            if 'peer_part' in part_df.columns:
                part_df['peer_part'] = np.nan
            for peer_part in false_gran:
                logger.debug(f'labeling false granularities from {peer_part}')
                id_cols_peer = plant_parts[peer_part]['id_cols']
                part_peer_df = self.prep_peer_part(
                    part_df, plant_parts, peer_part, part_name,
                    id_cols, id_cols_peer)

                part_df = self._find_false_grans(
                    part_df, part_peer_df, id_cols, id_cols_peer, peer_part)
        return part_df

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

    def add_record_id(self, part_df, id_cols):
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
            part_df.plant_part + "_" +
            part_df.ownership + "_" +
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

    def grab_consistent_qualifiers(self,
                                   part_df,
                                   record_name,
                                   id_cols,
                                   denorm_table,
                                   denorm_cols
                                   ):
        """
        Grab fully consistent qualifier records.

        For an individual compiled dataframe for each of the plant parts, we
        need to

        When qualitative data is consistent for every record in a plant part,
        we assign these catagoricals. If the records are not consistent, then
        nothing is added.

        Args:
            part_df (pandas.DataFrame)
            record_name (string) : name of qualitative record
            id_cols (list) : list of identifying columns.
            denorm_table (string) : name of table needed to denormalize
            denorm_cols (list)

        """
        if record_name in part_df.columns:
            logger.debug(f'{record_name} already here.. ')
            return part_df

        record_df = self.table_compiler.grab_the_table(
            qual_record_tables[record_name])

        if denorm_table and denorm_table != qual_record_tables[record_name]:
            if 'report_date' not in record_df.columns:
                record_df = (
                    record_df.merge(
                        self.table_compiler.grab_the_table('generators_eia860')
                        [list(set(denorm_cols + ['report_date']))],
                        how='left'))

            record_df = self.denoramlize_table(
                record_df,
                id_cols,
                denorm_table,
                denorm_cols,
            )

        if 'report_date' in record_df.columns:
            base_cols = id_cols + ['report_date']
        else:
            base_cols = id_cols
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

        logger.debug(f'merging in consistent {record_name}')
        return part_df.merge(consistent_records, how='left')

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
        if self.plant_parts_df is not None and self.clobber is False:
            return self.plant_parts_df
        if self.plant_gen_df is None:
            # 1) aggregate the data points by generator
            self.plant_gen_df = (
                self.aggregate_plant_part(plant_parts['plant_gen']).
                astype({'utility_id_eia': 'Int64'}).
                # 2) generating proportional data by ownership %s
                pipe(self.slice_by_ownership).
                astype({'utility_id_eia': 'Int64'}))

        # 3) aggreate everything by each plant part
        plant_parts_df = pd.DataFrame()
        plant_parts_ordered = ['plant_prime_fuel', 'plant_technology',
                               'plant_prime_mover', 'plant_gen', 'plant_unit',
                               'plant']
        for part_name in plant_parts_ordered:
            plant_part = plant_parts[part_name]
            logger.info(f'begin aggregation for: {part_name}')
            id_cols = plant_part['id_cols']
            ag_cols = plant_part['ag_cols']
            wtavg_cols = plant_part['wtavg_cols']

            if plant_part['denorm_table']:
                logger.info(f'denormalize {part_name}')
                df_in = self.denoramlize_table(self.plant_gen_df,
                                               id_cols,
                                               plant_part['denorm_table'],
                                               plant_part['denorm_cols'],
                                               )
            else:
                df_in = self.plant_gen_df
            thing = (
                self.table_compiler.agg_cols(
                    id_cols=id_cols,
                    ag_cols=ag_cols,
                    wtavg_cols=wtavg_cols,
                    df_in=df_in).
                pipe(self.add_install_year, id_cols,
                     plant_part['install_table']).
                assign(plant_part=part_name).
                pipe(self.remove_false_gran, id_cols=id_cols,
                     plant_parts=plant_parts, part_name=part_name).
                pipe(self.add_record_id, id_cols))

            for qual_record in qual_record_tables:
                logger.debug(f'grab consistent {qual_record} for {part_name}')
                thing = self.grab_consistent_qualifiers(
                    thing,
                    qual_record,
                    id_cols,
                    plant_part['denorm_table'],
                    plant_part['denorm_cols'])
            plant_parts_df = plant_parts_df.append(thing, sort=True)

        plant_parts_df = (self.add_additonal_cols(plant_parts_df).
                          pipe(pudl.helpers.organize_cols,
                               ['plant_id_eia',
                                'report_date',
                                'plant_part',
                                'generator_id',
                                'unit_id_pudl',
                                'prime_mover_code',
                                'energy_source_code_1',
                                'technology_description',
                                'utility_id_eia',
                                'fraction_owned'
                                ]).
                          pipe(self._clean_plant_parts))
        self.plant_parts_df = plant_parts_df
        return plant_parts_df


freq_ag_cols = {
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

qual_record_tables = {
    'energy_source_code_1': 'generators_eia860',
    'prime_mover_code': 'generators_entity_eia',
    'fuel_type_code_pudl': 'generators_eia860',
}


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
