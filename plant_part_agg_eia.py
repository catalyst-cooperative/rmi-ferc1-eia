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
                    df = self.agg_cols(id_cols=freq_ag_cols[table]['id_cols'] +
                                       ['utility_id_eia', 'fraction_owned'],
                                       ag_cols=freq_ag_cols[table]['ag_cols'],
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

    def __init__(self, table_compiler, plant_parts, clobber=False):
        """
        idk.

        Args:
            plant_parts (dict): a dictionary of information required to
                aggregate each plant part.
            table_compiler (object)
            clobber (bool) : if True, you will clobber plant_parts_df (the
                master unit list)

        """
        self.table_compiler = table_compiler
        self.freq = table_compiler.freq
        self.plant_parts = plant_parts
        self.plant_gen_df = None
        self.plant_parts_df = None
        self.clobber = clobber
        self.plant_parts_ordered = ['plant', 'plant_unit',
                                    'plant_prime_mover', 'plant_technology',
                                    'plant_prime_fuel', 'plant_gen']
        self.gen_util_ids = ['plant_id_eia', 'generator_id',
                             'report_date', 'utility_id_eia']
        # make a dictionary with the main id column (key) corresponding to the
        # plant part (values)
        self.ids_to_parts = {}
        for part, part_dict in self.plant_parts.items():
            self.ids_to_parts[self.plant_parts[part]['id_cols'][-1]] = part

        self.id_cols_dict = self.make_id_cols_dict()

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
        # own860_fake_totals = own860[own860['fraction_owned'] != 1][[
        #    'plant_id_eia', 'generator_id', 'report_date', 'utility_id_eia',
        #    'owner_utility_id_eia']].drop_duplicates()
        # asign 1 to all of the fraction_owned column
        # we'll be able to tell later if it is a total by the fraction owned
        #own860_fake_totals['fraction_owned'] = 1
        # squish that back into the ownership table
        #own860 = own860.append(own860_fake_totals, sort=True)
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

    def _relabel_sole_owner_gens(self, df, record_ids_cols):
        """
        Relabel the sole owner generators to be owned, not toal.

        This methods' assertions will fail if this is run after one round of
        re-labeling has been preformed. This is meant to be run at the end of
        self.slice_by_ownership and nowhere else.

        Args:
            plant_gen_df (pandas.DataFrame)
            record_ids_cols (list): list of id columns for generator/utility
                records.
        """
        owned_og = len(df[df.ownership == 'owned'])
        plant_gen_count = pd.DataFrame(df.groupby(record_ids_cols).size(),
                                       columns=['count']).reset_index()
        df = pd.merge(df, plant_gen_count, on=record_ids_cols)

        # do some checks
        if len(df[(df['count'] == 1) & (df['ownership'] != 'total')]) != 0:
            raise AssertionError(
                'All 1-record generator should be labeled total at this stage.'
                'Check calc in `plant_gen_count`'
                'OR whether re-labeling has already happened.'
            )
        if len(df[(df['ownership'] == 'total')]) < len(df[(df['count'] == 1)]):
            raise AssertionError(
                'There should be more total generators than 1-count generators'
                'Check calc in `plant_gen_count`.'
            )

        df.loc[(df['count'] == 1), 'ownership'] = 'owned'
        df = df.drop(columns=['count'])
        owned_post = len(df[df.ownership == 'owned'])
        logger.info(f'OG: {owned_og} / Post: {owned_post}')
        if not owned_post >= owned_og:
            raise AssertionError(
                'Owned-labeld plants should increase during relabeling.'
            )
        return df

    def slice_by_ownership(self, plant_gen_df, relabel):
        """Generate proportional data by ownership %s."""
        own860 = self.grab_ownership()
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
                len(plant_gen_df.drop_duplicates(subset=self.gen_util_ids))):
            raise AssertionError('something')
        if relabel:
            owned_og = len(plant_gen_df[plant_gen_df.ownership == 'owned'])
            dtypes_og = plant_gen_df.dtypes
            plant_gen_df = self._relabel_sole_owner_gens(
                plant_gen_df, self.gen_util_ids)
            owned_post = len(plant_gen_df[plant_gen_df.ownership == 'owned'])
            dtypes_post = plant_gen_df.dtypes
            logger.info(f'OG: {owned_og} / Post: {owned_post}')
            logger.info(f'OG:   {dtypes_og}'
                        f'Post: {dtypes_post}')
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
        # part_own = self.plant_gen_df[self.plant_gen_df.ownership == 'owned']
        # part_tot = self.plant_gen_df[self.plant_gen_df.ownership == 'total']
        part_own = self.plant_gen_df[self.plant_gen_df['fraction_owned'] < 1]
        part_tot = self.plant_gen_df[self.plant_gen_df['fraction_owned'] == 1]
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

        if record_name != 'operational_status':
            logger.debug(f'grabbing consistent {record_name}s')
            consistent_records = self.grab_consistent_qualifiers(
                record_df, base_cols, record_name)
        else:
            logger.debug(f'grabbing max {record_name}')
            sorter = ['existing', 'proposed', 'retired']
            consistent_records = self.grab_max_op_status(
                record_df, base_cols, record_name, sorter)
        logger.debug(f'merging in consistent {record_name}')
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

    def add_new_plant_name(self):
        """Add plants names into the compiled plant part df."""
        df = self.plant_parts_df
        id_cols_all = ['generator_id', 'unit_id_pudl', 'prime_mover_code',
                       'energy_source_code_1', 'technology_description']
        df['plant_name_new'] = df['plant_name_eia']
        col = 'generator_id'
        for col in id_cols_all:
            df.loc[df[col].notnull(), 'plant_name_new'] = (
                df['plant_name_new'] + " " + df[col].astype(str))
        self.plant_parts_df = df
        return self.plant_parts_df

    def prep_plant_gen_df(self, relabel):
        """Prepare plant gen dataframe."""
        # 1) aggregate the data points by generator
        self.plant_gen_df = (
            self.aggregate_plant_part(self.plant_parts['plant_gen']).
            astype({'utility_id_eia': 'Int64'}).
            # 2) generating proportional data by ownership %s
            pipe(self.slice_by_ownership, relabel).
            astype({'utility_id_eia': 'Int64'}))
        self.plant_gen_df = self.denorm_plant_gen()
        return self.plant_gen_df

    def prep_part_bools(self):
        """Prep the part_bools df that denotes true_gran for all generators."""
        self.part_bools = self.make_all_the_bools()
        for part_name1 in self.plant_parts.keys():
            self.part_bools = self.label_true_id_by_part(part_name1,
                                                         self.part_bools)
        return self.part_bools

    def generate_master_unit_list(self, relabel=True):
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
            self.plant_gen_df = self.prep_plant_gen_df(relabel)
        self.part_bools = self.prep_part_bools()

        # 3) aggreate everything by each plant part
        plant_parts_df = pd.DataFrame()
        plant_parts_ordered = ['plant_prime_fuel', 'plant_technology',
                               'plant_prime_mover', 'plant_gen', 'plant_unit',
                               'plant']
        for part_name in plant_parts_ordered:
            plant_part = self.plant_parts[part_name]
            id_cols = plant_part['id_cols']

            thing = (
                self.ag_part_by_own_slice(part_name).
                pipe(self.add_install_year, id_cols,
                     plant_part['install_table']).
                assign(plant_part=part_name).
                pipe(self.assign_true_gran, part_name).
                # pipe(self._relabel_sole_owner_gens,
                #     id_cols + ['report_date', 'utility_id_eia']).
                pipe(self.add_record_id, id_cols, plant_part_col='plant_part')
            )
            # add in the qualifier records
            for qual_record in qual_record_tables:
                logger.debug(f'grab consistent {qual_record} for {part_name}')
                thing = self.grab_qualifiers(
                    thing,
                    qual_record,
                    id_cols,
                    plant_part['denorm_table'],
                    plant_part['denorm_cols'])
            plant_parts_df = plant_parts_df.append(thing, sort=True)
        # clean up, add additional columns
        plant_parts_df = (
            self.add_additonal_cols(plant_parts_df).
            pipe(pudl.helpers.organize_cols,
                 ['plant_id_eia', 'report_date', 'plant_part', 'generator_id',
                  'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
                  'technology_description', 'utility_id_eia', 'true_gran',
                  'appro_part_label']).
            pipe(self._clean_plant_parts))

        self.plant_parts_df = plant_parts_df
        self.plant_parts_df = self.add_new_plant_name()
        return self.plant_parts_df


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
    'fuel_type_code_pudl': 'generators_eia860',
    'operational_status': 'generators_eia860',
    'planned_retirement_date': 'generators_eia860',
}
"""
dict: a dictionary of qualifier column name (key) and original table (value).
"""


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
