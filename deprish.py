"""
Extract and transform steps for depreciation studies.

Catalyst has compiled depreciation studies for a project with the Rocky
Mountain Institue. These studies were compiled from Public Utility Commission
proceedings as well as the FERC Form 1 table.

how to run this module:
file_path_deprish = pathlib.Path().cwd().parent/'depreciation_rmi.xlsx'
sheet_name_deprish='Depreciation Studies Raw'
transformer = deprish.Transformer(
    deprish.Extractor(
        file_path=file_path_deprish,
        sheet_name=sheet_name_deprish
    ).execute())
deprish_df = transformer.execute()
deprish_asset_df = agg_to_asset(deprish_df)

"""

import logging
from copy import deepcopy
import warnings
import sqlalchemy as sa

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process

import pudl
import make_plant_parts_eia

logger = logging.getLogger(__name__)


INT_IDS = ['utility_id_ferc1', 'utility_id_pudl',
           'plant_id_pudl', 'report_year']

NA_VALUES = ["-", "—", "$-", ".", "_", "n/a", "N/A", "N/A $", "•", "*"]

IDX_COLS_DEPRISH = [
    'report_date',
    'plant_id_pudl',
    'plant_part_name',
    'ferc_acct',
    'note',
    'utility_id_pudl',
    'data_source'
]

IDX_COLS_COMMON = [x for x in IDX_COLS_DEPRISH if x != 'plant_part_name']

# extract


class ExtractorPUDL:
    """
    Extractor for turning excel based depreciation data into a dataframe.

    Note: this should be overhualed if/when we switch from storing the
    depreciation studies into a CSV. Also, if/when we integrate this into pudl,
    we need to think more seriously about where to store the excel sheet/CSV.
    Is it in pudl.package_data or do we store it through the datastore? If it
    felt stable would it be worthwhile to store via zendo?.. in which case we
    will want to use a datastore object to handle the path.
    """

    def __init__(self,
                 file_path,
                 sheet_name,
                 skiprows=0):
        """
        Initialize a for deprish.Extractor.

        Args:
            file_path (path-like)
            sheet_name (str, int): String used for excel sheet name or
                integer used for zero-indexed sheet location.
            skiprows (int): rows to skip in zero-indexed column location,
                default is 0.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.skiprows = skiprows

    def execute(self):
        """Turn excel-based depreciation data into a dataframe."""
        logger.info(f"Reading the depreciation data from {self.file_path}")
        return (
            pd.read_excel(
                self.file_path,
                skiprows=self.skiprows,
                sheet_name=self.sheet_name,
                dtype={i: pd.Int64Dtype() for i in INT_IDS},
                na_values=NA_VALUES)
        )


class Transformer:
    """Transform class for cleaning depreciation study table."""

    def __init__(self, extract_df):
        """
        Initialize transform obect for cleaning depreciation study table.

        Args:
            extract_df (pandas.DataFrame): dataframe of extracted depreciation
                studies from ``Extractor.execute()``
        """
        # Note: should I pass in an instance of Extractor and make this call:
        # self.extract_df = extractor.execute()
        self.extract_df = extract_df

        self.tidy_df = None
        self.reshaped_df = None
        self.filled_df = None

    def execute(self, clobber=False):
        """
        Generate a transformed dataframe for the depreciation studies.

        Args:
            clobber (bool): if True and dataframe has already been generated,
                regenergate the datagframe.

        Returns:
            pandas.dataframe: depreciation study records that have been cleaned
            and nulls have been filled in.
        """
        self.tidy_df = self.early_tidy(clobber=clobber)
        # value transform
        self.filled_df = self.fill_in(clobber=clobber)
        self.reshaped_df = self.reshape(clobber=clobber)
        return self.reshaped_df

    def early_tidy(self, clobber=False):
        """Early transform type assignments and column assignments."""
        if clobber or self.tidy_df is None:
            # read in the depreciation sheet, assign types when required
            # we need the dtypes assigned early in this process because the
            # next steps involve splitting and filling in the null columns.
            self.tidy_df = (
                self.extract_df
                .pipe(self._convert_rate_cols)
                .pipe(pudl.helpers.convert_cols_dtypes,
                      'depreciation', name='depreciation')
                .assign(report_year=lambda x: x.report_date.dt.year)
                .pipe(pudl.helpers.simplify_strings, ['plant_part_name'])
            )
            # TODO: convert data_source='ferc' $1000s to $s
        return self.tidy_df

    def reshape(self, clobber=False):
        """
        Structural transformations.

        Right now, this implements ``split_allocate_common()`` which grabs the
        common records out of the main df and associates relevant dat columns
        with the related non-common records. In the end, we have a table that
        has no more common rows and a few extra columns that have neatly
        associated the common rows' data. We may need different types of
        reshaping later, so this method is here to accumulate reshaping
        methods.
        """
        if clobber or self.reshaped_df is None:
            self.reshaped_df = self.split_allocate_common()
        return self.reshaped_df

    def fill_in(self, clobber=False):
        """
        Clean % columns and fill in missing values.

        Args:
            clobber (bool): if True and dataframe has already been generated,
                regenergate the datagframe.

        Returns:
            pandas.DataFrame: depreciation study records that have been cleaned
            and nulls have been filled in.
        """
        if clobber or self.filled_df is None:
            filled_df = deepcopy(self.early_tidy())
            # convert % columns - which originally are a combination of whole
            # numbers of decimals (e.g. 88.2% would either be represented as
            # 88.2 or .882). Some % columns have boolean columns (ending in
            # type_pct) that we fleshed out to know wether the values were
            # reported as numbers or %s. There is one column that was easy to
            # clean by checking whether or not the value is greater than 1.
            filled_df.loc[filled_df['net_salvage_rate_type_pct'],
                          'net_salvage_rate'] = (
                filled_df.loc[filled_df['net_salvage_rate_type_pct'],
                              'net_salvage_rate'] / 100
            )
            filled_df.loc[filled_df['depreciation_annual_rate_type_pct'],
                          'depreciation_annual_rate'] = (
                filled_df.loc[filled_df['depreciation_annual_rate_type_pct'],
                              'depreciation_annual_rate'] / 100
            )
            filled_df.loc[abs(filled_df.reserve_rate) >= 1,
                          'reserve_rate'] = filled_df.loc[
                abs(filled_df.reserve_rate) >= 1, 'reserve_rate'] / 100
            logger.info(
                f"# of reserve_rate over 1 (100%): "
                f"{len(filled_df.loc[abs(filled_df.reserve_rate) >= 1])} "
                "Higher #s here may indicate an issue with the original data "
                "or the fill_in method"
            )
            # get rid of the bool columns we used to clean % columns
            filled_df = filled_df.drop(
                columns=filled_df.filter(like='num'))

            filled_df['net_salvage_rate'] = (
                - filled_df['net_salvage_rate'].abs()
            )
            filled_df['net_salvage'] = - filled_df['net_salvage'].abs()

            # then we need to do the actuall filling in
            def _fill_in_assign(filled_df):
                return filled_df.assign(
                    net_salvage_rate=lambda x:
                        # first clean % v num, then net_salvage/book_value
                        x.net_salvage_rate.fillna(
                            x.net_salvage / x.plant_balance),
                    net_salvage=lambda x:
                        x.net_salvage.fillna(
                            x.net_salvage_rate * x.plant_balance),
                    book_reserve=lambda x:
                        # step one, fill in w/ reserve_rate.
                        x.book_reserve.fillna(
                            (x.plant_balance * x.reserve_rate)
                            - x.net_salvage)
                        .fillna(x.plant_balance - x.net_salvage
                                - (x.depreciation_annual_epxns
                                   * x.remaining_life_avg)),
                    unaccrued_balance=lambda x:
                        x.unaccrued_balance.fillna(
                            x.plant_balance - x.book_reserve),
                    reserve_rate=lambda x: x.reserve_rate.fillna(
                        x.book_reserve / x.plant_balance),
                )
            # we want to do this filling in twice because the order matters.
            self.filled_df = _fill_in_assign(filled_df).pipe(_fill_in_assign)

        return self.filled_df

    def _convert_rate_cols(self, tidy_df):
        """Convert percent columns to numeric."""
        to_num_cols = ['net_salvage_rate',
                       'reserve_rate',
                       'depreciation_annual_rate']
        for col in to_num_cols:
            tidy_df[col] = pd.to_numeric(tidy_df[col])
        return tidy_df

    def split_merge_common_records(self,
                                   common_suffix='_common',
                                   addtl_cols=['plant_balance']):
        """
        Split apart common records and merge back specific columns.

        The depreciation studies
        Label and split the common records from the rest of the depreication
        records.

        Args:
            common_suffix (string): suffix that will be assigned to the common
                records ``addtl_cols`` when merging.
            addtl_cols (list of strings): list of columns that will be included
                with the common records when they are merged into the plant
                records that the commond records are assocaited with.
        """
        # there is a boolean column that is mostly nulls that let us flag
        # common records if they don't actually have 'common' in the plant name
        # so we are grabbing common records based on that bool column as well
        # as a string search of the plant name
        deprish_df = self.fill_in().assign(
            common=lambda x: x.common.fillna(
                x.plant_part_name.fillna('fake name so the col wont be null')
                .str.contains('common'))
        )

        # if there is no plant_id_pudl, there will be no plant for the common
        # record to be allocated across, so for now we need to assume these
        # records are not common
        deprish_c_df = deprish_df.loc[
            deprish_df.common & deprish_df.plant_id_pudl.notnull()
        ]
        deprish_df = deprish_df.loc[
            ~deprish_df.common | deprish_df.plant_id_pudl.isnull()]

        # we're going to capture the # of common records so we can check if we
        # get the right # of records in the end of the common munging
        self.common_len = len(deprish_c_df)
        self.plant_balance_c_og = deprish_c_df['plant_balance'].sum()
        logger.info(
            f"Common record rate: {self.common_len/len(deprish_df):.02%}")

        dupes = deprish_df.loc[(deprish_df.duplicated(subset=IDX_COLS_DEPRISH))
                               & (deprish_df.plant_id_pudl.notnull())]
        if not dupes.empty:
            # save it so we can see the dupes
            self.dupes = dupes
            raise ValueError(
                f"There are {len(dupes)} duplicate records of the depreciation"
                f" records. Check if there are duplicate with idx columns: "
                f"{IDX_COLS_DEPRISH}"
            )
        # there are some plants with mulitple common lines... bc utilities are
        # weird so we need to squish those together so we don't get a bunch of
        # duplicate records
        deprish_c_df = (
            deprish_c_df
            .groupby(by=IDX_COLS_COMMON, dropna=False)
            [addtl_cols].sum(min_count=1).reset_index()
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation')
        )

        # merge the common records in with the non-common records and merge the
        # counts/anys
        df_w_c = (
            pd.merge(
                deprish_df,
                deprish_c_df[IDX_COLS_COMMON + addtl_cols],
                on=IDX_COLS_COMMON,
                how='left',
                suffixes=('', common_suffix)
            )
        )
        if len(df_w_c) - len(deprish_df) != 0:
            raise AssertionError(
                f"whyyyy are there this many more records "
                f"{len(df_w_c) - len(deprish_df)} after we merge in the common"
                " records."
            )
        return df_w_c

    def split_allocate_common(self,
                              split_col='plant_balance',
                              common_suffix='_common'):
        """
        Split and allocate the common plant depreciation lines.

        The depreciations studies have common plant records sprinkled
        throughout, which represent the shared infrastructure (read capital in
        this context) of a plant with multiple units. Because we care about the
        sub-units of a plant, we don't actually care about the individual
        common records. We want to distribute the undepreciated plant balances
        associated with "common" records that pertain to no generation unit in
        particular, across all generation units, in proportion to each unit's
        own remaining plant balance.

        Args:
            split_col (string): column name of common records to split and
                allocate. Column must contain numeric data. Default
                'plant_balance'.
            common_suffix (string): suffix to use for the common columns when
                they are merged into the other plant-part records.
        """
        # the new  data col we are trying to generate
        new_data_col = f'{split_col}_w{common_suffix}'

        deprish_w_c = self.split_merge_common_records(
            common_suffix=common_suffix, addtl_cols=[split_col])

        simple_case_df = self.calc_common_portion_simple(
            deprish_w_c, split_col, common_suffix, new_data_col)
        edge_case_df = self.calc_common_portion_with_no_part_balance(
            deprish_w_c, split_col, common_suffix, new_data_col)

        deprish_w_common_allocated = pd.concat([simple_case_df, edge_case_df])

        # finally, calcuate the new column w/ the % of the total group. if
        # there is no common data, fill in this new data column with the og col
        deprish_w_common_allocated[new_data_col] = (
            deprish_w_common_allocated[f"{split_col}_common_portion"].fillna(0)
            + deprish_w_common_allocated[split_col].fillna(0))

        if len(deprish_w_common_allocated) != len(deprish_w_c):
            raise AssertionError(
                "smh.. the number of alloacted records "
                f"({len(deprish_w_common_allocated)}) don't match the "
                f"original records ({len(deprish_w_c)})... "
                "so something went wrong here."
            )
        _ = self._check_common_allocation(
            deprish_w_common_allocated, split_col, new_data_col, common_suffix)

        return deprish_w_common_allocated

    def calc_common_portion_simple(self,
                                   deprish_w_c,
                                   split_col,
                                   common_suffix,
                                   new_data_col):
        """
        Generate the portion of the common plant based on the split_col.

        Most of the deprecation records have data in our default ``split_col``
        (which is ``plant_balance``). For these records, calculating the
        portion of the common records to allocate to each plant-part is simple.
        This method calculates the portion of the common plant balance that
        should be allocated to each plant-part/ferc_acct records based on the
        ratio of each records' plant balance compared to the total
        plant/ferc_acct plant balance.
        """
        # exclude the nulls and the 0's
        simple_case_df = (
            deprish_w_c[
                (deprish_w_c[split_col].notnull())
                & (deprish_w_c[split_col] != 0)
            ]
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation'))
        logger.info(
            f"We are calculating the common portion for {len(simple_case_df)} "
            f"records w/ {split_col}")

        # we want to know the sum of the potential split_cols for each ferc1
        # option
        gb_df = (
            simple_case_df
            .groupby(by=IDX_COLS_COMMON, dropna=False)
            [[split_col]].sum(min_count=1).reset_index()
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation')
        )

        df_w_tots = (
            pd.merge(
                simple_case_df,
                gb_df,
                on=IDX_COLS_COMMON,
                how='left',
                suffixes=("", "_sum"))
        )

        df_w_tots[f"{split_col}_ratio"] = (
            df_w_tots[split_col] / df_w_tots[f"{split_col}_sum"]
        )

        # the default way to calculate each plant sub-part's common plant
        # portion is to multiply the ratio (calculated above) with the total
        # common plant balance for the plant/ferc_acct group.
        df_w_tots[f"{split_col}_common_portion"] = (
            df_w_tots[f'{split_col}{common_suffix}']
            * df_w_tots[f"{split_col}_ratio"])

        return df_w_tots

    def calc_common_portion_with_no_part_balance(self,
                                                 deprish_w_c,
                                                 split_col,
                                                 common_suffix,
                                                 new_data_col):
        """
        Calculate portion of common when ``split_col`` is null.

        There are a handfull of records where there is ``split_col`` values
        from the common records, but the ``split_col`` for that plant sub-part
        is null. In these cases, we still want to check if we need to assocaite
        a portion of the common ``split_col`` should be broken up based on the
        number of other records that the common value is assocaitd with
        (within the group of the ``IDX_COLS_COMMON``). We check to see if
        there are other plant sub-parts in the common plant grouping that have
        non-zero/non-null ``split_col`` - if they do then we don't assign the
        common portion to these records because their record relatives will be
        assigned the full common porportion in the
        ``calc_common_portion_simple()``.
        """
        # there are a handfull of records which have no plant balances
        # but do have common plant_balances.
        edge_case_df = deprish_w_c.loc[
            (deprish_w_c[split_col].isnull()) | (deprish_w_c[split_col] == 0)
        ]

        logger.info(
            f"We are calculating the common portion for {len(edge_case_df)} "
            f"records w/o {split_col}")

        # for future manipliations, we want a count of the number of records
        # within each group and have a bool column that lets us know whether or
        # not any of the records in a group have a plant balance
        edge_case_count = (
            deprish_w_c
            .assign(
                plant_bal_count=1,
                plant_bal_any=np.where(
                    deprish_w_c.plant_balance > 0,
                    True, False)
            )
            .groupby(by=IDX_COLS_COMMON, dropna=False)
            .agg({'plant_bal_count': 'count',
                  'plant_bal_any': 'any'})
            .reset_index()
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation')
        )
        edge_case_df = pd.merge(
            edge_case_df,
            edge_case_count,
            on=IDX_COLS_COMMON,
            how='left'
        )
        # if there is no other plant records with plant balances in the same
        # plant/ferc_acct group (denoted by the plant_bal_any column), we split
        # the plant balance evenly amoung the records using plant_bal_count.
        # if there are other plant sub part records with plant balances, the
        # common plant balance will already be distributed amoung those records
        edge_case_df[f"{split_col}_common_portion"] = np.where(
            ~edge_case_df['plant_bal_any'],
            (edge_case_df[f'{split_col}{common_suffix}'] /
             edge_case_df['plant_bal_count']),
            np.nan
        )

        return edge_case_df

    def _check_common_allocation(self,
                                 df_w_tots,
                                 split_col='plant_balance',
                                 new_data_col='plant_balance_w_common',
                                 common_suffix='_common'
                                 ):
        """Check to see if the common plant allocation was effective."""
        calc_check = (
            df_w_tots
            .groupby(by=IDX_COLS_DEPRISH, dropna=False)
            [[f"{split_col}_ratio", f"{split_col}_common_portion"]]
            .sum(min_count=1)
            .add_suffix("_check")
            .reset_index()
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation')
        )
        df_w_tots = pd.merge(
            df_w_tots, calc_check, on=IDX_COLS_DEPRISH, how='outer'
        )

        df_w_tots[f"{split_col}_common_portion_check"] = np.where(
            (df_w_tots.plant_balance.isnull() &
             df_w_tots.plant_balance_common.notnull()),
            df_w_tots[f"{split_col}_common_portion"] *
            df_w_tots["plant_bal_count"],
            df_w_tots[f"{split_col}_common_portion_check"]
        )

        # sum up all of the slices of the plant balance column.. these will be
        # used in the logs/asserts below
        plant_balance_og = (
            self.tidy_df[self.tidy_df.plant_part_name.notnull()][
                split_col].sum())
        plant_balance = df_w_tots[split_col].sum()
        plant_balance_w_common = df_w_tots[new_data_col].sum()
        plant_balance_c = (
            df_w_tots.drop_duplicates(
                subset=[c for c in IDX_COLS_DEPRISH if c != 'plant_part_name'],
                keep='first')
            [f"{split_col}{common_suffix}"].sum())

        logger.info(
            f"The resulting {split_col} allocated is "
            f"{plant_balance_w_common / plant_balance_og:.02%} of the original"
        )
        if plant_balance_w_common / plant_balance_og < .99:
            warnings.warn(
                f"ahhh the {split_col} allocation is off. The resulting "
                f"{split_col} is "
                f"{plant_balance_w_common/plant_balance_og:.02%} of the "
                f"original. og {plant_balance_og:.3} vs new: "
                f"{plant_balance_w_common:.3}"
            )

        if (plant_balance + plant_balance_c) / plant_balance_og < .99:
            warnings.warn(
                "well something went wrong here. even before proportionally "
                "assigning the common plant balance, the plant balance + "
                "common doesn't add up."
            )

        if len(df_w_tots) + self.common_len != len(self.fill_in()):
            warnings.warn(
                'ahhh we have a problem here with the number of records being '
                'generated here'
            )

        bad_ratio_check = (
            df_w_tots[~df_w_tots['plant_balance_ratio_check']
                      .round(0).isin([1, 0, np.nan])]
        )
        if not bad_ratio_check.empty:
            self.bad_ratio_check = bad_ratio_check
            warnings.warn(
                f"why would you do this?!?! there are {len(bad_ratio_check)} "
                f"records that are not passing our {split_col} check. "
                "The common records are being split and assigned incorrectly. "
            )
        # check for records w/ associated common plant balance
        no_common = df_w_tots[
            (df_w_tots.plant_balance_common.isnull()
             & (df_w_tots.plant_balance.notnull()))
            & (df_w_tots.plant_balance != df_w_tots.plant_balance_w_common)
        ]
        if not no_common.empty:
            warnings.warn(
                f"Ack! We have {len(no_common)} records that have no common "
                f"{split_col} but the og {split_col} is different than "
                f"the {new_data_col}"
            )
        return df_w_tots


def agg_to_asset(deprish_df):
    """
    Aggregate the depreciation data to the asset level.

    The depreciation data is reported at the plant-part and ferc_acct level.

    Args:
        deprish_df (pandas.DataFrame): table of depreciation data at the
            plant_part_name/ferc_acct level. Result of
            `Transformer().execute()`.

    Returns:
        pandas.DataFrame: table of depreciation data scaled down to the asset
        level. This functionally removes the granularity of the FERC account #.

    """
    # the unquie ids for the asset level are a subset of the IDX_COLS_DEPRISH
    idx_asset = [x for x in IDX_COLS_DEPRISH if x not in ['ferc_acct', 'note']]

    # prep for the groupby:
    # we have to break out the columns that need to be summed and the columns
    # which needs to be run through a weighted average aggregation for two
    # reasons. we need to insert the min_count=1 for the summed columns so we
    # don't end up with a bunch of 0's when it should be nulls. and because
    # there isn't a built in weighted average gb.agg function, so we need to
    # run it through our own.

    # sum agg section
    # enumerate sum cols
    sum_cols = [
        'plant_balance_w_common', 'plant_balance', 'book_reserve',
        'unaccrued_balance', 'net_salvage', 'depreciation_annual_epxns', ]
    # aggregate..
    deprish_asset = deprish_df.groupby(by=idx_asset)[sum_cols].sum(min_count=1)

    # weighted average agg section
    # enumerate wtavg cols
    avg_cols = ['service_life_avg', 'remaining_life_avg'] + \
        [x for x in deprish_df.columns
         if '_rate' in x and 'rate_type_pct' not in x]
    # prep dict with col to average (key) and col to weight on (value)
    # in this case we always want to weight based on unaccrued_balance
    wtavg_cols = {}
    for col in avg_cols:
        wtavg_cols[col] = 'unaccrued_balance'
    # aggregate..
    for data_col, weight_col in wtavg_cols.items():
        deprish_asset = (
            deprish_asset.merge(
                make_plant_parts_eia.weighted_average(
                    deprish_df,
                    data_col=data_col,
                    weight_col=weight_col,
                    by_col=idx_asset)
                .rename(columns={data_col: f"{data_col}_wt"}),
                how='outer', on=idx_asset))

    deprish_asset = deprish_asset.assign(
        remaining_life_avg=lambda x:
            x.unaccrued_balance / x.depreciation_annual_epxns,
        plant_balance_w_common_check=lambda x:
            x.book_reserve + x.unaccrued_balance,
        plant_balance_diff_check=lambda x:
            x.plant_balance_w_common_check / x.plant_balance_w_common,
    )

    return deprish_asset


def fill_in_tech_type(gens):
    """
    Fill in the generators' tech type based on energy source and prime mover.

    Args:
        gens (pandas.DataFrame): generators_eia860 table
    """
    # back fill the technology type
    idx_es_pm_tech = [
        'energy_source_code_1', 'prime_mover_code', 'technology_description'
    ]
    es_pm = ['energy_source_code_1', 'prime_mover_code']
    gens_f_pm_t = (
        gens.groupby(idx_es_pm_tech)
        [['plant_id_eia']].count().add_suffix('_count').reset_index()
    )

    logger.info(
        f"{len(gens_f_pm_t[gens_f_pm_t.duplicated(subset=es_pm)])} "
        "duplicate tech type mappings")
    tech_type_map = (
        gens_f_pm_t.sort_values('plant_id_eia_count', ascending=False)
        .drop_duplicates(subset=es_pm)
        .drop(columns=['plant_id_eia_count']))

    gens = (
        pd.merge(
            gens,
            tech_type_map,
            on=es_pm,
            how='left',
            suffixes=("", "_map"),
            validate="m:1"
        )
        .assign(technology_description=lambda x:
                x.technology_description.fillna(x.technology_description_map))
        .drop(columns=['technology_description_map'])
    )

    no_tech_type = gens[gens.technology_description.isnull()]
    logger.info(
        f"{len(no_tech_type)/len(gens):.01%} of generators don't map to tech "
        "types"
    )
    return gens
###########################################
# Grab and clean ferc1 depreciation table
###########################################


RESTRICT_MATCH_COLS = ['utility_id_ferc1']


class InputsCompiler:
    """Inputs Compiler for depreicaiton tables."""

    def __init__(self, ferc1_engine, pudl_engine):
        """
        Initialize inputs compiler for transforming the f1 deprceciation table.

        Args:
            ferc1_engine (sqlalchemy.engine.Engine): SQLAlchemy connection
                engine for the raw FERC Form 1 DB.
            pudl_engine (sqlalchemy.engine.Engine): SQLAlchemy connection
                engine for the PUDL DB.
        """
        print('hello')
        self.ferc1_engine = ferc1_engine
        self.pudl_engine = pudl_engine

        self.deprish_f1_raw = None
        self.plants_ferc1 = None
        self.utilities_ferc1 = None

    @staticmethod
    def get_full_sql_table(table_name, engine):
        """Get a full table from a SQL database."""
        pt = pudl.output.pudltabl.get_table_meta(engine)
        tbl = pt[table_name]
        select = sa.sql.select([tbl, ])
        return pd.read_sql(select, engine)

    def get_deprish_f1_raw(self):
        """Get the full depreciation table from the raw FERC1 database."""
        if self.deprish_f1_raw is None:
            self.deprish_f1_raw = self.get_full_sql_table(
                'f1_edcfu_epda', self.ferc1_engine
            )
        return self.deprish_f1_raw

    def get_plants_ferc1(self):
        """Get the full plants_ferc1 table from the PUDL database."""
        if self.plants_ferc1 is None:
            self.plants_ferc1 = self.get_full_sql_table(
                'plants_ferc1', self.pudl_engine
            )
        return self.plants_ferc1

    def get_utilities_ferc1(self):
        """Get the full utilities_ferc1 table from the PUDL database."""
        if self.utilities_ferc1 is None:
            self.utilities_ferc1 = self.get_full_sql_table(
                'utilities_ferc1', self.pudl_engine
            )
        return self.utilities_ferc1


class TransformerF1(Transformer):
    """
    Transformer for FERC1 deprecation table.

    This class enables the raw FERC1 deprceciation studies with the same
    methods as the PUDL compiled depreciation studies.

    """

    def __init__(self, inputs, *args, **kwargs):
        """Initialize the transformer for the FERC Form1 depreication table."""
        self.inputs = inputs
        super().__init__(*args, **kwargs)

    def early_tidy(self, clobber=False):
        """Override early_tidy method for the oddities of the FERC1 studies."""
        if clobber or self.tidy_df is None:
            self.tidy_df = (
                self.extract_df.rename(
                    columns={
                        'respondent_name': 'utility_name_ferc1',
                        'respondent_id': 'utility_id_ferc1',
                        'depr_plnt_base': 'plant_balance',
                        'est_avg_srvce_lf': 'total_life_avg',
                        'net_salvage': 'net_salvage_rate',
                        'apply_depr_rate': 'depreciation_annual_rate',
                        'mrtlty_crv_typ': 'survivor_curve_type',
                        'avg_remaining_lf': 'remaining_life_avg'
                    })
                .assign(
                    # null out non-numeric data in numeric columns
                    net_salvage_rate=lambda x:
                        pd.to_numeric(x.net_salvage_rate, errors='coerce'),
                    depreciation_annual_rate=lambda x:
                        pd.to_numeric(x.depreciation_annual_rate,
                                      errors='coerce'),
                    remaining_life_avg=lambda x:
                        pd.to_numeric(x.remaining_life_avg, errors='coerce'),
                    plant_balance=lambda x:
                        pd.to_numeric(x.plant_balance, errors='coerce'),
                    # add cols needed for add_plant_id_pudl-ing
                    report_date=lambda x: pd.to_datetime(
                        x.report_year, format='%Y'),
                    data_source='FERC',
                    note=np.nan,
                )
                .merge(  # add the utility_id_pudl
                    self.inputs.get_utilities_ferc1()[[
                        'utility_id_ferc1', 'utility_id_pudl']],
                    on=['utility_id_ferc1'],
                    how='left',
                    validate="m:1",
                )
                .pipe(self.split_acct_num)
                .pipe(self.add_plant_id_pudl)
                .pipe(self.agg_missing_ferc_acct)
                .assign(
                    net_salvage_rate_type_pct=True,
                    depreciation_annual_rate_type_pct=True,
                    # add in missing columns
                    common=pd.NA,
                    reserve_rate=np.nan,
                    book_reserve=np.nan,
                    unaccrued_balance=np.nan,
                    net_salvage=np.nan,
                    depreciation_annual_epxns=np.nan,
                )
                # replicate the functionality in the overriden method
                .pipe(self._convert_rate_cols)
                .pipe(pudl.helpers.convert_cols_dtypes,
                      'depreciation', name='depreciation')
            )
        return self.tidy_df

    def split_acct_num(self, deprish_f1_et):
        """
        Split the acct_num column into plant_part_name and ferc_acct cols.

        The depreication table includes an acct_num column which is trash. It
        generally contains the asset name (a plant or a sub-plant grouping like
        a unit) and the FERC account number. This method extracts the FERC
        account number and the plant part name from the acct_num column.
        """
        deprish_f1_et[['plant_part_name_1',
                       'ferc_acct',
                       'plant_part_name_2']] = (
            deprish_f1_et.acct_num.str.extract(
                pat=r"(^.*?)(\d{3}\.?\d{0,2})(.*?$)")
        )
        deprish_f1_et = (
            deprish_f1_et.assign(
                plant_part_name=lambda x:
                    (x.plant_part_name_1 + x.plant_part_name_2)
                    .fillna(x.acct_num)
            )
            .drop(columns=[  # 'acct_num',
                'plant_part_name_1', 'plant_part_name_2'])
            .pipe(remove_bad_string, col='plant_part_name', bad_string='-')
            .pipe(pudl.helpers.simplify_strings, ['plant_part_name'])
            .replace("", np.nan)
        )
        dumb_dumb_names = [
            '', '*', '**', '%', '()', '.', '+', ':',
            '=======================',
            '======================',
            '=====================',
            '========'
        ]
        deprish_f1_et.loc[:, 'plant_part_name'] = (
            deprish_f1_et.loc[:, 'plant_part_name']
            .replace(dumb_dumb_names, np.nan)
        )
        return deprish_f1_et

    def agg_missing_ferc_acct(self, deprish_f1_et):
        """
        Aggregate depreication records to IDX_COLS_DEPRISH.

        There are a ton of depreication records that have duplicate names in
        the raw FERC1 table. Many of these records appear to correspond to
        different FERC accounts. There are 8-10 records per plant per year
        which is consistent with other plants that have assocaited FERC
        accounts. Nonetheless, there is no way for us to really know which FERC
        account each record is associated. In order to use these records in
        future transformations, we need to aggreate the records on
        IDX_COLS_DEPRISH.
        """
        # we need to sum the plant_balance, but everything else should get a
        # weighted average.
        sum_cols = ['plant_balance']
        avg_cols = ['total_life_avg', 'net_salvage_rate',
                    'depreciation_annual_rate', 'remaining_life_avg']

        # aggregate..
        deprish_agg = deprish_f1_et.groupby(by=IDX_COLS_DEPRISH, dropna=False)[
            sum_cols].sum(min_count=1)

        # prep dict with col to average (key) and col to weight on (value)
        # in this case we always want to weight based on unaccrued_balance
        wtavg_cols = {}
        for col in avg_cols:
            wtavg_cols[col] = 'plant_balance'
        # aggregate..
        for data_col, weight_col in wtavg_cols.items():
            deprish_agg = (
                deprish_agg.merge(
                    make_plant_parts_eia.weighted_average(
                        deprish_f1_et,
                        data_col=data_col,
                        weight_col=weight_col,
                        by_col=IDX_COLS_DEPRISH),
                    how='outer', on=IDX_COLS_DEPRISH)
            )
        return deprish_agg

    def add_plant_id_pudl(self, deprish_f1_et):
        """Add plant_id_pudl via fuzzy matching on plant name when possible."""
        deprish_plant_parts = (
            self.prep_deprish_plants_for_matching(deprish_f1_et)
        )

        logger.info(
            "Fuzzy matching plant part names w/ ferc plant names. This will "
            "can take ~2 minutes with the full raw table.")
        deprish_plant_parts_matched = get_fuzzy_matches(
            df_left=deprish_plant_parts,
            df_right=self.inputs.get_plants_ferc1(),
            key_left='plant_part_name',
            key_right='plant_name_ferc1'
        )

        deprish_matches_w_ids = pd.merge(
            deprish_plant_parts_matched,
            self.inputs.get_plants_ferc1(),
            left_on=['utility_id_ferc1', 'plant_name_match'],
            right_on=['utility_id_ferc1', 'plant_name_ferc1'],
            validate='m:1'
        )
        deprish_w_ids = pd.merge(
            deprish_f1_et,
            deprish_matches_w_ids[['utility_id_ferc1',
                                   'plant_part_name', 'plant_id_pudl']],
            on=['utility_id_ferc1', 'plant_part_name'],
            validate='m:1',
            how='left'
        )
        return deprish_w_ids

    def prep_deprish_plants_for_matching(self, deprish_f1_et):
        """
        Prepare the depreication records for fuzzy matching.

        In this method we ensure that only one instance of each plant-part name
        from the depreicaiton table that corresponds to potential matches that
        show up in the FERC1 plants tables (plant for which we have generated
        plant_id_pudl's).

        """
        deprish_plant_parts = (
            deprish_f1_et[['utility_id_ferc1', 'plant_part_name', ]]
            .pipe(remove_bad_string, 'plant_part_name', 'common')
            .drop_duplicates()
            .dropna(subset=['plant_part_name', 'utility_id_ferc1'])
            .merge(
                self.inputs.get_plants_ferc1()
                [RESTRICT_MATCH_COLS].drop_duplicates(),
                on=RESTRICT_MATCH_COLS,
                validate='m:1'
            )
        )
        return deprish_plant_parts


def remove_bad_string(out_df, col, bad_string):
    """Remove bad strings."""
    out_df.loc[out_df[col].notnull(), col] = (
        out_df.loc[out_df[col].notnull(), col]
        .str.replace(bad_string, '')
    )
    return out_df


def get_plant_util_list(plant_name, df_left, df_right, key_left, key_right):
    """
    Get the possible key matches from df2 a plant_id_pudl and report_year.

    This selects for the plant id and year for each of the df1 records to
    match. This is for use within `get_fuzzy_matches`.
    """
    logger.debug(plant_name)
    options_index = (
        df_left.loc[df_left[key_left] == plant_name, RESTRICT_MATCH_COLS]
        .drop_duplicates().set_index(RESTRICT_MATCH_COLS).index)

    if options_index.empty:
        names = []
    else:
        # get the set of possible names
        names = (df_right.set_index(RESTRICT_MATCH_COLS)
                 .loc[options_index, key_right].to_list())
    return names


def get_fuzzy_matches(df_left, df_right, key_left, key_right, threshold=65):
    """Generate fuzzy matches."""
    # get the best match for each valye of the key1 column
    match_tuple_series = df_left.loc[:, key_left].apply(
        lambda x: process.extractOne(
            x, get_plant_util_list(x, df_left, df_right, key_left, key_right),
            scorer=fuzz.token_sort_ratio)
    )
    # process.extractOne returns a tuple with the matched name and the
    # match's score, so match_tuple_series contains tuples of the matching
    # plant name and the score. The plant_name_match assign only assigns the
    # match if the score is greater than or equal to the threshold.
    df_matched = df_left.assign(
        matches=match_tuple_series,
        plant_name_match=match_tuple_series.apply(
            lambda x: x[0] if x[1] >= threshold else None)
    )
    return df_matched
