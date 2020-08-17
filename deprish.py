"""
Extract and transform steps for depreciation studies.

Catalyst has compiled depreciation studies for a project with the Rocky
Mountain Institue. These studies were compiled from Public Utility Commission
proceedings as well as the FERC Form 1 table.
"""

import logging
from copy import deepcopy

import pandas as pd
import numpy as np

import pudl

logger = logging.getLogger(__name__)


INT_IDS = ['utility_id_ferc1', 'utility_id_pudl',
           'plant_id_pudl', 'report_year']

NA_VALUES = ["-", "—", "$-", ".", "_", "n/a", "N/A", "N/A $", "•"]

IDX_COLS_DEPRISH = [
    'report_date',
    'plant_id_pudl',
    'plant_name',
    'ferc_acct',
    # 'ferc_acct_full',
    'note',
]

IDX_WO_PLANT_NAME = [x for x in IDX_COLS_DEPRISH if x != 'plant_name']

# extract


class Extractor:
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
        self.reshaped_df = self.reshape(clobber=clobber)
        # value transform
        self.filled_df = self.fill_in(clobber=clobber)
        return self.filled_df

    def early_tidy(self, clobber=False):
        """Early transform type assignments and column assignments."""
        if clobber or self.tidy_df is None:
            # read in the depreciation sheet, assign types when required
            # we need the dtypes assigned early in this process because the
            # next steps involve splitting and filling in the null columns.
            self.tidy_df = (
                self.extract_df
                .pipe(self._convert_pct_cols)
                .pipe(pudl.helpers.convert_cols_dtypes,
                      'depreciation', name='depreciation')
                .assign(report_year=lambda x: x.report_date.dt.year)
                .pipe(pudl.helpers.strip_lower, ['plant_name'])
            )
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
            filled_df = deepcopy(self.reshape())
            # convert % columns - which originally are a combination of whole
            # numbers of decimals (e.g. 88.2% would either be represented as
            # 88.2 or .882). Some % columns have boolean columns (ending in
            # num_or_pct) that we fleshed out to know wether the values were
            # reported as numbers or %s. There is one column that was easy to
            # clean by checking whether or not the value is greater than 1.
            filled_df.loc[filled_df['net_salvage_num_or_pct'],
                          'net_salvage_pct'] = (
                filled_df.loc[filled_df['net_salvage_num_or_pct'],
                              'net_salvage_pct'] / 100
            )
            filled_df.loc[filled_df['depreciation_annual_num_or_pct'],
                          'depreciation_annual_pct'] = (
                filled_df.loc[filled_df['depreciation_annual_num_or_pct'],
                              'depreciation_annual_pct'] / 100
            )
            filled_df.loc[abs(filled_df.reserve_pct) >= 1,
                          'reserve_pct'] = filled_df.loc[
                abs(filled_df.reserve_pct) >= 1, 'reserve_pct'] / 100
            logger.info(
                f"# of reserve_pct over 100%: "
                f"{len(filled_df.loc[abs(filled_df.reserve_pct) >= 1])} "
                "Higher #s here may indicate an issue with the original data "
                "or the fill_in method"
            )
            # get rid of the bool columns we used to clean % columns
            filled_df = filled_df.drop(
                columns=filled_df.filter(like='num_or_pct'))

            # then we need to do the actuall filling in
            self.filled_df = filled_df.assign(
                net_salvage_pct=lambda x:
                    # first clean % v num, then net_salvage/book_value
                    x.net_salvage_pct.fillna(x.net_removal / x.book_reserve),
                net_removal=lambda x:
                    x.net_removal.fillna(x.net_salvage_pct * x.book_reserve),
                unaccrued_balance=lambda x:
                    x.unaccrued_balance.fillna(
                        x.plant_balance_w_common - x.book_reserve
                        + x.net_removal),
                reserve_pct=lambda x: x.book_reserve / x.plant_balance_w_common
            )

        return self.filled_df

    def _convert_pct_cols(self, tidy_df):
        """Convert percent columns to numeric."""
        to_num_cols = ['net_salvage_pct',
                       'reserve_pct',
                       'depreciation_annual_pct']
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
        deprish_df = self.early_tidy().assign(
            common=lambda x: x.common.fillna(
                x.plant_name.str.contains('common')))

        deprish_c_df = deprish_df.loc[deprish_df.common]
        deprish_df = deprish_df.loc[~deprish_df.common]

        # we're going to capture the # of common records so we can check if we
        # get the right # of records in the end of the common munging
        self.common_len = len(deprish_c_df)
        self.plant_balance_c_og = deprish_c_df['plant_balance'].sum()
        logger.info(
            f"Common record rate: {self.common_len/len(deprish_df):.02%}")

        dupes = deprish_df[(deprish_df.duplicated(subset=IDX_COLS_DEPRISH))
                           & (deprish_df.plant_id_pudl.notnull())]
        if not dupes.empty:
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
            .groupby(by=IDX_WO_PLANT_NAME, dropna=False)
            [addtl_cols].sum().reset_index()
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'depreciation', name='depreciation')
        )

        # merge the common records in with the non-common records and merge the
        # counts/anys
        df_w_c = (
            pd.merge(
                deprish_df,
                deprish_c_df[IDX_WO_PLANT_NAME + addtl_cols],
                on=IDX_WO_PLANT_NAME,
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

    def split_allocate_common(self):
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
        """
        # columns we want to split, suffix for the merged common cols, the new
        # data col we are trying to generate, and the id cols for groupby
        # Note: Some of these top level variables could easily be arguments to
        # this method instead of hard coded variables set here.... I'm just not
        # sure we'll ever want/need to.
        split_col = 'plant_balance'
        common_suffix = '_common'
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
            deprish_w_common_allocated[f"{split_col}_c_portion"].fillna(0)
            + deprish_w_common_allocated[split_col].fillna(0))

        if len(deprish_w_common_allocated) != len(deprish_w_c):
            raise AssertionError(
                "smh.. the number of alloacted records don't match the "
                "original records... so something went wrong here."
            )
        self._check_common_allocation(
            deprish_w_common_allocated, split_col, new_data_col, common_suffix)

        return deprish_w_common_allocated

    def calc_common_portion_simple(self,
                                   deprish_w_c,
                                   split_col,
                                   common_suffix,
                                   new_data_col):
        """
        Generate the portion of the common plant based on the split_col.

        Most of the deprecation records have data in our standard ``split_col``
        (which is ``plant_balance``). For these records, calculating the
        portion of the common records to allocate to each subpart is simple.
        This method calculated the ratio of the total ``split_col`` vs the
        summed ``split_col`` within the group of the ``IDX_WO_PLANT_NAME``.
        That ratio is used to generate the porportion of the common's
        ``split_col`` to allocate to each plant sub-part record.
        """
        simple_case_df = deprish_w_c[
            (deprish_w_c[split_col].notnull())
        ]
        logger.info(
            f"We are calculating the common portion for {len(simple_case_df)} "
            f"records w/ {split_col}")

        # we want to know the sum of the potential split_cols for each ferc1
        # option
        gb_df = (
            simple_case_df
            .groupby(by=IDX_WO_PLANT_NAME, dropna=False)
            [[split_col]].sum().reset_index()
        )

        df_w_tots = (
            pd.merge(
                simple_case_df,
                gb_df.reset_index(),
                on=IDX_WO_PLANT_NAME,
                how='left',
                suffixes=("", "_sum"))
        )

        df_w_tots[f"{split_col}_ratio"] = (
            df_w_tots[split_col] / df_w_tots[f"{split_col}_sum"]
        )

        # the default way to calculate each plant sub-part's common plant
        # portion is to multiply the ratio (calculated above) with the total
        # common plant balance for the plant/ferc_acct group.
        df_w_tots[f"{split_col}_c_portion"] = (
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
        (within the group of the ``IDX_WO_PLANT_NAME``). We check to see if
        there are other plant sub-parts in the common plant grouping that have
        non-zero/non-null ``split_col`` - if they do then we don't assign the
        common portion to these records because their record relatives will be
        assigned the full common porportion in the
        ``calc_common_portion_simple()``.
        """
        # there are a handfull of records which have no plant balances
        # but do have common plant_balances.
        edge_case_df = deprish_w_c[
            (deprish_w_c[split_col].isnull())
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
            .groupby(by=IDX_WO_PLANT_NAME, dropna=False)
            .agg({'plant_bal_count': 'count',
                  'plant_bal_any': 'any'})
        )
        edge_case_df = pd.merge(
            edge_case_df,
            edge_case_count.reset_index(),
            on=IDX_WO_PLANT_NAME,
            how='left'
        )
        # if there is no other plant records with plant balances in the same
        # plant/ferc_acct group (denoted by the plant_bal_any column), we split
        # the plant balance evenly amoung the records using plant_bal_count.
        # if there are other plant sub part records with plant balances, the
        # common plant balance will already be distributed amoung those records
        edge_case_df[f"{split_col}_c_portion"] = np.where(
            ~edge_case_df['plant_bal_any'],
            (edge_case_df[f'{split_col}{common_suffix}'] /
             edge_case_df['plant_bal_count']),
            np.nan
        )

        return edge_case_df

    def _check_common_allocation(self,
                                 df_w_tots,
                                 split_col,
                                 new_data_col,
                                 common_suffix):
        """Check to see if the common plant allocation was effective."""
        calc_check = (
            df_w_tots
            .groupby(by=IDX_COLS_DEPRISH, dropna=False)
            [[f"{split_col}_ratio", f"{split_col}_c_portion"]]
            .sum()
            .add_suffix("_check")
            .reset_index()
        )
        df_w_tots = pd.merge(
            df_w_tots, calc_check, on=IDX_COLS_DEPRISH, how='left'
        )

        df_w_tots[f"{split_col}_c_portion_check"] = np.where(
            (df_w_tots.plant_balance.isnull() &
             df_w_tots.plant_balance_common.notnull()),
            df_w_tots[f"{split_col}_c_portion"] * df_w_tots["plant_bal_count"],
            df_w_tots[f"{split_col}_c_portion_check"]
        )

        # sum up all of the slices of the plant balance column.. these will be
        # used in the logs/asserts below
        plant_balance_og = self.tidy_df[split_col].sum()
        plant_balance = df_w_tots[split_col].sum()
        plant_balance_w_common = df_w_tots[new_data_col].sum()
        plant_balance_c = (
            df_w_tots.drop_duplicates(
                subset=[c for c in IDX_COLS_DEPRISH if c != 'plant_name'],
                keep='first')
            [f"{split_col}{common_suffix}"].sum())

        logger.info(
            f"The resulting {split_col} allocated is "
            f"{plant_balance_w_common / plant_balance_og:.02%} of the original"
        )
        if plant_balance_w_common / plant_balance_og < .99:
            raise AssertionError(
                f"ahhh the {split_col} allocation is off. The resulting "
                f"{split_col} is "
                f"{plant_balance_w_common/plant_balance_og:.02%} of the "
                f"original. og {plant_balance_og:.3} vs new: "
                f"{plant_balance_w_common:.3}"
            )

        if (plant_balance + plant_balance_c) / plant_balance_og < .99:
            raise AssertionError(
                "well something went wrong here. even before proportionally "
                "assigning the common plant balance, the plant balance + "
                "common doesn't add up."
            )

        if len(df_w_tots) + self.common_len != len(self.early_tidy()):
            raise AssertionError(
                'ahhh we have a problem here with the number of records being '
                'generated here'
            )

        bab_ratio_check = (df_w_tots[~df_w_tots['plant_balance_ratio_check']
                                     .round(0).isin([1, 0])])
        if not bab_ratio_check.empty:
            raise AssertionError(
                f"why would you do this?!?! there are {len(bab_ratio_check)} "
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
            raise AssertionError(
                f"Ack! We have {len(no_common)} records that have no common "
                f"{split_col} but the og {split_col} is different than "
                f"the {new_data_col}"
            )
