"""hello, world."""

import logging
from copy import deepcopy

import pandas as pd

import pudl

logger = logging.getLogger(__name__)


INT_IDS = ['utility_id_ferc1', 'utility_id_pudl',
           'plant_id_pudl', 'report_year']

NA_VALUES = ["-", "—", "$-", ".", "_", "n/a", "N/A", "N/A $", "•"]

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
                dtypes={i: pd.Int64Dtype() for i in INT_IDS},
                na_values=NA_VALUES)
        )


class Transformer:
    """Transform class for cleaning depreciation study table."""

    def __init__(self, extract_df):
        """
        Initialize transform obect for cleaning depreciation study table.

        Args:
            extract (pandas.DataFrame):
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
            )
        return self.tidy_df

    def reshape(self, clobber=False):
        """
        Structural transformations.

        Right now, this implements ``split_allocate_common`` which grabs the
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
            filled_df = deepcopy(self.reshape(clobber=clobber))
            # convert % columns - which originally are a combination of whole
            # numbers of decimals (e.g. 88.2% would either be represented as
            # 88.2 or .882). Some % columns have boolean columns (ending in
            # num_or_pct) that we fleshed out to know wether the values were
            # reported as numbers or %s. There is one column that was easy to
            # clean by checking whether or not the value is greater than 1.
            filled_df.loc[filled_df['net_salvage_num_or_pct'],
                          'net_salvage_pct'] = filled_df.net_salvage_pct / 100
            filled_df.loc[filled_df['depreciation_annual_num_or_pct'],
                          'depreciation_annual_pct'] = \
                filled_df.depreciation_annual_pct / 100
            filled_df.loc[abs(filled_df.reserve_pct) >= 1,
                          'reserve_pct'] = filled_df.reserve_pct / 100
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

        idx_cols_deprish = [
            'report_date',
            'plant_id_pudl',
            'plant_name',
            'ferc_acct',
            'ferc_acct_full',
            'note',
        ]
        dupes = deprish_df[(deprish_df.duplicated(subset=idx_cols_deprish))
                           & (deprish_df.plant_id_pudl.notnull())]
        if not dupes.empty:
            raise ValueError(
                f"There are duplicate records of the depreciation records. "
                f"Check if there are duplicate with idx columns: "
                f"{idx_cols_deprish}"
            )

        merge_col = [x for x in idx_cols_deprish if x != 'plant_name']
        # TODO: restrict this merge on.. also the ferc_acct... idk
        # check Ft Saint Vrain Unit 1 as an example of error. Getting many
        # common records generating multiple instances of the same record w/
        # diff common $s
        df_w_c = (
            pd.merge(
                deprish_df,
                deprish_c_df[idx_cols_deprish + addtl_cols],
                on=merge_col,
                how='left',
                suffixes=('', common_suffix)
            )
        )
        return df_w_c

    def split_allocate_common(self):
        """
        Split and allocate the common plant depreciation lines.

        This function finds the common depreciation records and allocates the
        plant balance from the common records to the associated plant records.
        """
        # columns we want to split, suffix for the merged common cols, the new
        # data col we are trying to generate, and the id cols for groupby
        # Note: Some of these top level variables could easily be arguments to
        # this method instead of hard coded variables set here.... I'm just not
        # sure we'll ever want/need to.
        # Note #2: should the new_data_col actually end up being plant_balance?
        split_col = 'plant_balance'
        common_suffix = '_common'
        new_data_col = f'{split_col}_w{common_suffix}'
        idx_cols = ['report_date', 'plant_id_pudl', 'ferc_acct']

        deprish_w_c = (
            self.split_merge_common_records(common_suffix=common_suffix,
                                            addtl_cols=[split_col])
            .dropna(subset=idx_cols))
        # we want to know the sum of the potential split_cols for each ferc1
        # option
        df_w_tots = (
            pd.merge(
                deprish_w_c,
                deprish_w_c.groupby(by=idx_cols)[[split_col]]
                .sum().reset_index(),
                on=idx_cols,
                suffixes=("", "_fgb"))
        )
        # finally, calcuate the new column w/ the % of the total group. if
        # there is no common data, fill in this new data column with the og col
        df_w_tots[new_data_col] = (
            df_w_tots[f'{split_col}{common_suffix}']
            * (df_w_tots[split_col] / df_w_tots[f"{split_col}_fgb"])
        ).fillna(df_w_tots[split_col])

        # merge in the newly generated split/assigned data column
        deprish_w_c = pd.merge(
            deprish_w_c,
            df_w_tots[idx_cols + [new_data_col]].drop_duplicates(),
        )
        return deprish_w_c
