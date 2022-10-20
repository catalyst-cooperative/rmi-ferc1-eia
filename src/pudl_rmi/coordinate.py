"""
Coordinate the acquisition and generation of RMI's interrelated outputs.

The outputs in this repo are dependent on one another. See `README` for a
diagram of the relations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pudl
import sqlalchemy as sa
from pudl.output.pudltabl import PudlTabl

import pudl_rmi
from pudl_rmi import make_plant_parts_eia

# from memory_profiler import profile


logger = logging.getLogger(__name__)


class Output:
    """Class to manage all of the interconnected RMI outputs."""

    def __init__(self, pudl_out):
        """
        Initialize output coordinator for the RMI ouptus.

        This object enables short term disk caching and an easy way to
        regenerate any or all of the interconnected RMI outputs. Each `get_`
        method in this object grabs a pickle file off disk if it is available,
        or generates the output if it is not available and/or if you set
        clobber to True.

        Because some outputs rely on others, this object enables clobbering of
        both the main output and the outputs which the main object relies on.
        This in effect enables a clobber and deep clobbers.

        Most of the outputs are generated via a ``execute()`` function from the
        cooresponding module for that output. The plant-part list is generated
        from the `pudl_out` object.

        Args:
            pudl_out (object): instance of `pudl.output.pudltabl.PudlTabl()`.
                The frequency (`freq`) of `pudl_out` must be `AS`. For best
                results  `fill_fuel_cost`, `roll_fuel_cost`, and `fill_net_gen`
                should all be True. `start_date` and `end_date` should be set
                if using only a portion of the EIA data.
        """
        self.pudl_out = pudl_out
        if pudl_out.freq != "AS":
            raise AssertionError(
                f"Frequency of `pudl_out` must be `AS` but got {pudl_out.freq}"
            )

    # @profile
    def plant_parts_eia(self, clobber=False, pickle_train_connections=False):
        """
        Get the EIA plant-parts; generate it or get if from a file.

        If you generate the PPE, it will be saved at the file path given. The
        EIA plant-parts is generated via the pudl_out object.

        TODO: Change to ``plant_parts_eia()`` when there aren't a bunch of
        branches using these bbs.

        Args:
            clobber (boolean): True if you want to regenerate the EIA
                plant-parts whether or not the output is already pickled.
                Default is False.
            pickle_train_connections (boolean): True if you also want to connect
                and pickle the connection between the training data and EIA
                plant-parts for use in EIA to FERC1 matching. This is
                primarily used for memory efficiency when running the CI.
                Default is False.
        """
        file_path = pudl_rmi.PLANT_PARTS_EIA_PKL
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber:
            logger.info(
                f"Master unit list not found {file_path} Generating a new "
                "master unit list. This should take ~10 minutes."
            )
            # actually make the master plant parts list
            plant_parts_eia = self.pudl_out.plant_parts_eia()
            # verify that record_id_eia is the index
            if plant_parts_eia.index.name != "record_id_eia":
                logger.error("Plant parts list index is not record_id_eia.")
            plant_parts_eia = plant_parts_eia[
                ~plant_parts_eia.index.duplicated(keep="first")
            ]
            # export
            plant_parts_eia.to_pickle(file_path)
        else:
            logger.info(f"Reading the EIA plant-parts from {file_path}")
            plant_parts_eia = pd.read_pickle(file_path)
            if plant_parts_eia.index.name != "record_id_eia":
                logger.error("Plant parts list index is not record_id_eia.")
        # more efficient memory use for CI
        if pickle_train_connections:
            pudl_rmi.connect_ferc1_to_eia.prep_train_connections(
                ppe=plant_parts_eia,
                start_date=self.pudl_out.start_date,
                end_date=self.pudl_out.end_date,
            ).to_pickle(
                pudl_rmi.CONNECTED_TRAIN_PKL,
            )

        return plant_parts_eia

    def plant_parts_eia_distinct(self, clobber=False, clobber_ppe=False):
        """Get the EIA plant_parts with only the unique granularities.

        Read in the pickled dataframe or generate it from the full PPE. Get only
        the records of the PPE that are "true granularities" and those which are not
        duplicates based on their ownership so the FERC to EIA matching model
        doesn't get confused as to which option to pick if there are many records
        with duplicate data.

        Arguments:
            clobber (boolean): True if you want to regenerate the distinct
                plant parts list whether or not the output is already pickled.
                Default is False.
            clobber_ppe (boolean): True if you want to regenerate the full EIA
                plant parts list whether or not the output is already pickled.
                Default is False.
        """
        file_path = pudl_rmi.DISTINCT_PLANT_PARTS_EIA_PKL
        check_is_file_or_not_exists(file_path)
        clobber_any = any([clobber, clobber_ppe])
        if not file_path.exists() or clobber_any:
            logger.info(
                f"Distinct EIA plant-parts not found at {file_path}. Generating a new "
                "distinct dataframe."
            )
            plant_parts_eia = self.plant_parts_eia(clobber=clobber_ppe)
            plant_parts_eia = plant_parts_eia.assign(
                plant_id_report_year_util_id=lambda x: x.plant_id_report_year
                + "_"
                + x.utility_id_pudl.map(str)
            ).astype({"installation_year": "float"})
            distinct_ppe = plant_parts_eia[
                (plant_parts_eia["true_gran"]) & (~plant_parts_eia["ownership_dupe"])
            ]
            distinct_ppe.to_pickle(file_path)
        else:
            logger.info(f"Reading the distinct EIA plant-parts from {file_path}")
            distinct_ppe = pd.read_pickle(file_path)
        if distinct_ppe.index.name != "record_id_eia":
            logger.error("Plant parts list index is not record_id_eia.")
        return distinct_ppe

    # @profile
    def deprish(self, clobber=False, start_year=None, end_year=None):
        """
        Generate or grab the cleaned depreciation studies.

        Args:
            clobber (boolean): True if you want to regenerate the depreciation
                data whether or not the output is already pickled. Default is
                False.
            start_year (int): The start year of the date range to extract.
                Default is None and all years before end_year will be extracted.
            end_year (int): The end year of the date range to extract.
                Default is None and all years after start_year will be extracted.
        """
        file_path = pudl_rmi.DEPRISH_PKL
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber:
            logger.info("Generating new depreciation study output.")
            deprish = pudl_rmi.deprish.execute(start_year=start_year, end_year=end_year)
            deprish.to_pickle(file_path)
        else:
            logger.info(f"Grabbing depreciation study output from {file_path}")
            deprish = pd.read_pickle(file_path)
        return deprish

    # @profile
    def deprish_to_eia(
        self,
        clobber: bool = False,
        clobber_deprish: bool = False,
        clobber_plant_parts_eia: bool = False,
        save_to_xlsx: bool = False,
    ):
        """
        Generate or grab the connection between the depreciation data and EIA.

        Args:
            clobber: True if you want to regenerate the connection between the
                depreciation data and EIA whether or not the output is already
                pickled. Default is False.
            clobber_deprish : True if you want to regenerate the depreciation
                data whether or not the output is already pickled. The
                depreciation data is an interim input to make the connection
                between depreciation and EIA. Default is False.
            clobber_plant_parts_eia: True if you want to regenerate the EIA
                plant-part list whether or not the output is already pickled.
                Default is False.
            save_to_xlsx: If True, save the output of this process to an excel
                file (`pudl_rmi.DEPRISH_RAW_XLSX`). Default is False. If you
                haven't updated the mannual mapping in `pudl_rmi.DEPRISH_RAW_XLSX`
                it is recommended to not save because it takes up lotsa git
                space. If you do update the overrides, it's recommended that
                you run this with True. Issue #169 will deprecate this.
        """
        clobber_any = any([clobber, clobber_deprish, clobber_plant_parts_eia])
        file_path = pudl_rmi.DEPRISH_EIA_PKL
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber_any:
            deprish_eia = pudl_rmi.connect_deprish_to_eia.execute(
                deprish=self.deprish(clobber=clobber_deprish),
                plant_parts_eia=self.plant_parts_eia(clobber=clobber_plant_parts_eia),
                save_to_xlsx=save_to_xlsx,
            )
            deprish_eia.to_pickle(file_path)
        else:
            deprish_eia = pd.read_pickle(file_path)
        return deprish_eia

    def ferc1_to_eia(
        self,
        clobber=False,
        clobber_plant_parts_eia=False,
        clobber_plant_parts_eia_distinct=False,
        five_year_test=False,
    ):
        """
        Generate or grab a connection between FERC1 and EIA.

        Either generate or grab an on-disk cached version of the connection
        between the FERC1 plant data and the EIA plant part list.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not it is saved at the file_path_mul
            clobber_plant_parts_eia (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
            clobber_plant_parts_eia_distinct(boolean): Generate and cache a new
                output of the distinct EIA plant part list
            five_year_test (boolean): Whether the connection is being made with
                five years of FERC and EIA data for integration testing.
                Default is False.
        """
        file_path = pudl_rmi.FERC1_EIA_PKL
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any(
            [clobber, clobber_plant_parts_eia, clobber_plant_parts_eia_distinct]
        )
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber_any:
            logger.info(
                f"FERC to EIA granular connection not found at {file_path}... "
                "Generating a new output."
            )
            # get or generate connected training data
            train_file_path = pudl_rmi.CONNECTED_TRAIN_PKL
            check_is_file_or_not_exists(train_file_path)
            if not train_file_path.exists():
                train_df = None
            else:
                train_df = pd.read_pickle(train_file_path)
            ferc1_eia = pudl_rmi.connect_ferc1_to_eia.execute(
                train_df,
                self.pudl_out,
                self.plant_parts_eia_distinct(
                    clobber=clobber_plant_parts_eia_distinct,
                    clobber_ppe=clobber_plant_parts_eia,
                ),
                five_year_test=five_year_test,
            )
            # export
            ferc1_eia.to_pickle(file_path)
        else:
            logger.info(f"Reading the FERC to EIA connection from {file_path}")
            ferc1_eia = pd.read_pickle(file_path)
        return ferc1_eia

    # @profile
    def deprish_to_ferc1(
        self,
        clobber=False,
        clobber_plant_parts_eia=False,
        clobber_deprish=False,
        clobber_deprish_eia=False,
        clobber_ferc1_eia=False,
    ):
        """
        Generate or grab a connection between deprecaiton data and FERC1.

        Either generate or grab an on-disk cached version of the connection
        between the depreciation study data and FERC1.

        Args:
            clobber (boolean): Generate and cache a new output even if it
                exists on disk. This does not necessarily regenerate the
                interim inputs - see other `clobber_` arguments. Default is
                False.
            clobber_plant_parts_eia (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
            clobber_deprish : True if you want to regenerate the depreciation
                data whether or not the output is already pickled. The
                depreciation data is an interim input to make the connection
                between depreciation and EIA. Default is False.
            clobber_deprish_eia (boolean): Generate and cache a new interim
                output of the connection between EIA and depreciation data and
                generate a new version of the depreciaiton to FERC1 output.
                Default is False.
            clobber_ferc1_eia (boolean): Generate and cache a new interim
                output of the connection between FERC and EIA and generate a
                new version of the depreciaiton to FERC1 output. Default is
                False.

        Returns:
            pandas.DataFrame: depreciation study data connected to FERC1 data
            which has been scaled down or aggregated to the level of reporting
            in the depreciaiton studies.
        """
        file_path = pudl_rmi.DEPRISH_FERC1_PKL
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any(
            [
                clobber,
                clobber_deprish,
                clobber_plant_parts_eia,
                clobber_deprish_eia,
                clobber_ferc1_eia,
            ]
        )
        check_is_file_or_not_exists(file_path)

        if not file_path.exists() or clobber_any:
            logger.info(
                "Deprish to FERC1 granular connection not found at "
                f"{file_path}. Generating a new output."
            )
            deprish_ferc1 = pudl_rmi.connect_deprish_to_ferc1.execute(
                plant_parts_eia=self.plant_parts_eia(clobber=clobber_plant_parts_eia),
                deprish_eia=self.deprish_to_eia(
                    clobber=clobber_deprish_eia,
                    clobber_deprish=clobber_deprish,
                ),
                ferc1_eia=self.ferc1_to_eia(clobber=clobber_ferc1_eia),
            )
            # export
            deprish_ferc1.to_pickle(file_path)

        else:
            logger.info(f"Reading the depreciation to FERC connection from {file_path}")
            deprish_ferc1 = pd.read_pickle(file_path)
        return deprish_ferc1

    def optimus(self, clobber_ferc1_eia: bool = False, **kwargs):
        """
        Generate output in Optimus format.

        Args:
            clobber_ferc1_eia (boolean): Generate and cache a new interim
                output of the connection between FERC and EIA and generate a
                new version of the depreciaiton to FERC1 output.
            kwargs: Additional kwargs to be passed into
                :func:``pudl_rmi.formatter_optimus.select_from_deprish_ferc1``
                if you want to select specific utilities/years/data sources.
        """
        optimus_out = pudl_rmi.formatter_optimus.execute(
            deprish_ferc1_eia=self.deprish_to_ferc1(clobber=clobber_ferc1_eia),
            plants_eia860=self.pudl_out.plants_eia860(),
            utils_eia860=self.pudl_out.utils_eia860(),
            **kwargs,
        )
        return optimus_out

    def run_all(self, clobber_all=False):
        """
        Gotta catch em all. Get all of the RMI outputs.

        Read from disk or regenerate all of the RMI outputs. This method is mostly for
        testing purposes because it returns all 5 outputs. To grab individual outputs,
        it is recommended to use the output-specific method.

        Args:
            clobber_all (boolean): Deafult is False, which will read saved
                outputs from disk if they already exist, or generate them if
                they don't. True will re-calculate the outputs regardless of
                whether they exist on disk, and save them to disk.
                Re-generating everything will take ~15 minutes.

        Returns:
            pandas.DataFrame: EIA plant-part list - table of "plant-parts"
                which are groups of aggregated EIA generators that coorespond
                to portions of plants from generators to fuel types to whole
                plants.
            pandas.DataFrame: a table of depreciation studies. These records
                have been cleaned and standardized with plant's "common" lines
                allocated across their cooresponding plant records.
            pandas.DataFrame: a table of the connection between the
                depreciation studies and the EIA plant-parts list.
            pandas.DataFrame: a table of the connection between the FERC1
                plants and the EIA plant-parts list.
            pandas.DataFrame: a table of the conneciton between the
                depreciation studies and the FERC1 plants.
        """
        ppl = self.plant_parts_eia(clobber=clobber_all)
        d = self.deprish(clobber=clobber_all)
        de = self.deprish_to_eia(clobber=clobber_all)
        fe = self.ferc1_to_eia(clobber=clobber_all)
        df = self.deprish_to_ferc1(clobber=clobber_all)
        return ppl, d, de, fe, df


def check_is_file_or_not_exists(file_path: Path):
    """
    Raise assertion if the path exists but is not a file. Do nothing if not.

    Raises:
        AssertionError: If the path exists but is not a file - i.e. if the path
            is a directory, this assertion will be raised so the directory
            isn't wiped out.
    """
    if file_path.exists() and not file_path.is_file():
        raise AssertionError(
            f"Path exists but is not a file. Check if {file_path} is a "
            "directory. It should be either a pickled file or nothing."
        )


def prep_train_connections(ppe, start_date=None, end_date=None):
    """
    Get and prepare the training connections.

    We have stored training data, which consists of records with ids
    columns for both FERC and EIA. Those id columns serve as a connection
    between ferc1 plants and the EIA plant-parts. These connections
    indicate that a ferc1 plant records is reported at the same granularity
    as the connected EIA plant-parts record. These records to train a
    machine learning model.

    Arguments:
        ppe (pandas.DataFrame): The EIA plant parts list. Records from
            this dataframe will be connected to the training data records.
        start_date (pd.Timestamp): Beginning date for records from the
            training data. Should match the start date of `ppe`. Default
            is None and all the training data will be used.
        end_date (pd.Timestamp): Ending date for records from the
            training data. Should match the end date of `ppe`. Default is
            None and all the training data will be used.

    Returns:
        pandas.DataFrame: training connections. A dataframe with has a
        MultiIndex with record_id_eia and record_id_ferc1.
    """
    ppe_cols = [
        "true_gran",
        "appro_part_label",
        "appro_record_id_eia",
        "plant_part",
        "ownership_dupe",
    ]
    train_df = (
        # we want to ensure that the records are associated with a
        # "true granularity" - which is a way we filter out whether or
        # not each record in the EIA plant-parts is actually a
        # new/unique collection of plant parts
        # once the true_gran is dealt with, we also need to convert the
        # records which are ownership dupes to reflect their "total"
        # ownership counterparts
        pd.read_csv(
            pudl_rmi.TRAIN_FERC1_EIA_CSV,
        )
        .pipe(pudl.helpers.cleanstrings_snake, ["record_id_eia"])
        .drop_duplicates(subset=["record_id_ferc1", "record_id_eia"])
    )
    # filter training data by year range
    # first get list of all years to grab from training data
    if start_date is None and end_date is None:
        years = None
    elif start_date is None:
        years = [
            str(year) for year in np.arange(ppe.report_year.min(), end_date.year + 1)
        ]
    elif end_date is None:
        years = [
            str(year) for year in np.arange(start_date.year, ppe.report_year.max() + 1)
        ]
    else:
        years = [str(year) for year in np.arange(start_date.year, end_date.year + 1)]
    if years is not None:
        train_df = train_df[
            pd.DataFrame(train_df.record_id_eia.str.split("_").tolist())
            .isin(years)
            .any(1)
            .values
        ]

    train_df = (
        train_df.merge(
            ppe[ppe_cols].reset_index(),
            how="left",
            on=["record_id_eia"],
            indicator=True,
        )
        .assign(
            plant_part=lambda x: x["appro_part_label"],
            record_id_eia=lambda x: x["appro_record_id_eia"],
        )
        .pipe(make_plant_parts_eia.reassign_id_ownership_dupes)
        .fillna(
            value={
                "record_id_eia": pd.NA,
            }
        )
        .set_index(  # recordlinkage and sklearn wants MultiIndexs to do the stuff
            [
                "record_id_ferc1",
                "record_id_eia",
            ]
        )
    )
    not_in_ppe = train_df[train_df._merge == "left_only"]
    # if not not_in_ppe.empty:
    if len(not_in_ppe) > 12:
        raise AssertionError(
            "Not all training data is associated with EIA records.\n"
            "record_id_ferc1's of bad training data records are: "
            f"{list(not_in_ppe.reset_index().record_id_ferc1)}"
        )
    train_df = train_df.drop(columns=ppe_cols + ["_merge"])
    return train_df


# @profile
def main():
    """Exercise all output methods for memory profiling."""
    pudl_settings = pudl.workspace.setup.get_defaults()
    pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
    pudl_out = PudlTabl(
        pudl_engine=pudl_engine,
        freq="AS",
        fill_fuel_cost=False,
        roll_fuel_cost=True,
        fill_net_gen=True,
    )
    rmi_out = Output(pudl_out)
    ppl = rmi_out.plant_parts_eia(clobber=True)
    del ppl
    for ppl_df in ["plant_parts_eia", "gens_mega_eia"]:
        if ppl_df in rmi_out.pudl_out._dfs:
            del rmi_out.pudl_out._dfs[ppl_df]

    deprish = rmi_out.deprish(clobber=True)
    del deprish

    deprish_to_eia = rmi_out.deprish_to_eia(clobber=True)
    del deprish_to_eia

    ferc1_to_eia = rmi_out.ferc1_to_eia(clobber=True)
    del ferc1_to_eia

    deprish_to_ferc1 = rmi_out.deprish_to_ferc1(clobber=True)
    del deprish_to_ferc1


if __name__ == "__main__":
    main()
