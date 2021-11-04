"""
Coordinate the acquisition and generation of RMI's interrelated outputs.

The outputs in this repo are dependent on one another. See `README` for a
diagram of the relations.
"""

import logging
import pandas as pd
from pathlib import Path

import pudl_rmi

logger = logging.getLogger(__name__)


class Output():
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
                should all be True.
        """
        self.pudl_out = pudl_out
        if pudl_out.freq != 'AS':
            raise AssertionError(
                f"Frequency of `pudl_out` must be `AS` but got {pudl_out.freq}"
            )

    def grab_plant_part_list(self, clobber=False):
        """
        Get the master unit list; generate it or get if from a file.

        If you generate the MUL, it will be saved at the file path given. The
        plant-parts list is generated via the pudl_out object.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not the output is already pickled. Default is
                False.
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
            # export
            plant_parts_eia.to_pickle(file_path)
        else:
            logger.info(f"Reading the plant part list from {file_path}")
            plant_parts_eia = pd.read_pickle(file_path)
        return plant_parts_eia

    def grab_deprish(self, clobber=False):
        """
        Generate or grab the cleaned deprecaition studies.

        Args:
            clobber (boolean): True if you want to regenerate the depreciation
                data whether or not the output is already pickled. Default is
                False.
        """
        file_path = pudl_rmi.DEPRISH_PKL
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber:
            logger.info("Generating new depreciation study output.")
            deprish_df = pudl_rmi.deprish.execute()
            deprish_df.to_pickle(file_path)
        else:
            logger.info(f"Grabbing depreciation study output from {file_path}")
            deprish_df = pd.read_pickle(file_path)
        return deprish_df

    def grab_deprish_to_eia(
        self,
        clobber=False,
        clobber_plant_part_list=False,
    ):
        """
        Generate or grab the connection between the depreciation data and EIA.

        Args:
            clobber (boolean): True if you want to regenerate the connection
                between the depreciation data and EIA whether or not the output
                is already pickled. Default is False.
            clobber_plant_part_list (boolean): True if you want to regenerate
                the masterunit list whether or not the output is already
                pickled. Default is False.
        """
        file_path = pudl_rmi.DEPRISH_EIA_PKL
        clobber_any = any([clobber, clobber_plant_part_list])
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber_any:
            deprish_match_df = pudl_rmi.connect_deprish_to_eia.execute(
                plant_parts_df=self.grab_plant_part_list(
                    clobber=clobber_plant_part_list)
            )
            deprish_match_df.to_pickle(file_path)
        else:
            deprish_match_df = pd.read_pickle(file_path)
        return deprish_match_df

    def grab_ferc1_to_eia(self, clobber=False, clobber_plant_part_list=False):
        """
        Generate or grab a connection between FERC1 and EIA.

        Either generate or grab an on-disk cached version of the connection
        between the FERC1 plant data and the EIA plant part list.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not it is saved at the file_path_mul
            clobber_plant_part_list (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
        """
        file_path = pudl_rmi.FERC1_EIA_PKL
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any([clobber, clobber_plant_part_list])
        check_is_file_or_not_exists(file_path)
        if not file_path.exists() or clobber_any:
            logger.info(
                f"FERC to EIA granular connection not found at {file_path}... "
                "Generating a new output."
            )
            connects_ferc1_eia = pudl_rmi.connect_ferc1_to_eia.execute(
                self.pudl_out,
                self.grab_plant_part_list(clobber=clobber_plant_part_list)
            )
            # export
            connects_ferc1_eia.to_pickle(file_path)
        else:
            logger.info(f"Reading the FERC to EIA connection from {file_path}")
            connects_ferc1_eia = pd.read_pickle(file_path)
        return connects_ferc1_eia

    def grab_deprish_to_ferc1(
        self,
        clobber=False,
        clobber_plant_part_list=False,
        clobber_deprish_eia=False,
        clobber_ferc1_eia=False
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
            clobber_plant_part_list (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
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
        clobber_any = any([
            clobber,
            clobber_plant_part_list,
            clobber_deprish_eia,
            clobber_ferc1_eia
        ])
        check_is_file_or_not_exists(file_path)

        if not file_path.exists() or clobber_any:
            logger.info(
                "Deprish to FERC1 granular connection not found at "
                f"{file_path}. Generating a new output."
            )
            connects_deprish_ferc1 = pudl_rmi.connect_deprish_to_ferc1.execute(
                plant_parts_eia=self.grab_plant_part_list(
                    clobber=clobber_plant_part_list),
                deprish_eia=self.grab_deprish_to_eia(
                    clobber=clobber_deprish_eia),
                ferc1_to_eia=self.grab_ferc1_to_eia(
                    clobber=clobber_ferc1_eia),
                clobber=clobber
            )
            # export
            connects_deprish_ferc1.to_pickle(pudl_rmi.file_path)

        else:
            logger.info(
                f"Reading the depreciation to FERC connection from {file_path}"
            )
            connects_deprish_ferc1 = pd.read_pickle(file_path)
        return connects_deprish_ferc1

    def grab_all(self, clobber_all=False):
        """
        Gotta catch em all. Get all of the RMI outputs.

        Get or regenerate all of the RMI outputs. This method is mostly for
        testing purposes because it returns all 5 outputs. To grab individual
        outputs, it is recommended to use the output-specific method.

        Args:
            clobber_all (boolean): Deafult is False, which will grab the
                outputs if they already exist, or generate them if they don't
                exist. True will re-generate the outputs whether they exist on
                disk. Re-generating everything will take ~15 minutes.

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
        ppl = self.grab_plant_part_list(clobber=False)
        d = self.grab_deprish(clobber=clobber_all)
        de = self.grab_deprish_to_eia(clobber=clobber_all)
        fe = self.grab_ferc1_to_eia(clobber=clobber_all)
        df = self.grab_deprish_to_ferc1(clobber=clobber_all)
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