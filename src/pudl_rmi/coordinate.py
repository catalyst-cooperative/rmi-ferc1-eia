"""
Coordinate the acquisition and generation of RMI's interrelated outputs.

The outputs in this repo are depended on one another. See `README` for a
diagram of the realtions.
"""

import logging
import pandas as pd

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

    def get_plant_part_list(self, clobber=False):
        """
        Get the master unit list; generate it or get if from a file.

        If you generate the MUL, it will be saved at the file path given. The
        plant-parts list is generated via the pudl_out object.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not the output is already pickled. Default is
                False.
        """
        if not pudl_rmi.FILE_PATH_PLANT_PARTS_EIA.is_file() or clobber:
            logger.info(
                "Master unit list not found "
                f"{pudl_rmi.FILE_PATH_PLANT_PARTS_EIA} Generating a new master"
                " unit list. This should take ~10 minutes."
            )
            # actually make the master plant parts list
            plant_parts_eia = self.pudl_out.plant_parts_eia()
            # export
            plant_parts_eia.to_pickle(
                pudl_rmi.FILE_PATH_PLANT_PARTS_EIA, compression='gzip'
            )
        else:
            logger.info(
                "Reading the plant part list from "
                f"{pudl_rmi.FILE_PATH_PLANT_PARTS_EIA}"
            )
            plant_parts_eia = pd.read_pickle(
                pudl_rmi.FILE_PATH_PLANT_PARTS_EIA, compression='gzip'
            )
        return plant_parts_eia

    def get_deprish(self, clobber=False):
        """
        Generate of grab the cleaned deprecaition studies.

        Args:
            clobber (boolean): True if you want to regenerate the depreciation
                data whether or not the output is already pickled. Default is
                False.
        """
        if not pudl_rmi.FILE_PATH_DEPRISH.is_file() or clobber:
            logger.info("Generating new depreciation study output.")
            deprish_df = pudl_rmi.deprish.execute()
            deprish_df.to_pickle(pudl_rmi.FILE_PATH_DEPRISH)
        else:
            logger.info(
                "Grabbing depreciation study output from "
                f"{pudl_rmi.FILE_PATH_DEPRISH}"
            )
            deprish_df = pd.read_pickle(pudl_rmi.FILE_PATH_DEPRISH)
        return deprish_df

    def get_deprish_to_eia(
        self,
        clobber=False,
        clobber_ppl=False,
    ):
        """
        Generate or grab the connection between the depreciation data and EIA.

        Args:
            clobber (boolean): True if you want to regenerate the connection
                between the depreciation data and EIA whether or not the output
                is already pickled. Default is False.
            clobber_ppl (boolean): True if you want to regenerate the master
                unit list whether or not the output is already pickled. Default
                is False.
            clobber_d (boolean): True if you want to regenerate the
                depreciation data whether or not the output is already pickled.
                Default is False.
        """
        clobber_any = any([clobber, clobber_ppl])
        if not pudl_rmi.FILE_PATH_DEPRISH_EIA.is_file() or clobber_any:
            deprish_match_df = pudl_rmi.connect_deprish_to_eia.execute(
                plant_parts_df=self.get_plant_part_list(clobber=clobber_ppl),
            )
            deprish_match_df.to_pickle(pudl_rmi.FILE_PATH_DEPRISH_EIA)
        else:
            deprish_match_df = pd.read_pickle(pudl_rmi.FILE_PATH_DEPRISH_EIA)
        return deprish_match_df

    def get_ferc1_to_eia(self, clobber=False, clobber_ppl=False):
        """
        Generate or grab a connection between FERC1 and EIA.

        Either generate or grab an on-disk cached version of the connection
        between the FERC1 plant data and the EIA plant part list.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not it is saved at the file_path_mul
            clobber_ppl (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
        """
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any([clobber, clobber_ppl])
        if not pudl_rmi.FILE_PATH_FERC1_EIA.is_file() or clobber_any:
            logger.info(
                "FERC to EIA granular connection not found at "
                f"{pudl_rmi.FILE_PATH_FERC1_EIA}... Generating a new output."
            )
            connects_ferc1_eia = pudl_rmi.connect_ferc1_to_eia.execute(
                self.pudl_out,
                self.get_plant_part_list(clobber=clobber_ppl)
            )
            # export
            connects_ferc1_eia.to_pickle(pudl_rmi.FILE_PATH_FERC1_EIA)
        else:
            logger.info(
                "Reading the FERC1 to EIA connection from "
                f"{pudl_rmi.FILE_PATH_FERC1_EIA}"
            )
            connects_ferc1_eia = pd.read_pickle(pudl_rmi.FILE_PATH_FERC1_EIA)
        return connects_ferc1_eia

    def get_deprish_to_ferc1(
        self,
        clobber=False,
        clobber_ppl=False,
        clobber_de=False,
        clobber_fe=False
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
            clobber_ppl (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output. Default is False.
            clobber_de (boolean): Generate and cache a new interim output of
                the connection between EIA and depreciation data and generate
                a new version of the depreciaiton to FERC1 output. Default is
                False.
            clobber_fe (boolean): Generate and cache a new interim output of
                the connection between FERC and EIA and generate a new version
                of the depreciaiton to FERC1 output. Default is False.

        Returns:
            pandas.DataFrame: depreciation study data connected to FERC1 data
            which has been scaled down or aggregated to the level of reporting
            in the depreciaiton studies.
        """
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any([clobber, clobber_ppl, clobber_de, clobber_fe])
        if not pudl_rmi.FILE_PATH_DEPRISH_FERC1.is_file() or clobber_any:
            logger.info(
                "Deprish to FERC1 granular connection not found at "
                f"{pudl_rmi.FILE_PATH_DEPRISH_FERC1}. Generating a new output."
            )
            connects_deprish_ferc1 = pudl_rmi.connect_deprish_to_ferc1.execute(
                plant_parts_eia=self.get_plant_part_list(clobber=clobber_ppl),
                deprish_eia=self.get_deprish_to_eia(clobber=clobber_de),
                ferc1_to_eia=self.get_ferc1_to_eia(clobber=clobber_fe),
                clobber=clobber
            )
            # export
            connects_deprish_ferc1.to_pickle(pudl_rmi.FILE_PATH_DEPRISH_FERC1)

        else:
            logger.info(
                "Reading the depreciation to FERC1 connection from "
                f"{pudl_rmi.FILE_PATH_DEPRISH_FERC1}"
            )
            connects_deprish_ferc1 = pd.read_pickle(
                pudl_rmi.FILE_PATH_DEPRISH_FERC1
            )
        return connects_deprish_ferc1

    def get_all(self, clobber_all=False):
        """
        Gotta catch em all. Get all of the RMI outputs.

        Get or regenerate all of the RMI outputs.

        Args:
            clobber_all (boolean): Deafult is False, which will grab the
                outputs if they already exist, or generate them if they don't
                exist. True will re-generate the outputs whether they exist on
                disk. Re-generating everything will take ~15 minutes.
        """
        ppl = self.get_plant_part_list(clobber=clobber_all)
        d = self.get_deprish(clobber=clobber_all)
        de = self.get_deprish_to_eia(clobber=clobber_all)
        fe = self.get_ferc1_to_eia(clobber=clobber_all)
        df = self.get_deprish_to_ferc1(clobber=clobber_all)
        return ppl, d, de, fe, df
