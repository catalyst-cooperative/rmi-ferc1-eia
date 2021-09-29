"""
Frickin ey.

We have to do stuff.

"""

import logging
import pandas as pd
# import numpy as np

from pudl_rmi.connect_deprish_to_eia import generate_depreciation_matches
from pudl_rmi.connect_ferc1_to_eia import ferc1_to_eia
import pudl_rmi.connect_deprish_to_ferc1

logger = logging.getLogger(__name__)


class Output():
    """Class to manage all of the interconnected RMI outputs."""

    def __init__(
        self,
        pudl_out,
        file_path_mul,
        file_path_deprish,
        file_path_deprish_eia,
        file_path_training,
        file_path_ferc1_eia,
        file_path_deprish_ferc1,
    ):
        """
        Initialize output coordinator for the RMI ouptus.

        This object enables short term disk caching and an easy way to
        regenerate any or all of the interconnected RMI outputs. Each `get_`
        method in this object grabs a pickle file off disk if it is available,
        or generates the output if it is not available and/or if you set
        clobber to True.

        Because some outputs rely on others, this object enables clobbering of
        both the main output and the outputs which the main object relies on.
        This in effect enables a clobber and a deep clobber.

        TODO: Deal with all of these fickin paths. I can imagine using a little
        settings dictionary. These files should always be in the inputs or
        outputs directories, so this should be easy enough to manage.

        Args:
            file_path_ferc1_eia (pathlib.Path): file path to the connection
                between FERC1 and EIA. If the file already exists, it will be
                read and returned. If it does not exist (or clobber is set to
                True), the output will be created and saved at this path.
        """
        self.pudl_out = pudl_out
        self.file_path_mul = file_path_mul
        self.file_path_deprish = file_path_deprish
        self.file_path_deprish_eia = file_path_deprish_eia
        self.file_path_training = file_path_training
        self.file_path_ferc1_eia = file_path_ferc1_eia
        self.file_path_deprish_ferc1 = file_path_deprish_ferc1
        # ensure the files are pickle files (except file_path_deprish)
        non_pickles = [
            p for p in [
                file_path_mul,
                file_path_deprish_eia,
                file_path_ferc1_eia,
                file_path_deprish_ferc1]
            if '.pkl' not in p.suffixes
        ]
        if non_pickles:
            raise AssertionError(f"{non_pickles} must be a pickle file")

    def get_plant_part_list(self, clobber=False):
        """
        Get the master unit list; generate it or get if from a file.

        If you generate the MUL, it will be saved at the file path given.

        Args:
            file_path_mul (pathlib.Path): where you want the master unit list
                to live. Must be a compressed pickle file ('.pkl.gz').
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not it is saved at the file_path_mul
        """
        if not self.file_path_mul.is_file() or clobber:
            logger.info(
                f"Master unit list not found {self.file_path_mul} "
                "Generating a new master unit list. This should take ~10 "
                "minutes."
            )
            # actually make the master plant parts list
            plant_parts_eia = self.pudl_out.plant_parts_eia()
            # export
            plant_parts_eia.to_pickle(
                self.file_path_mul, compression='gzip'
            )
        elif self.file_path_mul.is_file() or not clobber:
            logger.info(
                f"Reading the master unit list from {self.file_path_mul}")
            plant_parts_eia = pd.read_pickle(
                self.file_path_mul, compression='gzip')
        return plant_parts_eia

    def get_deprish_to_eia(self, clobber=False, clobber_ppl=False):
        """
        Generate or grab the connection between the depreciation data and EIA.

        Args:
            clobber (boolean):
        """
        if not self.file_path_deprish_eia.is_file() or clobber:
            deprish_match_df = (
                generate_depreciation_matches(
                    plant_parts_df=self.get_plant_part_list(
                        clobber=clobber_ppl),
                    file_path_deprish=self.file_path_deprish,
                )
            )
            deprish_match_df.to_pickle(self.file_path_deprish_eia)
        elif self.file_path_deprish_eia.is_file() or not clobber:
            deprish_match_df = pd.read_pickle(
                self.file_path_deprish_eia)
        return deprish_match_df

    def get_ferc1_to_eia(self, clobber=False, clobber_ppl=False):
        """
        Generate or grab a connection between FERC1 and EIA.

        Either generate or grab an on-disk cached version of the connection
        between the FERC1 plant data and the EIA plant part list.

        Args:
            clobber (boolean): True if you want to regenerate the master unit
                list whether or not it is saved at the file_path_mul
        """
        if not self.file_path_ferc1_eia.is_file() or clobber:
            logger.info(
                "FERC<>EIA granular connection not found at "
                f"{self.file_path_ferc1_eia}... Generating a new output."
            )
            connects_ferc1_eia = ferc1_to_eia(
                self.file_path_training,
                self.pudl_out,
                self.get_plant_part_list(clobber=clobber_ppl)
            )
            # export
            connects_ferc1_eia.to_pickle(
                self.file_path_ferc1_eia, compression='gzip')

        else:
            logger.info(
                "Reading the FERC1 to EIA connection from "
                f"{self.file_path_ferc1_eia}"
            )
            connects_ferc1_eia = pd.read_pickle(
                self.file_path_ferc1_eia, compression='gzip')
        return connects_ferc1_eia

    def get_deprish_to_ferc1(
        self,
        clobber=False,
        clobber_plant_part=False,
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
            clobber_plant_part (boolean): Generate and cache a new interim
                output of the EIA plant part list and generate a new version of
                the depreciaiton to FERC1 output.
            clobber_de (boolean):
            clobber_fe (boolean):

        Returns:
            pandas.DataFrame: depreciation study data connected to FERC1 data
            which has been scaled down or aggregated to the level of reporting
            in the depreciaiton studies.
        """
        # if any of the clobbers are on, we want to regenerate the main output
        clobber_any = any(
            [clobber, clobber_plant_part, clobber_de, clobber_fe])
        if not self.file_path_deprish_ferc1.is_file() or clobber_any:
            logger.info(
                "Deprish to FERC1 granular connection not found at "
                f"{self.file_path_ferc1_eia} ... Generating a new output."
            )
            connects_deprish_ferc1 = pudl_rmi.connect_deprish_to_eia.main(
                plant_parts_eia=self.get_plant_part_list(
                    clobber=clobber_plant_part),
                deprish_eia=self.get_deprish_to_eia(clobber=clobber_de),
                ferc1_to_eia=self.get_ferc1_to_eia(clobber=clobber_fe),
                clobber=clobber
            )
            # export
            connects_deprish_ferc1.to_pickle(
                self.file_path_deprish_ferc1, compression='gzip')

        else:
            logger.info(
                "Reading the depreciation to FERC1 connection from "
                f"{self.file_path_deprish_eia}"
            )
            connects_deprish_ferc1 = pd.read_pickle(
                self.file_path_deprish_ferc1, compression='gzip')
        return connects_deprish_ferc1

    def get_all(self, clobber_all=False):
        """Gotta catch em all. Get all of the RMI outputs."""
        ppl = self.get_plant_part_list(clobber=clobber_all)
        de = self.get_deprish_to_eia(clobber=clobber_all)
        fe = self.get_ferc1_to_eia(clobber=clobber_all)
        df = self.get_deprish_to_ferc1(clobber=clobber_all)
        return ppl, de, fe, df
