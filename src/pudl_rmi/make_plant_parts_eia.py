"""
EIA Plant-parts list is actively being moved into PUDL.

Here are some dangly bits that we are actively using to read the plant-parts
list from PUDL into the other realms of this repo.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


############################
# Keep in rmi repo functions
############################


def reassign_id_ownership_dupes(plant_parts_df):
    """
    Reassign the record_id for the records that are labeled ownership_dupe.

    This function is used after the EIA plant-parts table is created.

    Args:
        plant_parts_df (pandas.DataFrame): master unit list. Result of
            ``generate_master_unit_list()`` or ``get_master_unit_list_eia()``.
            Must have boolean column ``ownership_dupe`` and string column or
            index of ``record_id_eia``.

    """
    # the record_id_eia's need to be a column to mess with it and record_id_eia
    # is typically the index of plant_parts_df, so we are going to reset index
    # if record_id_eia is the index
    og_index = False
    if plant_parts_df.index.name == "record_id_eia":
        plant_parts_df = plant_parts_df.reset_index()
        og_index = True
    # reassign the record id and ownership_record_type col when the record is a dupe
    plant_parts_df = plant_parts_df.assign(
        record_id_eia=lambda x: np.where(
            x.ownership_dupe,
            x.record_id_eia.str.replace("owned", "total"),
            x.record_id_eia,
        )
    )
    if "ownership_record_type" in plant_parts_df.columns:
        plant_parts_df = plant_parts_df.assign(
            ownership_record_type=lambda x: np.where(
                x.ownership_dupe, "total", x.ownership_record_type
            )
        )
    # then we reset the index so we return the dataframe in the same structure
    # as we got it.
    if og_index:
        plant_parts_df = plant_parts_df.set_index("record_id_eia")
    return plant_parts_df
