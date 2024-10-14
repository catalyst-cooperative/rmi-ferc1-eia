"""Unit tests for connection between Depreciation and FERC1."""

import pandas as pd

from pudl_rmi.connect_deprish_to_ferc1 import allocate_cols


def test_allocate_col():
    """Test allocation of stacked depreication and EIA."""
    de_stacked_test = pd.DataFrame(
        {
            "record_id_eia": ["gen1", "gen2", "gen3"],
            "capacity_mw": [70, 20, 10],
            "line_id": ["test_1"] * 3,
            "plant_balance": [100] * 3,
        }
    )
    plant_balance_expected = de_stacked_test.copy()
    plant_balance_expected.loc[:, "plant_balance"] = [70.0, 20.0, 10.0]

    pd.testing.assert_frame_equal(
        allocate_cols(
            to_allocate=de_stacked_test,
            by=["line_id"],
            data_and_allocator_cols={"plant_balance": ["capacity_mw"]},
        ),
        plant_balance_expected,
    )
