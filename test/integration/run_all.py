"""Big run. Run it all. Make all of the RMI outputs."""

import sqlalchemy as sa
import pytest

import pudl
import pudl_rmi


@pytest.fixture(scope="module")
def pudl_out():
    """Make annual PUDL output object."""
    # pudl output object
    pudl_engine = sa.create_engine(
        pudl.workspace.setup.get_defaults()["pudl_db"]
    )
    return pudl.output.pudltabl.PudlTabl(
        pudl_engine,
        freq='AS',
        fill_fuel_cost=True,
        roll_fuel_cost=True,
        fill_net_gen=True,
    )


@pytest.fixture(scope="module")
def rmi_out(pudl_out):
    """Make RMI output object."""
    return pudl_rmi.coordinate.Output(pudl_out)


def test_all(rmi_out):
    """Generate all of the RMI outputs."""
    ppl, d, de, fe, fde = rmi_out.grab_all(clobber_all=True)
