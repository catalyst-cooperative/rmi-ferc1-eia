"""PyTest configuration module. Defines useful fixtures, command line args."""
import logging
import os
from pathlib import Path

import pudl
import pytest
import sqlalchemy as sa
from pudl.output.pudltabl import PudlTabl

import pudl_rmi

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add a command line option Requiring fresh data download."""
    parser.addoption(
        "--cached-plant-parts-eia",
        action="store_true",
        default=False,
        help="Use the pickled EIA Plant Parts List.",
    )
    parser.addoption(
        "--cached-plant-parts-eia-distinct",
        action="store_true",
        default=False,
        help="Use the pickled distinct EIA Plant Parts List.",
    )
    parser.addoption(
        "--cached-deprish",
        action="store_true",
        default=False,
        help="Use the pickled Depreciation data.",
    )
    parser.addoption(
        "--cached-deprish-eia",
        action="store_true",
        default=False,
        help="Use the pickled Depreciation to EIA connection.",
    )
    parser.addoption(
        "--cached-ferc1-eia",
        action="store_true",
        default=False,
        help="Use the pickled FERC 1 to EIA connection",
    )
    parser.addoption(
        "--five-year-coverage",
        action="store_true",
        default=False,
        help="Use data only from 2015-2020 for all datasets to \
            save time and memory while testing.",
    )


@pytest.fixture(scope="session")
def pudl_settings_fixture():  # noqa: C901
    """
    Determine some settings for the test session.

    * On a user machine, it should use their existing PUDL_DIR.
    * In CI, it should use PUDL_DIR=$HOME/pudl-work containing the
      downloaded PUDL DB.

    """
    logger.info("setting up the pudl_settings_fixture")

    # In CI we want a hard-coded path for input caching purposes:
    if os.environ.get("GITHUB_ACTIONS", False):
        pudl_out = Path(os.environ["HOME"]) / "pudl-work"
        pudl_in = pudl_out
    # Otherwise, default to the user's existing datastore:
    else:
        try:
            defaults = pudl.workspace.setup.get_defaults()
        except FileNotFoundError as err:
            logger.critical("Could not identify PUDL_IN / PUDL_OUT.")
            raise err
        pudl_out = defaults["pudl_out"]
        pudl_in = defaults["pudl_in"]

    # Set these environment variables for future reference...
    logger.info("Using PUDL_IN=%s", pudl_in)
    os.environ["PUDL_IN"] = str(pudl_in)
    logger.info("Using PUDL_OUT=%s", pudl_out)
    os.environ["PUDL_OUT"] = str(pudl_out)

    # Build all the pudl_settings paths:
    pudl_settings = pudl.workspace.setup.derive_paths(
        pudl_in=pudl_in, pudl_out=pudl_out
    )
    # Set up the pudl workspace:
    pudl.workspace.setup.init(pudl_in=pudl_in, pudl_out=pudl_out)

    logger.info("pudl_settings being used: %s", pudl_settings)
    return pudl_settings


@pytest.fixture(scope="session")
def pudl_engine(pudl_settings_fixture):
    """
    Grab a connection to the PUDL Database.

    If we are using the test database, we initialize the PUDL DB from scratch.
    If we're using the live database, then we just make a conneciton to it.
    """
    logger.info("setting up the pudl_engine fixture")
    engine = sa.create_engine(pudl_settings_fixture["pudl_db"])
    logger.info("PUDL Engine: %s", engine)
    return engine


@pytest.fixture(scope="session")
def pudl_out(pudl_engine, request):
    """Make an annual PUDL output object with all filling enabled."""
    if request.config.getoption("--five-year-coverage"):
        start_date = "2015-01-01"
        end_date = "2020-12-31"
    else:
        start_date = None
        end_date = None
    return PudlTabl(
        pudl_engine=pudl_engine,
        freq="AS",
        fill_fuel_cost=False,
        roll_fuel_cost=True,
        fill_net_gen=True,
        start_date=start_date,
        end_date=end_date,
    )


@pytest.fixture(scope="module")
def rmi_out(pudl_out):
    """Make RMI output object."""
    return pudl_rmi.coordinate.Output(pudl_out)
