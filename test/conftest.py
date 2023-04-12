"""PyTest configuration module. Defines useful fixtures, command line args."""
import json
import logging
import os
from pathlib import Path
from typing import Any

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
def pudl_env(pudl_input_dir: dict[Any, Any]) -> None:
    """Set PUDL_OUTPUT/PUDL_INPUT/DAGSTER_HOME environment variables."""
    pudl.workspace.setup.get_defaults(**pudl_input_dir)

    logger.info(f"PUDL_OUTPUT path: {os.environ['PUDL_OUTPUT']}")
    logger.info(f"PUDL_INPUT path: {os.environ['PUDL_INPUT']}")


@pytest.fixture(scope="session")
def pudl_input_dir() -> dict[Any, Any]:
    """Determine where the PUDL input/output dirs should be."""
    input_override = None

    # In CI we want a hard-coded path for input caching purposes:
    if os.environ.get("GITHUB_ACTIONS", False):
        # hard-code input dir for CI caching
        input_override = Path(os.environ["HOME"]) / "pudl-work/data"

    return {"input_dir": input_override}


@pytest.fixture(scope="session", name="pudl_settings_fixture")
def pudl_settings_dict(request, pudl_input_dir):  # type: ignore
    """Determine some settings (mostly paths) for the test session."""
    logger.info("setting up the pudl_settings_fixture")
    pudl_settings = pudl.workspace.setup.get_defaults(**pudl_input_dir)
    pudl.workspace.setup.init(pudl_settings)

    pretty_settings = json.dumps(
        {str(k): str(v) for k, v in pudl_settings.items()}, indent=2
    )
    logger.info(f"pudl_settings being used: {pretty_settings}")
    return pudl_settings


@pytest.fixture(scope="session", name="pudl_engine")
def pudl_engine(pudl_settings_fixture: dict[Any, Any]) -> sa.engine.Engine:
    """Grab a connection to the PUDL Database.

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


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests."""
    return Path(__file__).parent
