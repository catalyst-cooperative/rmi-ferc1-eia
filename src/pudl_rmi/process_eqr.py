"""Load EQR contracts and identities to a sqlite database."""
import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

import coloredlogs
import pandas as pd
import sqlalchemy as sa
from tqdm import tqdm

from pudl_rmi import EQR_DATA_DIR, EQR_DB_PATH

logger = logging.getLogger(__name__)

engine = sa.create_engine(f"sqlite:///{EQR_DB_PATH}")

FILE_END_STRS_TO_TABLE_NAMES = {
    "indexPub.CSV": "index_publishing",
    "ident.CSV": "identities",
    "contracts.CSV": "contracts",
}

TABLE_DTYPES = {
    "identities": {"contact_zip": "string", "contact_phone": "string"},
    "contracts": {
        "seller_history_name": "string",
    },
}

WORKING_PARTITIONS = {"years": [2020], "quarters": ["Q1", "Q2" "Q3", "Q4"]}


class FercEqrPartition(NamedTuple):
    """Represents FercEqr partition identifying unique resource file."""

    year: int
    quarter: str


def parse_command_line(argv):
    """Parse script command line arguments. See the -h option.

    Args:
        argv (list): command line arguments including caller file name.

    Returns:
        dict: A dictionary mapping command line arguments to their values.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        help="""Which years of FERC EQR data to process. Defaults to all years.""",
        default=WORKING_PARTITIONS["years"],
    )
    parser.add_argument(
        "-q",
        "--quarters",
        nargs="+",
        type=str.upper,
        help="""Which quarters to parocess. Defaults to all quarters.""",
        default=WORKING_PARTITIONS["quarters"],
    )
    parser.add_argument(
        "-c",
        "--clobber",
        action="store_true",
        default=False,
        help="Clobber existing PUDL SQLite and Parquet outputs if they exist.",
    )

    arguments = parser.parse_args(argv[1:])
    return arguments


def convert_to_stringdtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns with nans to StringDtypes.

    This is a temporary fix to make sure pandas doesn't infer
    the wrong datatype for a column. This will be removed once
    we identify the correct types for all of the columns.
    """
    for column in df.columns:
        if df[column].isnull().any():
            df[column] = df[column].astype("string")
    return df


def extract_seller(seller_zip: zipfile.ZipFile, partition) -> None:
    """
    Extract the tables and load them to a sqlite db for a seller.

    Args:
        seller_zip: A zipfile containing the tables for a single seller.
        partition: One quarter partition of EQR ferc data.
    """
    with seller_zip as seller:
        for table_type, table_name in FILE_END_STRS_TO_TABLE_NAMES.items():
            # find a file in seller_zip that matches the substring
            table_csv_path = list(
                filter(lambda x: x.endswith(table_type), seller.namelist())
            )
            assert len(table_csv_path) <= 1
            if table_csv_path:
                df = pd.read_csv(
                    io.BytesIO(seller_zip.read(table_csv_path[0])),
                    encoding="ISO-8859-1",
                    dtype=TABLE_DTYPES.get(table_name),
                    parse_dates=True,
                )
                df = convert_to_stringdtype(df)

                df["year"] = partition.year
                df["quarter"] = partition.quarter

                with engine.connect() as conn:
                    df.to_sql(table_name, conn, index=False, if_exists="append")


def extract_partition(partition: FercEqrPartition) -> None:
    """
    Extract a quarter of EQR data.

    Args:
        partition: One quarter partition of EQR ferc data.
    """
    quarter_zip_path = EQR_DATA_DIR / f"CSV_{partition.year}_{partition.quarter}.zip"
    if not quarter_zip_path.exists():
        raise FileNotFoundError(
            f"""
            Oops! It looks like that partition of data doesn't exist in {EQR_DATA_DIR}.
            Download the desired quarter from https://eqrreportviewer.ferc.gov/ to
            {EQR_DATA_DIR}.
            """
        )

    with zipfile.ZipFile(quarter_zip_path, mode="r") as quarter_zip:
        for seller_path in tqdm(quarter_zip.namelist()):
            seller_zip_bytes = io.BytesIO(quarter_zip.read(seller_path))
            seller_zip = zipfile.ZipFile(seller_zip_bytes)
            extract_seller(seller_zip, partition)


def main():
    """Load EQR contracts and identities to a sqlite database."""
    args = parse_command_line(sys.argv)
    eqr_logger = logging.getLogger("pudl_rmi.process_eqr")
    log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
    coloredlogs.install(fmt=log_format, level="INFO", logger=eqr_logger)

    if args.clobber:
        EQR_DB_PATH.unlink(missing_ok=True)
    else:
        if EQR_DB_PATH.exists():
            raise SystemExit(
                "The FERC EQR DB already exists, and we don't want to clobber it.\n"
                f"Move {EQR_DB_PATH} aside or set clobber=True and try again."
            )

    partitions = [
        FercEqrPartition(year, quarter)
        for year in args.years
        for quarter in args.quarters
    ]

    for partition in tqdm(partitions):
        logger.info(f"Processing {partition}")
        extract_partition(partition)


if __name__ == "__main__":
    main()
