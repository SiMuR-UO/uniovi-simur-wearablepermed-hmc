import argparse
import logging
import sys
import os

from data_import._bin2csv_0616 import bin2csv
from pathlib import Path

__author__ = "Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="BIN to CSV Converter")
    parser.add_argument(
        "-bin-matrix-PMP",
        "--bin-matrix-PMP",
        dest="bin_matrix_PMP",
        help="path to the '.bin' file containing all data recorded by MATRIX.")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def create_file_name_csv(csv_matrix_PMP):
    path_name = Path(csv_matrix_PMP)
    folder_name = os.path.dirname(csv_matrix_PMP)
    file_name = path_name.stem
    
    return os.path.join(folder_name, file_name + '.csv')

def converter(bin_matrix_PMP, csv_matrix_PMP):
    bin2csv(bin_matrix_PMP, csv_matrix_PMP)

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.debug("Starting converter ...")
    file_name_csv = create_file_name_csv(args.bin_matrix_PMP)
    _logger.debug("The file name is " + file_name_csv)
    converter(args.bin_matrix_PMP, file_name_csv)
    _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
