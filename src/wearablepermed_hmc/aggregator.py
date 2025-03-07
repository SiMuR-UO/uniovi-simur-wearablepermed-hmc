import argparse
import logging
import sys

from _WPM_file_management import load_scale_WPM_data, plot_segmented_matrix_data, plot_segmented_matrix_data

__author__ = "Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# This function encapsulates the code to perform load and scaling of WPM data Segmentation is not applied in this function.
#
# - Input Parameters:
# * csv_file_PMP: string, path to the ".csv" file containing all data recorded by MATRIX.
# * segment_body: string, body segment where the IMU is placed ("Thigh", "Wrist", or "Hip").
# * excel_file_path: string, path to the corresponding Activity  Log of the PMP dataset.
# * calibrate_with_start_WALKING_USUAL_SPEED: int. The sample, visually inspected, that corresponds to the start of the "WALKING-USUAL SPEED" activity. If not specified, its default value is None.
#
# - Return Value:
# Returns WPM data properly scaled and the corresponding dictionary timing from the Excel file.
def scale(csv_file_PMP, segment_body, excel_file_path, calibrate_with_start_WALKING_USUAL_SPEED):
    scaled_data, dictionary_timing = load_scale_WPM_data(csv_file_PMP, segment_body, excel_file_path, calibrate_with_start_WALKING_USUAL_SPEED=None)

    return scaled_data, dictionary_timing

# Segments activity data based on defined time periods for various activities.
#
# Parameters:
# dictionary_hours_wpm (dict): Dictionary containing time data for various activities.
# imu_data (numpy.ndarray): Array containing the IMU data to be segmented.
#
# Returns:
# dict: A dictionary containing segmented data for each activity.
def segment(scaled_data, dictionary_timing):
    segmented_activity_data = plot_segmented_matrix_data(scaled_data, dictionary_timing)

    return segmented_activity_data

# Plot activity-by-activity segmented data from MATRIX.
def plot(dictionary_timing, file_name):
    segment_WPM_activity_data(dictionary_timing, file_name)

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
        "-csv-file-PMP",
        "--csv-file-PMP",
        dest="csv_file_PMP", 
        help="string, path to the '.csv' file containing all data recorded by MATRIX.")
    parser.add_argument(
        "-segment-body",
        "--segment-body",
        dest="segment_body", 
        help="string, body segment where the IMU is placed ('Thigh', 'Wrist', or 'Hip')")
    parser.add_argument(
        "-segment-body",
        "--segment-body",
        dest="segment_body", 
        help="string, body segment where the IMU is placed ('Thigh', 'Wrist', or 'Hip')")
    parser.add_argument(
        "-excel-file-path",
        "--excel-file-path",
        dest="excel_file_path", 
        help="string, path to the corresponding Activity Log of the PMP dataset")                
    parser.add_argument(
        "-calibrate-with-start-WALKING-USUAL-SPEED",
        "--calibrate-with-start-WALKING-USUAL-SPEED",
        dest="calibrate_with_start_WALKING_USUAL_SPEED", 
        help="int. The sample, visually inspected, that corresponds to the start of the 'WALKING-USUAL SPEED' activity. If not specified, its default value is None")           
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
    
    _logger.info("Script starts here")

    _logger.debug("Starting Scale Data...")
    scaled_data, dictionary_timing = scale(args.csv_file_PMP, args.segment_body, args.excel_file_path, args.calibrate_with_start_WALKING_USUAL_SPEED)

    _logger.debug("Starting Segment Data...")
    segmented_activity_data = segmented_activity_data = segment(scaled_data, dictionary_timing)

    _logger.debug("Ploting Data...")
    plot(segmented_activity_data, file_name)

    #plt.show()

    _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
