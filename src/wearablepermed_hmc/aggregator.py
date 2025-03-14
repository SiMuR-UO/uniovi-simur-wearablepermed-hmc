import argparse
import logging
import sys

from _WPM_file_management import load_scale_WPM_data
from _WPM_segmentation import segment_WPM_activity_data, plot_segmented_matrix_data


__author__ = "Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

_DEF_SEGMENT_BODY = 'Thigh'
_DEF_CALIBRATE_WITH_START_WALKING_USUAL_SPEED = 13261119

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
def scale(csv_matrix_PMP, segment_body, activity_PMP, calibrate_with_start_WALKING_USUAL_SPEED):
    scaled_data, dictionary_timing = load_scale_WPM_data(csv_matrix_PMP, segment_body, activity_PMP, calibrate_with_start_WALKING_USUAL_SPEED)

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
    segmented_activity_data = segment_WPM_activity_data(scaled_data, dictionary_timing)

    return segmented_activity_data

# Plot activity-by-activity segmented data from MATRIX.
def plot(segmented_activity_data, file_name):
    plot_segmented_matrix_data(segmented_activity_data, file_name)

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
        "-csv-matrix-PMP",
        "--csv-matrix-PMP",
        dest="csv_matrix_PMP",         
        help="string, path to the '.csv' file containing all data recorded by MATRIX.")
    parser.add_argument(
        "-segment-body",
        "--segment-body",
        default=_DEF_SEGMENT_BODY,
        dest="segment_body", 
        help="string, body segment where the IMU is placed ('Thigh', 'Wrist', or 'Hip')")
    parser.add_argument(
        "-activity-PMP",
        "--activity-PMP",
        dest="activity_PMP", 
        help="string, path to the corresponding Activity Log of the PMP dataset")                
    parser.add_argument(
        "-calibrate-with-start-WALKING-USUAL-SPEED",
        "--calibrate-with-start-WALKING-USUAL-SPEED",
        default=_DEF_CALIBRATE_WITH_START_WALKING_USUAL_SPEED,
        dest="calibrate_with_start_WALKING_USUAL_SPEED", 
        help="int. The sample, visually inspected, that corresponds to the start of the 'WALKING-USUAL SPEED' activity. If not specified, its default value is None")           
    parser.add_argument(
        "-file-name",
        "--file-name",
        dest="file_name", 
        help="identification PMP file name")  
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
    args = parse_args(args)
    setup_logging(args.loglevel)
    
    _logger.info("Aggregator starts here")

    _logger.debug("Starting Scale Data ...")
    scaled_data, dictionary_timing = scale(
        args.csv_matrix_PMP, 
        args.segment_body, 
        args.activity_PMP, 
        args.calibrate_with_start_WALKING_USUAL_SPEED)

    _logger.debug("Starting Segment Data ...")
    segmented_activity_data = segment(
        dictionary_timing,
        scaled_data)

    #_logger.debug("Starting Ploting Data ...")
    plot(
        segmented_activity_data, 
        args.file_name)

    _logger.info("Aggregator ends here")

def run():
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
