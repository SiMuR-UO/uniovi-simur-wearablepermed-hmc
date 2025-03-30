import argparse
import logging
import sys
from pathlib import Path
import os
import numpy as np

from data_import._WPM_file_management import load_scale_WPM_data
from data_import._WPM_segmentation import segment_WPM_activity_data, plot_segmented_WPM_data, apply_windowing_WPM_segmented_data
from basic_functions._autocalibration import count_stuck_vals, get_calibration_coefs, auto_calibrate

__author__ = "Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

_DEF_CALIBRATE_WITH_START_WALKING_USUAL_SPEED = 13261119
_DEF_WINDOW_SIZE_SAMPLES = 250
_DEF_IMAGES_FOLDER = 'Images_activities'
_DEF_STACK_OF_DATA_EXPORTED = 'data_tot.npz'

_ACTIVITIES = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED',
               'CAMINAR ZIGZAG', 'DE PIE BARRIENDO', 'DE PIE DOBLANDO TOALLAS',
               'DE PIE MOVIENDO LIBROS', 'DE PIE USANDO PC', 'FASE REPOSO CON K5',
               'INCREMENTAL CICLOERGOMETRO', 'SENTADO LEYENDO', 'SENTADO USANDO PC',
               'SENTADO VIENDO LA TV', 'SIT TO STAND 30 s', 'SUBIR Y BAJAR ESCALERAS',
               'TAPIZ RODANTE', 'TROTAR', 'YOGA', 'ACTIVIDAD NO ESTRUCTURADA']


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
        "-window-size-samples",
        "--window-size-samples",
        default=_DEF_WINDOW_SIZE_SAMPLES,
        dest="window_size_samples", 
        help="Size of the windows generated during windowing.")         
    parser.add_argument(
        "-images-folder-name",
        "--images-folder-name",
        dest="images_folder_name", 
        default=_DEF_IMAGES_FOLDER,
        help="folder of the images created (activities segmented)")  
    parser.add_argument(
        "-export-folder-name",
        "--export-folder-name",
        dest="export_folder_name", 
        help="folder of the stack of data created.")  
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

# This function encapsulates the code to perform load and scaling of WPM data Segmentation is not applied in this function.
#
# - Input Parameters:
# * csv_file_PMP: string, path to the ".csv" file containing all data recorded by MATRIX.
# * segment_body: string, body segment where the IMU is placed ("Thigh", "Wrist", or "Hip").
# * excel_file_path: string, path to the corresponding Activity  Log of the PMP dataset.
# * calibrate_with_start_WALKING_USUAL_SPEED: int. The sample, visually inspected, that corresponds to the start of the "WALKING-USUAL SPEED" activity. If not specified, its default value is None.
#
# - Return Value:
def extract_metadata_from_csv(csv_matrix_PMP):
     folder_name_path = Path(csv_matrix_PMP)
     array_metadata = folder_name_path.stem.split('_')
     return array_metadata[0], array_metadata[1], array_metadata[2]
     
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
def plot(segmented_activity_data, images_folder_name, csv_matrix_PMP):
    plot_segmented_WPM_data(segmented_activity_data, images_folder_name, csv_matrix_PMP)

def autocalibrate(segmented_activity_data):
    datos_acc_actividad_no_estructurada = segmented_activity_data['ACTIVIDAD NO ESTRUCTURADA'][:,0:4]  # timestamps y datos de aceleraciónprint(datos_acc_actividad_no_estructurada)
    datos_acc_actividad_no_estructurada_autocalibrados_W1_PI, slope, offset = auto_calibrate(datos_acc_actividad_no_estructurada, fm = 25)
    for actividad in _ACTIVITIES:
        segmented_activity_data[actividad][:,1:4] = segmented_activity_data[actividad][:,1:4] * slope + offset # muslo
        
    return segmented_activity_data

def windowing(segmented_activity_data, window_size_samples):
    labels_thigh = []
    # Enventanar los datos para cada actividad del diccionario
    windowed_data = apply_windowing_WPM_segmented_data(segmented_activity_data, window_size_samples)
    labels_thigh.extend(segmented_activity_data.keys())  # Almacenar las etiquetas de la actividad
    
    return windowed_data, labels_thigh

def stack(windowed_data, segment_body, export_folder_name):
    if not os.path.isfile(export_folder_name):
        # Create the file
        with open(export_folder_name, "w") as file:
            file.write("")  # Creates an empty file
            _logger.debug("File did not exist, so it was created.")

    concatenated_data = []
    all_labels = []
    for activity, data in windowed_data.items():
        data_selected = data[:, 1:7, :]
        concatenated_data.append(data_selected)
        all_labels.extend([activity] * data_selected.shape[0])
        
    # Convertir la lista de arrays en un array final si no está vacío
    if concatenated_data:
        concatenated_data = np.vstack(concatenated_data)
    else:
        concatenated_data = np.array([])  # Array vacío si no hay datos
        
    return concatenated_data, all_labels

def export_data(concatenated_data, all_labels, export_folder_name):
    np.savez(export_folder_name, concatenated_data, all_labels)

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)
    
    _logger.info("Aggregator starts here")

    _logger.debug("Step 00: Extracting metadata ...")
    participant_id, measurement_date, segment_body = extract_metadata_from_csv(args.csv_matrix_PMP)
    
    _logger.debug("Step 01: Starting Scale Data ...")
    scaled_data, dictionary_timing = scale(
        args.csv_matrix_PMP,
        segment_body, 
        args.activity_PMP,
        args.calibrate_with_start_WALKING_USUAL_SPEED)

    _logger.debug("Step 02: Starting Segment Data ...")
    segmented_activity_data = segment(
        dictionary_timing,
        scaled_data)
    
    _logger.debug("Step 03: Starting Ploting Data ...")
    plot(segmented_activity_data, 
        args.images_folder_name,
        args.csv_matrix_PMP)

    _logger.debug("Step 04: Starting Autocalibrating Data ...")
    segmented_activity_data_autocalibrated = autocalibrate(segmented_activity_data)
    
    _logger.debug("Step 05: Starting Windowing Data ...")
    windowed_data, labels = windowing(segmented_activity_data_autocalibrated, args.window_size_samples)
    
    _logger.debug("Step 06: Starting Stacking Data ...")
    concatenated_data, all_labels = stack(windowed_data, segment_body, args.export_folder_name)
    
    _logger.debug("Step 07: Starting Exporting Data ...")
    export_data(concatenated_data, all_labels, args.export_folder_name)
    
    _logger.info("Aggregator ends here")

def run():
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
