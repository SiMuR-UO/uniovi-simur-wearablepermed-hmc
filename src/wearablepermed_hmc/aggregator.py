import os
import argparse
import logging
import sys
from enum import Enum

import numpy as np

__author__ = "Miguel Salinas <uo34525@uniovi.es>, Alejandro <uo265351@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

class ML_Sensor(Enum):
    PI = 'thigh'
    M = 'wrist'
    C = 'hip'

def parse_ml_model(value):
    try:
        """Parse a comma-separated list of CML Models lor values into a list of ML_Sensor enums."""
        values = [v.strip() for v in value.split(',') if v.strip()]
        result = []
        invalid = []
        for v in values:
            try:
                result.append(ML_Model(v))
            except ValueError:
                invalid.append(v)
        if invalid:
            valid = ', '.join(c.value for c in ML_Model)
            raise argparse.ArgumentTypeError(
                f"Invalid color(s): {', '.join(invalid)}. "
                f"Choose from: {valid}"
            )
        return result
    except ValueError:
        valid = ', '.join(ml_model.value for ml_model in ML_Model)
        raise argparse.ArgumentTypeError(f"Invalid ML Model '{value}'. Choose from: {valid}")
    
def parse_ml_sensor(value):
    try:
        """Parse a comma-separated list of CML Models lor values into a list of ML_Sensor enums."""
        values = [v.strip() for v in value.split(',') if v.strip()]
        result = []
        invalid = []
        for v in values:
            try:
                result.append(ML_Sensor(v))
            except ValueError:
                invalid.append(v)
        if invalid:
            valid = ', '.join(c.value for c in ML_Sensor)
            raise argparse.ArgumentTypeError(
                f"Invalid color(s): {', '.join(invalid)}. "
                f"Choose from: {valid}"
            )
        return result
    except ValueError:
        valid = ', '.join(ml_model.value for ml_model in ML_Sensor)
        raise argparse.ArgumentTypeError(f"Invalid ML Model '{value}'. Choose from: {valid}")
    
def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Machine Learning Model Trainer")
    parser.add_argument(
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Case unique identifier."
    )    
    parser.add_argument(
        "-ml-models",
        "--ml-models",
        type=parse_ml_model,
        nargs='+',
        dest="ml_models",        
        required=True,
        help=f"Available ML models: {[c.value for c in ML_Model]}."
    )
    parser.add_argument(
        "-ml-sensors",
        "--ml-sensors",
        type=parse_ml_sensor,
        nargs='+',
        dest="ml_sensors",        
        required=True,
        help=f"Available ML sensors: {[c.value for c in ML_Sensor]}."
    )
    parser.add_argument(
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Choose the dataset root folder."
    )
    parser.add_argument(
        "-participants-file",
        "--participants-file",
        type=argparse.FileType("r"),
        required=True,
        help="Choose the dataset participant text file"
    )
    parser.add_argument(
        "-case-id-folder",
        "--case-id-folder",
        dest="case_id_folder",
        required=True,
        help="Choose the case id root folder."
    )      
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG.",
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

def convolution_model_selected(models):
    for model in models:
        if model.value in [ML_Model.CAPTURE24.value, ML_Model.ESANN.value]:
            return True
        
    return False

def feature_model_selected(models):
    for model in models:
        if model.value in [ML_Model.RANDOM_FOREST.value, ML_Model.XGBOOST.value]:
            return True
        
    return False

def combine_participant_dataset(dataset_folder, participant, models, sensors):
    participant_folder = os.path.join(dataset_folder, participant)
    participant_files = [
        f for f in os.listdir(participant_folder) 
        if os.path.isfile(os.path.join(participant_folder, f)) and
        ".npz" in f and
        any(sensor in f for sensor in ['_' + item for item in [sensor.name for sensor in sensors]])
    ]

    participant_dataset = []
    participant_label_dataset = []
    
    participant_feature_dataset = []
    participant_feature_label_dataset = []
    
    # aggregate datasets
    for participant_file in participant_files:
        # aggregate not feature datasets: wrist and thing 
        if "features" not in participant_file and convolution_model_selected(models):
            participant_sensor_file = os.path.join(participant_folder, participant_file)
            participant_sensor_dataset = np.load(participant_sensor_file)
            
            participant_dataset.append(participant_sensor_dataset[WINDOW_CONCATENATED_DATA])
            participant_label_dataset.append(participant_sensor_dataset[WINDOW_ALL_LABELS])
            
         # aggregate feature datasets: wrist and thing
        if "features" in participant_file and feature_model_selected(models):
            participant_sensor_feature_file = os.path.join(participant_folder, participant_file)
            participant_sensor_feature_dataset = np.load(participant_sensor_feature_file)
            
            participant_feature_dataset.append(participant_sensor_feature_dataset[WINDOW_CONCATENATED_DATA])
            participant_feature_label_dataset.append(participant_sensor_feature_dataset[WINDOW_ALL_LABELS])

    if len(participant_dataset) > 0:
        participant_dataset = np.concatenate(participant_dataset, axis=0)
        participant_label_dataset = np.concatenate(participant_label_dataset, axis=0)
    
        participant_sensor_all_file = os.path.join(participant_folder, 'data_' + participant + "_all.npz")
        np.savez(participant_sensor_all_file, participant_dataset, participant_label_dataset)
    
    if len(participant_feature_dataset) > 0:
        participant_feature_dataset = np.concatenate(participant_feature_dataset, axis=0)
        participant_feature_label_dataset = np.concatenate(participant_feature_label_dataset, axis=0)
                
        participant_sensor_feature_all_file = os.path.join(participant_folder, 'data_' + participant + "_features_all.npz")
        np.savez(participant_sensor_feature_all_file, participant_feature_dataset, participant_feature_label_dataset) 
                
def combine_datasets(case_id_folder, dataset_folder, participants, sensors):
    dataset = []
    dataset_label = []
    dataset_feature = []
    dataset_feature_label = []

    for participant in participants:
        participant_folder = os.path.join(dataset_folder, participant)

        if len([sensor.name for sensor in sensors]) == 1:
            participant_files = [
                f for f in os.listdir(participant_folder) 
                if os.path.isfile(os.path.join(participant_folder, f)) and 
                "_" + [sensor.name for sensor in sensors][0] + ".npz" in f
            ]
        else:
            participant_files = [
                f for f in os.listdir(participant_folder) 
                if os.path.isfile(os.path.join(participant_folder, f)) and 
                "_all.npz" in f
            ]

        # aggregate datasets
        for participant_file in participant_files:
            # aggregate not feature datasets: wrist and thing 
            if "features" not in participant_file:
                participant_sensor_file = os.path.join(participant_folder, participant_file)
                participant_sensor_dataset = np.load(participant_sensor_file)
                
                dataset.append(participant_sensor_dataset[WINDOW_CONCATENATED_DATA])
                dataset_label.append(participant_sensor_dataset[WINDOW_ALL_LABELS])
                
            # aggregate feature datasets: wrist and thing
            if "features" in participant_file:
                participant_sensor_feature_file = os.path.join(participant_folder, participant_file)
                participant_sensor_feature_dataset = np.load(participant_sensor_feature_file)
                
                dataset_feature.append(participant_sensor_feature_dataset[WINDOW_CONCATENATED_DATA])
                dataset_feature_label.append(participant_sensor_feature_dataset[WINDOW_ALL_LABELS])

    if len(dataset) > 0:
        dataset = np.concatenate(dataset, axis=0)
        dataset_label = np.concatenate(dataset_label, axis=0)
    
        dataset_all_file = os.path.join(case_id_folder, "data_all.npz")
        np.savez(dataset_all_file, dataset, dataset_label)
    
    if len(dataset_feature) > 0:
        dataset_feature = np.concatenate(dataset_feature, axis=0)
        dataset_feature_label = np.concatenate(dataset_feature_label, axis=0)
                
        dataset_feature_all_file = os.path.join(case_id_folder, "data_feature_all.npz")
        np.savez(dataset_feature_all_file, dataset_feature, dataset_feature_label)                              

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

    _logger.info("Agregator starts here")

    # create the output case id folder if not exist
    case_id_folder = os.path.join(args.case_id_folder, args.case_id)
    os.makedirs(case_id_folder, exist_ok=True)

    participants = []
    for line in args.participants_file:
        participants = participants + line.strip().split(',')

    # Participant datasets agregation
    if len(args.ml_sensors[0]) > 1:
        for participant in participants:
            combine_participant_dataset(args.dataset_folder, participant, args.ml_models[0], args.ml_sensors[0])
    
    # Total datasets agregation
    combine_datasets(case_id_folder, args.dataset_folder, participants, args.ml_sensors[0])

    _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
