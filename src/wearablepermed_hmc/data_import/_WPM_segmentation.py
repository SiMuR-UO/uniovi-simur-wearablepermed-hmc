import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl                                        # Librería para leer datos desde una Hoja Excel
from datetime import time, timedelta, date, datetime
import matplotlib.pyplot as plt
import os
from scipy.signal import resample

import openpyxl
from datetime import time, timedelta, date

from pathlib import Path

######## Segmentations Functions ########

def find_closest_timestamp(arr, target_timestamp):
    """Find the index of the value in `arr` closest to `target_timestamp` using binary search.
    
    Args:
        arr (np.array): Array of timestamps.
        target_timestamp (float): The target timestamp to find the closest value to.
    
    Returns:
        int: Index of the closest timestamp or -1 if the array is empty.
    """
    if len(arr) == 0:
        return -1

    # Binary search initialization
    left, right = 0, len(arr) - 1

    # Perform binary search
    while left <= right:
        mid = (left + right) // 2  # Calculate middle index
        if arr[mid] == target_timestamp:
            return mid
        elif arr[mid] < target_timestamp:
            left = mid + 1
        else:
            right = mid - 1

    # Determine the closest index between left and right
    if left >= len(arr):
        return right
    if right < 0:
        return left

    # Return the closest index
    if abs(arr[left] - target_timestamp) < abs(arr[right] - target_timestamp):
        return left
    else:
        return right

def segment_MATRIX_data_by_dates(MATRIX_data, start_date, end_date):
    """Extract a segment of IMU data between two dates.
    
    Args:
        IMU_data (np.array): Array of IMU data with timestamps.
        start_date (datetime): Start date for the segment.
        end_date (datetime): End date for the segment.
    
    Returns:
        np.array: Segment of IMU data between the specified dates.
    """
    # Convert start and end dates to timestamps (in milliseconds)
    if not start_date:
        timestamp_start = MATRIX_data[0, 0]
    else:
        timestamp_start = start_date.timestamp() * 1000
    
    if not end_date:
        timestamp_end = MATRIX_data[-1, 0]
    else:
        timestamp_end = end_date.timestamp() * 1000
    
    # Find the closest indices for start and end timestamps
    start_index = find_closest_timestamp(MATRIX_data[:, 0], timestamp_start)
    end_index = find_closest_timestamp(MATRIX_data[:, 0], timestamp_end)
    
    return MATRIX_data[start_index:end_index+1, :]
 

def segment_WPM_activity_data(dictionary_hours_wpm, imu_data):
    """
    Segments activity data based on defined time periods for various activities.
    
    Parameters:
    dictionary_hours_wpm (dict): Dictionary containing time data for various activities.
    imu_data (numpy.ndarray): Array containing the IMU data to be segmented.

    Returns:
    dict: A dictionary containing segmented data for each activity.
    """
    # Create a new dictionary to store segmented data
    segmented_data_wpm = {}

    # List of activities to segment with their corresponding sheet keys
    activities = [
        ('FASE REPOSO CON K5', 'FASE REPOSO CON K5 - Hora de inicio', 'FASE REPOSO CON K5 - Hora de fin', 'Fecha día 1', 'Fecha día 1'),
        ('TAPIZ RODANTE', 'TAPIZ RODANTE - Hora de inicio', 'TAPIZ RODANTE - Hora de fin', 'Fecha día 1', 'Fecha día 1'),
        ('SIT TO STAND 30 s', 'SIT TO STAND 30 s - Hora de inicio', 'SIT TO STAND 30 s - Hora de fin', 'Fecha día 1', 'Fecha día 1'),
        ('INCREMENTAL CICLOERGOMETRO', 'INCREMENTAL CICLOERGOMETRO - Hora de inicio REPOSO', 'INCREMENTAL CICLOERGOMETRO - Hora de fin', 'Fecha día 1', 'Fecha día 1'),
        ('YOGA', 'YOGA - Hora de inicio', 'YOGA - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('SENTADO VIENDO LA TV', 'SENTADO VIENDO LA TV - Hora de inicio', 'SENTADO VIENDO LA TV - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('SENTADO LEYENDO', 'SENTADO LEYENDO - Hora de inicio', 'SENTADO LEYENDO - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('SENTADO USANDO PC', 'SENTADO USANDO PC - Hora de inicio', 'SENTADO USANDO PC - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('DE PIE USANDO PC', 'DE PIE USANDO PC - Hora de inicio', 'DE PIE USANDO PC - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('DE PIE DOBLANDO TOALLAS', 'DE PIE DOBLANDO TOALLAS - Hora de inicio', 'DE PIE DOBLANDO TOALLAS - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('DE PIE MOVIENDO LIBROS', 'DE PIE MOVIENDO LIBROS - Hora de inicio', 'DE PIE MOVIENDO LIBROS - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('DE PIE BARRIENDO', 'DE PIE BARRIENDO - Hora de inicio', 'DE PIE BARRIENDO - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('CAMINAR USUAL SPEED', 'CAMINAR USUAL SPEED - Hora de inicio', 'CAMINAR USUAL SPEED - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('CAMINAR CON MÓVIL O LIBRO', 'CAMINAR CON MÓVIL O LIBRO - Hora de inicio', 'CAMINAR CON MÓVIL O LIBRO - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('CAMINAR CON LA COMPRA', 'CAMINAR CON LA COMPRA - Hora de inicio', 'CAMINAR CON LA COMPRA - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('CAMINAR ZIGZAG', 'CAMINAR ZIGZAG - Hora de inicio', 'CAMINAR ZIGZAG - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('TROTAR', 'TROTAR - Hora de inicio', 'TROTAR - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('SUBIR Y BAJAR ESCALERAS', 'SUBIR Y BAJAR ESCALERAS - Hora de inicio', 'SUBIR Y BAJAR ESCALERAS - Hora de fin', 'Fecha día 7', 'Fecha día 7'),
        ('ACTIVIDAD NO ESTRUCTURADA', 'ACTIVIDAD NO ESTRUCTURADA - Hora de inicio', 'ACTIVIDAD NO ESTRUCTURADA - Hora de fin', 'Fecha día 1', 'Fecha día 7')
    ]

    # Iterate over the activity definitions and segment data
    for activity_name, start_key, end_key, start_date_key, end_date_key in activities:
        start_time = datetime.combine(dictionary_hours_wpm[start_date_key], dictionary_hours_wpm[start_key])
        end_time = datetime.combine(dictionary_hours_wpm[end_date_key], dictionary_hours_wpm[end_key])
        data = segment_MATRIX_data_by_dates(imu_data, start_time, end_time)
        segmented_data_wpm[activity_name] = data

    return segmented_data_wpm


def plot_segmented_WPM_data(WPM_data, images_folder_name, csv_matrix_PMP):
    """
    Plot activity-by-activity segmented data from MATRIX.

    Parameters:
    -----------
    * WPM_data: Dictionary where each key is the name of an activity, and the corresponding entry
      contains the associated data.

    Returns:
    --------
    None.
    """

    activities = ['FASE REPOSO CON K5', 'TAPIZ RODANTE', 'SIT TO STAND 30 s',
                  'INCREMENTAL CICLOERGOMETRO', 'YOGA', 'SENTADO VIENDO LA TV',
                  'SENTADO LEYENDO', 'SENTADO USANDO PC', 'DE PIE USANDO PC',
                  'DE PIE DOBLANDO TOALLAS', 'DE PIE MOVIENDO LIBROS',
                  'DE PIE BARRIENDO', 'CAMINAR USUAL SPEED',
                  'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR CON LA COMPRA',
                  'CAMINAR ZIGZAG', 'TROTAR', 'SUBIR Y BAJAR ESCALERAS']          # A total of 18 activities.

    csv_folder_file = os.path.dirname(csv_matrix_PMP)
    full_path = os.path.join(csv_folder_file, images_folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    
    image_file_name = Path(csv_matrix_PMP).stem
    for activity in activities:
        plt.figure()
        activity_data = WPM_data[activity]
        plt.plot(activity_data[:, 1:4])  # Plot acceleration data (columns 1 to 3)
        plt.title(activity)
        plt.xlabel('Sample [-]')
        plt.ylabel('Accelerometer data [g]')
        plt.grid(True)

        file_name = os.path.join(full_path, image_file_name + '_' + activity + '.jpg')
        plt.savefig(file_name, format='jpg')


################ Windowing and Stacking Functions ################


def save_segmented_data_to_compressed_npz(file_name, segmented_activity_data_wpm):
    """
    Save segmented activity data into a compressed .npz file.

    Parameters:
    -----------
    file_name : str
        Name of the compressed .npz file to be created (without extension).
    segmented_activity_data_wpm : dict
        Dictionary containing segmented activity data to be saved.

    Returns:
    --------
    None
    """
    folder_path = "Segmented_WPM_Data"  # Folder where the compressed .npz file will be saved

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the dictionary of data into a .npz file
    np.savez(os.path.join(folder_path, file_name + ".npz"), **segmented_activity_data_wpm)
    print(f"WPM data saved in the folder: {folder_path}")

def load_dicts_from_npz(file_path):
    """
    Load dictionaries from a .npz file.

    Parameters:
    -----------
    file_path : str
        Path to the .npz file.

    Returns:
    --------
    dict
        A dictionary containing the loaded arrays.
    """
    # Load the .npz file
    loaded_data = np.load(file_path, allow_pickle=True)
    
    # Convert the loaded object into a dictionary
    loaded_dicts = {key: loaded_data[key] for key in loaded_data.files}
    
    print(f"Dictionaries successfully loaded from file: {file_path}")
    return loaded_dicts


def apply_windowing_WPM_segmented_data(data_dict, window_size_samples, step_size_samples=None):
    """
    Applies windowing to a dictionary of arrays based on the number of samples.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the arrays to be windowed.
    window_size_samples : int
        Size of the window in number of samples.
    step_size_samples : int, optional
        Step size for windowing (default is no overlap).

    Returns:
    --------
    dict
        A dictionary with the windowed arrays.
    """
    
    # Set step size to window size if not provided
    if step_size_samples is None:
        step_size_samples = window_size_samples
    else:
        step_size_samples = int(step_size_samples)

    windowed_dict = {}  # Dictionary to store windowed data
    
    for key, array in data_dict.items():
        if array.ndim == 2:
            # Windowing for 2D arrays
            if array.shape[0] <= window_size_samples:
                print(f"Not enough data for windowng.")
                windowed_dict[key] = array
                continue

            num_windows = (array.shape[0] - window_size_samples) // step_size_samples + 1
            if num_windows <= 0:
                print(f"Warning: The array for key '{key}' is too short for windowing.")
                continue
            
            # Generate windows
            windowed_array = np.lib.stride_tricks.sliding_window_view(array, window_shape=(window_size_samples, array.shape[1]))
            windowed_array = windowed_array[:num_windows * step_size_samples:step_size_samples]
            windowed_array = np.squeeze(windowed_array)
            windowed_array_reordered = windowed_array.transpose(0, 2, 1)
            windowed_dict[key] = windowed_array_reordered
            print(f"Windowing applied to '{key}': {windowed_array_reordered.shape} windows (2D)")

        else:
            print(f"Warning: The array for key '{key}' has unsupported dimensions: {array.ndim}D")

    return windowed_dict

def create_labeled_stack_wpm(list_of_windowed_dicts):
    """
    Creates a labeled stack from a list of dictionaries containing windowed data.

    Parameters:
    -----------
    list_of_windowed_dicts : list
        List of dictionaries where each dictionary contains windowed arrays.

    Returns:
    --------
    tuple
        A tuple containing the stacked data (numpy.ndarray) and an array of categorical labels (numpy.ndarray).
    """
    if not isinstance(list_of_windowed_dicts, list):
        raise ValueError("Input must be a list.")

    stacked_data_list = []
    labels_list = []

    # Iterate through the list of dictionaries
    for dict_index, windowed_data_dict in enumerate(list_of_windowed_dicts):
        if not isinstance(windowed_data_dict, dict):
            raise ValueError(f"Element at index {dict_index} is not a dictionary.")
        
        # Iterate through the dictionary to stack data and assign labels
        for label, array in windowed_data_dict.items():
            stacked_data_list.append(array)  # Append the array to the stack
            # Extend the labels list with the corresponding label for each window
            labels_list.extend([label] * array.shape[0])

    # Concatenate all stacked arrays into a single array
    stacked_data = np.concatenate(stacked_data_list, axis=0)

    return stacked_data, np.array(labels_list)


def save_stacked_data_and_labels(stacked_data, labels, folder_path, file_name):
    """
    Saves the stacked data and labels into a .npz file. Creates the folder if it does not exist.

    Parameters:
    -----------
    stacked_data : np.ndarray
        The stacked data to be saved.
    labels : np.ndarray
        The corresponding labels for the data.
    folder_path : str
        Path to the folder where the data will be saved.
    file_name : str
        Name of the file to save the data.

    Returns:
    --------
    None
    """
    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Save the data
    np.savez(file_path, data=stacked_data, labels=labels)
    print(f"Data saved at {file_path}")


def load_stacked_data_and_labels(file_path):
    """
    Loads the stacked data and labels from a .npz file.

    Parameters:
    -----------
    file_path : str
        Path to the file from which the data will be loaded.

    Returns:
    --------
    tuple
        A tuple containing the stacked data (np.ndarray) and labels (np.ndarray).
    """
    try:
        # Load the .npz file
        loaded_data = np.load(file_path, allow_pickle=True)
        
        # Verify that the required keys are present
        if 'data' not in loaded_data or 'labels' not in loaded_data:
            raise KeyError("The file does not contain the keys 'data' or 'labels'.")

        # Retrieve the data and labels
        stacked_data = loaded_data['data']
        labels = loaded_data['labels']
        
        # Ensure they are of type np.ndarray
        assert isinstance(stacked_data, np.ndarray), "Loaded data is not of type np.ndarray"
        assert isinstance(labels, np.ndarray), "Loaded labels are not of type np.ndarray"

        return stacked_data, labels
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except KeyError as e:
        print(f"Error: Missing key '{e.args[0]}' in the file.")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None



if __name__ == "__main__":
	print("main empty")
    