import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl                                        # Librería para leer datos desde una Hoja Excel
from datetime import time, timedelta, date, datetime
import matplotlib.pyplot as plt

import openpyxl
from datetime import time, timedelta, date

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

def segment_data_by_dates(MATRIX_data, start_date, end_date):
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
        ('FASE REPOSO CON K5', 'FASE REPOSO CON K5 - Hora de inicio', 'FASE REPOSO CON K5 - Hora de fin', 'Fecha día 1'),
        ('TAPIZ RODANTE', 'TAPIZ RODANTE - Hora de inicio', 'TAPIZ RODANTE - Hora de fin', 'Fecha día 1'),
        ('SIT TO STAND 30 s', 'SIT TO STAND 30 s - Hora de inicio', 'SIT TO STAND 30 s - Hora de fin', 'Fecha día 1'),
        ('INCREMENTAL CICLOERGOMETRO', 'INCREMENTAL CICLOERGOMETRO - Hora de inicio REPOSO', 'INCREMENTAL CICLOERGOMETRO - Hora de fin', 'Fecha día 1'),
        ('YOGA', 'YOGA - Hora de inicio', 'YOGA - Hora de fin', 'Fecha día 7'),
        ('SENTADO VIENDO LA TV', 'SENTADO VIENDO LA TV - Hora de inicio', 'SENTADO VIENDO LA TV - Hora de fin', 'Fecha día 7'),
        ('SENTADO LEYENDO', 'SENTADO LEYENDO - Hora de inicio', 'SENTADO LEYENDO - Hora de fin', 'Fecha día 7'),
        ('SENTADO USANDO PC', 'SENTADO USANDO PC - Hora de inicio', 'SENTADO USANDO PC - Hora de fin', 'Fecha día 7'),
        ('DE PIE USANDO PC', 'DE PIE USANDO PC - Hora de inicio', 'DE PIE USANDO PC - Hora de fin', 'Fecha día 7'),
        ('DE PIE DOBLANDO TOALLAS', 'DE PIE DOBLANDO TOALLAS - Hora de inicio', 'DE PIE DOBLANDO TOALLAS - Hora de fin', 'Fecha día 7'),
        ('DE PIE MOVIENDO LIBROS', 'DE PIE MOVIENDO LIBROS - Hora de inicio', 'DE PIE MOVIENDO LIBROS - Hora de fin', 'Fecha día 7'),
        ('DE PIE BARRIENDO', 'DE PIE BARRIENDO - Hora de inicio', 'DE PIE BARRIENDO - Hora de fin', 'Fecha día 7'),
        ('CAMINAR USUAL SPEED', 'CAMINAR USUAL SPEED - Hora de inicio', 'CAMINAR USUAL SPEED - Hora de fin', 'Fecha día 7'),
        ('CAMINAR CON MÓVIL O LIBRO', 'CAMINAR CON MÓVIL O LIBRO - Hora de inicio', 'CAMINAR CON MÓVIL O LIBRO - Hora de fin', 'Fecha día 7'),
        ('CAMINAR CON LA COMPRA', 'CAMINAR CON LA COMPRA - Hora de inicio', 'CAMINAR CON LA COMPRA - Hora de fin', 'Fecha día 7'),
        ('CAMINAR ZIGZAG', 'CAMINAR ZIGZAG - Hora de inicio', 'CAMINAR ZIGZAG - Hora de fin', 'Fecha día 7'),
        ('TROTAR', 'TROTAR - Hora de inicio', 'TROTAR - Hora de fin', 'Fecha día 7'),
        ('SUBIR Y BAJAR ESCALERAS', 'SUBIR Y BAJAR ESCALERAS - Hora de inicio', 'SUBIR Y BAJAR ESCALERAS - Hora de fin', 'Fecha día 7'),
    ]

    # Iterate over the activity definitions and segment data
    for activity_name, start_key, end_key, date_key in activities:
        try:
            start_time = datetime.combine(dictionary_hours_wpm[date_key], dictionary_hours_wpm[start_key])
            end_time = datetime.combine(dictionary_hours_wpm[date_key], dictionary_hours_wpm[end_key])
            data = segment_data_by_dates(imu_data, start_time, end_time)
            segmented_data_wpm[activity_name] = data
        except Exception as e:
            print(f"An error occurred: {e}")

    return segmented_data_wpm

def plot_segmented_matrix_data(WPM_data, file_name):
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

    for activity in activities:
        plt.figure()
        activity_data = WPM_data[activity]
        plt.plot(activity_data[:, 1:4])  # Plot acceleration data (columns 1 to 3)
        plt.title(activity)
        plt.xlabel('Sample [-]')
        plt.ylabel('Accelerometer data [g]')
        plt.grid(True)

        if file_name is not None:
            plt.savefig(file_name + '_' + activity + '.jpg', format='jpg')

if __name__ == "__main__":
	print("main empty")
    