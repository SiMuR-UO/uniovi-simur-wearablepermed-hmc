import argparse
import logging
import sys
from pathlib import Path
import os
import numpy as np
from scipy.signal import find_peaks
from math import atan2, sqrt

from data_import._WPM_file_management import load_scale_WPM_data
from data_import._WPM_segmentation import segment_WPM_activity_data, plot_segmented_WPM_data, apply_windowing_WPM_segmented_data
from basic_functions._autocalibration import count_stuck_vals, get_calibration_coefs, auto_calibrate

from basic_functions._spectral_features_calculator import obtener_caracteristicas_espectrales

__author__ = "Alejandro Castellanos Alonso, Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo, Alejandro Castellanos Alonso"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

_DEF_TIME_OFF = True
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
        "-has-timeoff",
        "--has-timeoff",
        default=_DEF_TIME_OFF,
        dest="has_timeoff", 
        help="activity regiter has time off")                  
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
        "-make-feature-extractions",
        "--make-feature-extractions",
        dest="make_feature_extractions",
        action='store_true',
        help="make feature extractions?.")          
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)

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
def scale(csv_matrix_PMP, segment_body, activity_PMP, calibrate_with_start_WALKING_USUAL_SPEED=None):
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

def extract_features(data):
    # ***************
    # 1.- Cuantiles *
    # ***************
    # El vector de características empleado en el entrenamiento del Random-Forest será:
    # [Mín, Máx, Mediana, Percentil 25,Percentil 75] para Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Acc, Gyr.
    # self.X_train = data.X_train
    minimos_train = np.quantile(data, 0, axis=2, keepdims=True)
    maximos_train = np.quantile(data, 1, axis=2, keepdims=True)
    medianas_train = np.quantile(data, 0.5, axis=2, keepdims=True)
    Percentil_25_train = np.quantile(data, 0.25, axis=2, keepdims=True)
    Percentil_75_train = np.quantile(data, 0.75, axis=2, keepdims=True)
    Matriz_de_cuantiles_train = np.hstack((minimos_train, maximos_train, medianas_train, Percentil_25_train, Percentil_75_train))
    Matriz_de_cuantiles_train = np.squeeze(Matriz_de_cuantiles_train, axis=2)
    
    
    # *********************************
    # 2.- Características espectrales *
    # *********************************
    # Inicializamos las matrices de resultados
    num_filas = (data).shape[0]  # m ejemplos
    num_columnas = (data).shape[1]  # 12
    
    matriz_resultados_armonicos = np.zeros((num_filas,30))    # 1 IMU
    # matriz_resultados_armonicos = np.zeros((num_filas,60))    # 2 IMUs
    # Recorremos cada serie temporal y calculamos las características
    for i in range(num_filas):
        armonicos_totales = np.zeros((6,5))      # 1 IMU  
        # armonicos_totales = np.zeros((12,5))   # 2 IMUs
        for j in range(num_columnas):
            # Extraemos la serie temporal de longitud 250
            serie = data[i, j, :]
            # Calculamos las características espectrales
            resultado_armonicos,_ = obtener_caracteristicas_espectrales(serie,25)
            armonicos_totales[j, :] = resultado_armonicos
        armonicos_totales_2 = np.reshape(armonicos_totales,(1,-1))
        matriz_resultados_armonicos[i,:] = armonicos_totales_2
    
    
    # *****************************************
    # 3.- Número de picos y prominencia media *
    # *****************************************
    matriz_resultados_numero_picos = np.zeros((num_filas,12))   # 1 IMUs
    # matriz_resultados_numero_picos = np.zeros((num_filas,24))   # 2 IMUs
    # # Recorremos cada serie temporal y calculamos los picos
    for i in range(num_filas):  
        picos_totales = np.zeros(6)         # 1 IMU
        prominencias_totales = np.zeros(6)  # 1 IMUs
        # picos_totales = np.zeros(12)      # 2 IMUs
        # prominencias_totales = np.zeros(12) # 2 IMUs
        for j in range(num_columnas):
            # Extraemos la serie temporal de longitud 250
            serie = data[i, j, :]
            # Calculamos las características espectrales
            indices_picos, propiedades_picos = find_peaks(serie, prominence=True)
            numero_picos=len(indices_picos)
            if numero_picos > 0:
                # Si se detectaron picos, podemos proceder con el cálculo
                prominencias_picos = propiedades_picos['prominences']
                # Por ejemplo, calcular la mediana de la prominencia de los picos
                prominencia_media = np.median(prominencias_picos)
                #print(f"Mediana de prominencia: {prominencia_media}")
            else:
                # prominencia_media = np.NaN
                prominencia_media = 0
            
            #prominencias_picos = propiedades_picos['prominences']
            # Guardamos los resultados en las matrices correspondientes
            # f1_mat[i, j]      = f1
            # p1_mat[i, j]      = p1
            # f2_mat[i, j]      = f2
            # p2_mat[i, j]      = p2
            # entropy_mat[i, j] = entropy
            
            picos_totales[j] = numero_picos
            prominencias_totales[j] = prominencia_media
            
        picos_totales_2 = np.reshape(picos_totales,(1,-1))
        prominencias_totales_2 = np.reshape(prominencias_totales,(1,-1))
        matriz_resultados_numero_picos[i,:] = np.hstack((picos_totales_2, prominencias_totales_2))
    
    
    # *******************
    # 4.- Correlaciones *
    # *******************
    matriz_correlaciones = np.zeros((num_filas,15))  # 1 IMU
    # matriz_correlaciones = np.zeros((num_filas,66))  # 2 IMUs
    for i in range(num_filas):
        # Calcular la matriz de correlación entre las filas
        correlacion = np.corrcoef(data[i,:,:], rowvar=True)
        # Extraer la parte superior de la matriz sin la diagonal principal
        upper_triangle_values = correlacion[np.triu_indices_from(correlacion, k=1)]
        # print(upper_triangle_values)
        
        matriz_correlaciones[i,:] = upper_triangle_values
    #self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones))
    #print(self.X_train)
    
    # **************************************
    # 5.- Autocorrelación del acelerómetro *
    # **************************************
    matriz_resultados_autocorrelacion = np.zeros((num_filas, 1))
    # matriz_resultados_autocorrelacion = np.zeros((num_filas, 2))
    # Recorremos cada serie temporal y calculamos los picos
    for i in range(num_filas):
        serie = np.linalg.norm(data[i,0:3,:], axis=0)
        # serie_desplazada = np.pad(serie[-25], (25,), mode='constant', constant_values=0)
        serie_desplazada = np.empty_like(serie)
        serie_desplazada[:25] = 0
        serie_desplazada[25:] = serie[:-25]
            
        autocorrelacion_acc_IMU1 = np.corrcoef(serie, serie_desplazada)

        serie = np.linalg.norm(data[i,6:9,:], axis=0)
        serie_desplazada = np.empty_like(serie)
        serie_desplazada[:25] = 0
        serie_desplazada[25:] = serie[:-25]
        # serie_desplazada = np.pad(serie[:,-25], (25,0), mode='constant', constant_values=0)
        autocorrelacion_acc_IMU2 = np.corrcoef(serie, serie_desplazada)
        
        # modulo_acc_IMU1 = np.linalg.norm(data.X_train[i,0:3,:], axis=0)
        # modulo_acc_IMU2 = np.linalg.norm(data.X_train[i,6:9,:], axis=0)
        # autocorrelacion_acc_IMU2 = np.corrcoef(modulo_acc_IMU2, nlags=25)
        
        matriz_resultados_autocorrelacion[i,0] = autocorrelacion_acc_IMU1[0,1]
        # matriz_resultados_autocorrelacion[i,1] = autocorrelacion_acc_IMU2[0,1]
    
    # self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion))      
    
    # **************************************************
    # 6.- Componentes roll, pitch y yaw del movimiento *
    # **************************************************
    dt = 1/25      # Período de muestreo en [s]
    rolls_promedio = np.zeros((num_filas, 1))
    pitches_promedio = np.zeros((num_filas, 1))
    yaws_promedio = np.zeros((num_filas, 1))
    for i in range(num_filas):
        rolls = []
        pitches = []
        yaws = []
        # Extraemos las series temporales de longitud 250 muestras (acelerómetro y giroscopio)
        serie_acc_x = data[i, 0, :]
        serie_acc_y = data[i, 1, :]
        serie_acc_z = data[i, 2, :]
        serie_gyr_x = data[i, 3, :]
        serie_gyr_y = data[i, 4, :]
        serie_gyr_z = data[i, 5, :]
        
        yaw_acumulado = 0
        for j in range(len(serie_acc_x)):
            acc_x = serie_acc_x[j]
            acc_y = serie_acc_y[j]
            acc_z = serie_acc_z[j]
            gyr_x = serie_gyr_x[j]
            gyr_y = serie_gyr_y[j]
            gyr_z = serie_gyr_z[j]

            roll = atan2(acc_y, acc_z)                             # Roll: rotación alrededor del eje X
            pitch = atan2(-acc_x, sqrt(acc_y**2 + acc_z**2))  # Pitch: rotación alrededor del eje Y
            yaw = gyr_z * dt                                            # Integración simple para obtener el cambio de yaw
            yaw_acumulado += yaw                                        # Efecto acumulativo de la acción integral
            rolls.append(roll)
            pitches.append(pitch)
        yaws.append(yaw_acumulado)
        yaw_acumulado = 0
        
        rolls_promedio[i] = np.mean(rolls)
        pitches_promedio[i] = np.mean(pitches)
        yaws_promedio[i] = np.mean(yaws)
    
    X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion, rolls_promedio, pitches_promedio, yaws_promedio))    

    return X_train

def export_data(concatenated_data, all_labels, export_folder_name):
    np.savez(export_folder_name, concatenated_data, all_labels)

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)
    
    _logger.info("Aggregator starts here")

    _logger.debug("Step 00: Extracting metadata ...")
    participant_id, measurement_date, segment_body = extract_metadata_from_csv(args.csv_matrix_PMP)
    
    _logger.debug("Step 01: Starting Scale Data ...")
    if args.has_timeoff == True:
        scaled_data, dictionary_timing = scale(
            args.csv_matrix_PMP,
            segment_body, 
            args.activity_PMP)        
    else:
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
    export_data(concatenated_data, all_labels, args.export_folder_name) 

    if args.make_feature_extractions == True:
        _logger.debug("Step 07: Starting the calculus of features vectors ...")
        extract_features_data = extract_features(concatenated_data)
    
        _logger.debug("Step 08: Starting Exporting Data ...")
        folder = Path(args.export_folder_name)
        file_name = Path(args.export_folder_name).stem
        
        file_feature_extractions_name = os.path.join(folder.parent, file_name + "_features" + ".npz")
        
        export_data(extract_features_data, all_labels, file_feature_extractions_name)
    
    _logger.info("Aggregator ends here")

def run():
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
