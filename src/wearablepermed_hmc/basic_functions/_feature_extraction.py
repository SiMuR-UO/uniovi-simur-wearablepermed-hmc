import numpy as np
import scipy as sp
from . import _preprocessing as prep


def get_basic_stats(epochdata, filter_b = [], filter_a = []):
    """
    Calculate basic statistics for the given epoch data.

    Parameters:
    epochdata (numpy.ndarray): A 2D array where each column represents a signal and each row represents a temporal value.
    filter_b (numpy.ndarray): The numerator coefficient vector of the filter.
    filter_a (numpy.ndarray): The denominator coefficient vector of the filter.

    Returns:
    tuple: A tuple containing the basic statistics, truncated ENMO, and filtered ENMO.
    """
    means = np.mean(epochdata, axis=0)
    ranges = np.ptp(epochdata, axis=0)
    cov_matrix = np.cov(epochdata, rowvar=False) #rowvar=False significa que cada columna de epochdata representa una variable diferente y cada fila es una observación.
    std_devs = np.sqrt(np.diag(cov_matrix))
    covariances = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]

    # Calculate ENMO
    enmo = prep.ENMO(epochdata)
    if len(filter_b) and len(filter_a):
        enmo_filtered = sp.signal.lfilter(filter_b, filter_a, enmo)
    else:
        enmo_filtered = enmo
        
    enmo_trunc = enmo_filtered * (enmo_filtered > 0)
    enmo_trunc_mean = np.mean(enmo_trunc)
    enmo_abs = np.abs(enmo_filtered)
    enmo_abs_mean = np.mean(enmo_abs)

    basic_statistics = [enmo_trunc_mean, enmo_abs_mean] + means.tolist() + ranges.tolist() + std_devs.tolist() + covariances.tolist()
    return basic_statistics, enmo_trunc, enmo_filtered


def get_FFT_power(FFT, normalize=True):
    """
    Calculate the power of the FFT (Fast Fourier Transform) coefficients.

    Parameters:
    FFT (array-like): The FFT coefficients.
    normalize (bool, optional): If True, normalize the power by the square of the length of the FFT. Default is True.

    Returns:
    array-like: The power of the FFT coefficients, optionally normalized.
    """
    FFTpow = np.abs(FFT) ** 2
    if normalize:
        FFTpow /= len(FFT) ** 2
    return FFTpow

def get_FFT_magnitude(FFT, normalize=True):
    """
    Calculate the magnitudes from FFT (Fast Fourier Transform) coefficients.

    Parameters:
    FFT (array-like): The FFT coefficients.
    normalize (bool, optional): If True, normalize the magnitudes by the length of the FFT. Default is True.

    Returns:
    array-like: The magnitudes of the FFT coefficients.
    """
    # Use numpy's absolute function to get magnitudes from FFT coefficients
    FFTmag = np.abs(FFT)
    if normalize:
        FFTmag /= len(FFT)
    return FFTmag


