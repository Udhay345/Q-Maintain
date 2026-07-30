import numpy as np
import pandas as pd

def calculate_rms(x, y, z):
    """Calculate the Root Mean Square of three spatial components."""
    return np.sqrt(np.mean(x**2 + y**2 + z**2))

def calculate_magnitude(x, y, z):
    """Calculate vector magnitude for each sample."""
    return np.sqrt(x**2 + y**2 + z**2)

def extract_features(df_window):
    """
    Extracts time-window statistical features from a rolling window of sensor data.
    
    Args:
        df_window (pd.DataFrame): Rolling window buffer containing the 12 features.
                                  Target length is typically 100 samples.
                                  
    Returns:
        dict: Calculated statistical and RMS indicators.
    """
    if df_window is None or df_window.empty or len(df_window) < 5:
        # Return empty indicators if buffer has insufficient history
        return {}
        
    try:
        acc_x = df_window["Acceleration x (m/s^2)"]
        acc_y = df_window["Acceleration y (m/s^2)"]
        acc_z = df_window["Acceleration z (m/s^2)"]
        
        gyro_x = df_window["Gyroscope x (rad/s)"]
        gyro_y = df_window["Gyroscope y (rad/s)"]
        gyro_z = df_window["Gyroscope z (rad/s)"]
        
        lin_x = df_window["Linear Acceleration x (m/s^2)"]
        lin_y = df_window["Linear Acceleration y (m/s^2)"]
        lin_z = df_window["Linear Acceleration z (m/s^2)"]
        
        mag_x = df_window["Magnetic field x (µT)"]
        mag_y = df_window["Magnetic field y (µT)"]
        mag_z = df_window["Magnetic field z (µT)"]
        
        # Vector Magnitudes
        acc_mag = calculate_magnitude(acc_x, acc_y, acc_z)
        gyro_mag = calculate_magnitude(gyro_x, gyro_y, gyro_z)
        lin_mag = calculate_magnitude(lin_x, lin_y, lin_z)
        mag_mag = calculate_magnitude(mag_x, mag_y, mag_z)
        
        features = {
            "acc_rms": calculate_rms(acc_x, acc_y, acc_z),
            "acc_std": float(np.std(acc_mag)),
            "gyro_rms": calculate_rms(gyro_x, gyro_y, gyro_z),
            "gyro_std": float(np.std(gyro_mag)),
            "lin_acc_rms": calculate_rms(lin_x, lin_y, lin_z),
            "mag_magnitude": float(np.mean(mag_mag)),
            "sensor_variance": float(np.mean([np.var(acc_mag), np.var(gyro_mag), np.var(lin_mag), np.var(mag_mag)])),
            "peak_acc": float(np.max(acc_mag)),
            "peak_gyro": float(np.max(gyro_mag))
        }
        return features
    except Exception:
        return {}
