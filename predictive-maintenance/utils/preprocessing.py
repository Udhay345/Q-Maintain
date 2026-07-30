import pandas as pd
import numpy as np

REQUIRED_FEATURES = [
    "Acceleration x (m/s^2)",
    "Acceleration y (m/s^2)",
    "Acceleration z (m/s^2)",
    "Gyroscope x (rad/s)",
    "Gyroscope y (rad/s)",
    "Gyroscope z (rad/s)",
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)",
    "Magnetic field x (µT)",
    "Magnetic field y (µT)",
    "Magnetic field z (µT)"
]

def clean_columns(df):
    col_mapping = {}
    for col in df.columns:
        c_str = str(col).strip()
        # Clean potential character errors (e.g. latin-1 micro \xb5 or raw )
        c_str = c_str.replace('\xb5', 'µ').replace('', 'µ')
        c_low = c_str.lower()
        
        if 'acceleration x' in c_low and 'linear' not in c_low:
            col_mapping[col] = "Acceleration x (m/s^2)"
        elif 'acceleration y' in c_low and 'linear' not in c_low:
            col_mapping[col] = "Acceleration y (m/s^2)"
        elif 'acceleration z' in c_low and 'linear' not in c_low:
            col_mapping[col] = "Acceleration z (m/s^2)"
            
        elif 'gyroscope x' in c_low:
            col_mapping[col] = "Gyroscope x (rad/s)"
        elif 'gyroscope y' in c_low:
            col_mapping[col] = "Gyroscope y (rad/s)"
        elif 'gyroscope z' in c_low:
            col_mapping[col] = "Gyroscope z (rad/s)"
            
        elif 'linear acceleration x' in c_low:
            col_mapping[col] = "Linear Acceleration x (m/s^2)"
        elif 'linear acceleration y' in c_low:
            col_mapping[col] = "Linear Acceleration y (m/s^2)"
        elif 'linear acceleration z' in c_low:
            col_mapping[col] = "Linear Acceleration z (m/s^2)"
            
        elif ('magnetic field x' in c_low) or (c_low.startswith('magnetic') and ' x' in c_low):
            col_mapping[col] = "Magnetic field x (µT)"
        elif ('magnetic field y' in c_low) or (c_low.startswith('magnetic') and ' y' in c_low):
            col_mapping[col] = "Magnetic field y (µT)"
        elif ('magnetic field z' in c_low) or (c_low.startswith('magnetic') and ' z' in c_low):
            col_mapping[col] = "Magnetic field z (µT)"
            
        elif c_low == 'status' or c_low.endswith('status'):
            col_mapping[col] = "Status"
            
    return df.rename(columns=col_mapping)

def preprocess_features(df, medians=None):
    """
    Cleans columns, ensures all 12 REQUIRED_FEATURES are present and cast to float,
    and fills missing values using medians from training.
    """
    df_clean = clean_columns(df.copy())
    
    # Cast to numeric
    for col in REQUIRED_FEATURES:
        if col not in df_clean.columns:
            # If missing entirely, fill with NaN to be imputed
            df_clean[col] = np.nan
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    X = df_clean[REQUIRED_FEATURES].copy()
    if medians is not None:
        X = X.fillna(medians)
    return X
