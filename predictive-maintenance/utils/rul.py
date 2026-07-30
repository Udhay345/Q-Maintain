import numpy as np

def estimate_health_and_rul(features, classification_probs=None):
    """
    Computes a device health score and converts it to Remaining Useful Life (RUL).
    
    This function is structured as a prototype health indicator that combines:
    1. Vibration/anomaly penalty from raw sensor features (rolling standard deviations/RMS).
    2. Classifier probabilities (how confident the model is that the state is GOOD vs BAD/CRITICAL).
    
    It serves as a clean placeholder where a supervised regression model can later
    be registered and evaluated.
    
    Args:
        features (dict): Engineered rolling-window features.
        classification_probs (list/np.array, optional): Model prediction probabilities [p_good, p_bad, p_critical].
        
    Returns:
        tuple: (health_score (float 0-100), estimated_rul_hours (float), trend (str))
    """
    if not features:
        # Default starting values
        return 100.0, 72.0, "Stable"
        
    # Get key physical characteristics
    acc_std = features.get("acc_std", 0.0)
    gyro_rms = features.get("gyro_rms", 0.0)
    lin_acc_rms = features.get("lin_acc_rms", 0.0)
    
    # 1. Physics-based degradation index
    # We establish baseline bounds for a "Good" stationary phone:
    # acc_std <= 0.15 m/s^2, gyro_rms <= 0.05 rad/s, lin_acc_rms <= 0.10 m/s^2
    acc_penalty = max(0.0, (acc_std - 0.15) / 1.5)
    gyro_penalty = max(0.0, (gyro_rms - 0.05) / 1.0)
    lin_penalty = max(0.0, (lin_acc_rms - 0.10) / 1.0)
    
    # Weighted degradation value (0.0 to 1.0)
    degradation = 0.4 * acc_penalty + 0.3 * gyro_penalty + 0.3 * lin_penalty
    degradation = min(1.0, max(0.0, degradation))
    
    # Physical health score (0% to 100%)
    physical_health = 100.0 * (1.0 - degradation)
    
    # 2. Classifier-guided health evaluation
    classifier_health = 100.0
    if classification_probs is not None and len(classification_probs) >= 3:
        p_good = classification_probs[0]
        p_bad = classification_probs[1]
        p_critical = classification_probs[2]
        
        # Good counts for 100%, Bad is degradation warning, Critical indicates 0% health
        classifier_health = (p_good * 100.0) + (p_bad * 40.0) + (p_critical * 0.0)
        
    # Combine physical and classifier-based metrics (50/50 split)
    health_score = 0.5 * physical_health + 0.5 * classifier_health
    health_score = round(min(100.0, max(0.0, health_score)), 1)
    
    # 3. Remaining Useful Life (RUL) translation
    # System prototype assumes normal operating lifetime is 72 hours (3.0 days).
    # Lifetime degrades non-linearly with the health score.
    # RUL = Max_RUL * (health_score / 100) ^ 1.2
    estimated_rul_hours = 72.0 * ((health_score / 100.0) ** 1.2)
    estimated_rul_hours = round(max(0.0, estimated_rul_hours), 1)
    
    # 4. Health Trend Determination
    if health_score > 85.0:
        trend = "Stable"
    elif health_score > 50.0:
        trend = "Degrading"
    else:
        trend = "Critical Alert"
        
    return health_score, estimated_rul_hours, trend
