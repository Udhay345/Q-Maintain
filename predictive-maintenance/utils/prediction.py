import pandas as pd
import numpy as np

def predict_health_status(model, X_preprocessed):
    """
    Executes model prediction and extracts class probabilities.
    
    Args:
        model: Trained XGBClassifier
        X_preprocessed (pd.DataFrame): Dataframe with 12 features in exact expected order.
        
    Returns:
        tuple: (pred_class (int), pred_proba (np.array of length 3))
    """
    # Average probabilities across rows for stabler live predictions
    proba_matrix = model.predict_proba(X_preprocessed)
    pred_proba = np.mean(proba_matrix, axis=0)
    pred_class = int(np.argmax(pred_proba))
    return pred_class, pred_proba
