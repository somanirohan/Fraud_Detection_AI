
def predict_fraud(transaction_data):
    '''
    Predict fraud for a single transaction or batch
    
    Args:
        transaction_data: pandas DataFrame with same features as training data
        
    Returns:
        dict with predictions from all models
    '''
    import joblib
    
    # Load models
    xgb_model = joblib.load('models/xgboost_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    iso_model = joblib.load('models/isolation_forest_model.pkl')
    
    # Predictions
    xgb_proba = xgb_model.predict_proba(transaction_data)[:, 1]
    rf_proba = rf_model.predict_proba(transaction_data)[:, 1]
    iso_pred = iso_model.predict(transaction_data)
    iso_proba = (iso_pred == -1).astype(float)
    
    # Ensemble
    ensemble_proba = 0.70 * xgb_proba + 0.20 * iso_proba + 0.10 * rf_proba
    
    return {
        'xgboost_probability': xgb_proba,
        'random_forest_probability': rf_proba,
        'isolation_forest_prediction': iso_pred,
        'ensemble_probability': ensemble_proba,
        'final_prediction': (ensemble_proba >= 0.5).astype(int),
        'risk_level': ['HIGH' if p >= 0.8 else 'MEDIUM' if p >= 0.5 else 'LOW' for p in ensemble_proba]
    }
