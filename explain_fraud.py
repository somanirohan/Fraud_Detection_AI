
def explain_fraud_prediction(transaction_data):
    '''
    Explain fraud prediction using feature importance
    
    Args:
        transaction_data: pandas DataFrame with single transaction
        
    Returns:
        dict with explanation
    '''
    import joblib
    import pandas as pd
    
    # Load models and feature importance
    xgb_model = joblib.load('models/xgboost_model.pkl')
    feature_importance = pd.read_csv('models/combined_feature_importance.csv')
    
    # Get prediction
    prediction = xgb_model.predict(transaction_data)[0]
    probability = xgb_model.predict_proba(transaction_data)[0, 1]
    
    # Get top features
    top_features = feature_importance.head(10)
    
    # Get feature values
    transaction_values = transaction_data.iloc[0]
    
    # Create explanation
    key_factors = []
    for _, row in top_features.iterrows():
        feature_name = row['Feature']
        feature_value = transaction_values[feature_name]
        importance = row['Average_Importance_%']
        
        key_factors.append({
            'feature': feature_name,
            'value': feature_value,
            'importance_percent': f"{importance:.2f}%"
        })
    
    # Risk level
    if probability >= 0.80:
        risk_level = 'HIGH RISK'
        action = 'BLOCK & INVESTIGATE'
    elif probability >= 0.50:
        risk_level = 'MEDIUM RISK'
        action = 'FLAG FOR REVIEW'
    else:
        risk_level = 'LOW RISK'
        action = 'ALLOW'
    
    return {
        'prediction': 'FRAUD' if prediction == 1 else 'Legitimate',
        'fraud_probability': f"{probability*100:.2f}%",
        'risk_level': risk_level,
        'recommended_action': action,
        'key_factors': key_factors,
        'explanation': f"This transaction has a {probability*100:.1f}% probability of being fraud. "
                      f"The prediction is based on {len(key_factors)} key features including: "
                      f"{', '.join([f['feature'] for f in key_factors[:3]])}."
    }

# Example usage:
# result = explain_fraud_prediction(transaction_df)
# print(result['explanation'])
# print(f"Risk Level: {result['risk_level']}")
# print(f"Action: {result['recommended_action']}")
