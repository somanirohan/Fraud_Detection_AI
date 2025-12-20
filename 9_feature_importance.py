"""
3. Feature Importance Explainability (Alternative to SHAP)
Explain model predictions using feature importance
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("FEATURE IMPORTANCE EXPLAINABILITY FOR FRAUD DETECTION")
print("="*100)

# Load models and data
print("\n📦 Loading models and data...")
xgb_model = joblib.load('models/xgboost_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
X_test = pd.read_csv('processed_data/X_test.csv')
y_test = pd.read_csv('processed_data/y_test.csv')['is_fraud']
feature_names = joblib.load('models/feature_names.pkl')

print("✅ Models and data loaded\n")

# ============================================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================================
print(f"{'='*100}")
print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
print(f"{'='*100}\n")

# Get feature importance
xgb_importance = xgb_model.feature_importances_

# Create DataFrame
xgb_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance,
    'Importance_%': xgb_importance / xgb_importance.sum() * 100
}).sort_values('Importance', ascending=False)

print("Top 30 Most Important Features for Fraud Detection:\n")
print(xgb_feature_importance.head(30).to_string(index=False))

# Save
xgb_feature_importance.to_csv('models/xgboost_feature_importance.csv', index=False)
print("\n💾 Saved to: models/xgboost_feature_importance.csv")

# ============================================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print(f"\n{'='*100}")
print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
print(f"{'='*100}\n")

# Get feature importance
rf_importance = rf_model.feature_importances_

# Create DataFrame
rf_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance,
    'Importance_%': rf_importance / rf_importance.sum() * 100
}).sort_values('Importance', ascending=False)

print("Top 30 Most Important Features for Fraud Detection:\n")
print(rf_feature_importance.head(30).to_string(index=False))

# Save
rf_feature_importance.to_csv('models/random_forest_feature_importance.csv', index=False)
print("\n💾 Saved to: models/random_forest_feature_importance.csv")

# ============================================================================
# COMBINED FEATURE IMPORTANCE
# ============================================================================
print(f"\n{'='*100}")
print("COMBINED FEATURE IMPORTANCE (XGBoost + Random Forest)")
print(f"{'='*100}\n")

# Average importance from both models
combined_importance = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost_Importance': xgb_importance,
    'RandomForest_Importance': rf_importance,
    'Average_Importance': (xgb_importance + rf_importance) / 2,
    'Average_Importance_%': ((xgb_importance + rf_importance) / 2) / ((xgb_importance + rf_importance) / 2).sum() * 100
}).sort_values('Average_Importance', ascending=False)

print("Top 30 Most Important Features (Combined):\n")
print(combined_importance.head(30)[['Feature', 'Average_Importance_%']].to_string(index=False))

# Save
combined_importance.to_csv('models/combined_feature_importance.csv', index=False)
print("\n💾 Saved to: models/combined_feature_importance.csv")

# ============================================================================
# EXPLAIN SPECIFIC PREDICTION
# ============================================================================
print(f"\n{'='*100}")
print("EXAMPLE: EXPLAINING A FRAUD PREDICTION")
print(f"{'='*100}\n")

# Get a fraud transaction
fraud_idx = y_test[y_test == 1].index[0]
fraud_transaction = X_test.iloc[fraud_idx:fraud_idx+1]

print(f"Transaction Index: {fraud_idx}")
print(f"Actual Label: 🚨 FRAUD")

# Get prediction
xgb_pred = xgb_model.predict(fraud_transaction)[0]
xgb_proba = xgb_model.predict_proba(fraud_transaction)[0, 1]

print(f"XGBoost Prediction: {'🚨 FRAUD' if xgb_pred == 1 else '✅ Legitimate'}")
print(f"Fraud Probability: {xgb_proba*100:.2f}%\n")

# Get feature values for this transaction
transaction_features = fraud_transaction.iloc[0]

# Get top features by importance and their values
top_features = combined_importance.head(20)
explanation_data = []

for _, row in top_features.iterrows():
    feature_name = row['Feature']
    feature_value = transaction_features[feature_name]
    importance = row['Average_Importance_%']
    
    explanation_data.append({
        'Feature': feature_name,
        'Value': feature_value,
        'Importance_%': f"{importance:.2f}",
        'Impact': 'High' if importance > 2 else 'Medium' if importance > 1 else 'Low'
    })

explanation_df = pd.DataFrame(explanation_data)
print("Top 20 Features Contributing to This Prediction:\n")
print(explanation_df.to_string(index=False))

# ============================================================================
# CREATE EXPLANATION FUNCTION
# ============================================================================
print(f"\n{'='*100}")
print("CREATING EXPLANATION FUNCTION")
print(f"{'='*100}\n")

explanation_code = """
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
"""

with open('explain_fraud.py', 'w') as f:
    f.write(explanation_code)

print("✅ Explanation function saved to: explain_fraud.py")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print("✅ FEATURE IMPORTANCE EXPLAINABILITY COMPLETE!")
print(f"{'='*100}\n")

print("📝 Summary:")
print("  ✅ XGBoost feature importance calculated (243 features)")
print("  ✅ Random Forest feature importance calculated (243 features)")
print("  ✅ Combined feature importance created")
print("  ✅ Individual prediction explanation demonstrated")
print("  ✅ Reusable explanation function saved")

print("\n💾 Saved Files:")
print("  • models/xgboost_feature_importance.csv")
print("  • models/random_forest_feature_importance.csv")
print("  • models/combined_feature_importance.csv")
print("  • explain_fraud.py")

print(f"\n{'='*100}")
print("🎯 TOP 10 MOST IMPORTANT FEATURES FOR FRAUD DETECTION")
print(f"{'='*100}\n")

for i, row in combined_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['Feature']:<50} {row['Average_Importance_%']:>6.2f}%")

print("\n✨ These features have the highest impact on fraud detection decisions")
