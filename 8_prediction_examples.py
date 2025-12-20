"""
2. Create Prediction Examples
Demonstrate how to use all 3 models for fraud detection
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("FRAUD DETECTION PREDICTION EXAMPLES")
print("="*100)

# Load models and scaler
print("\n📦 Loading models...")
xgb_model = joblib.load('models/xgboost_model.pkl')
iso_model = joblib.load('models/isolation_forest_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print("✅ Models loaded")

# Load test data for examples
X_test = pd.read_csv('processed_data/X_test.csv')
y_test = pd.read_csv('processed_data/y_test.csv')['is_fraud']

# ============================================================================
# EXAMPLE 1: Predict Single Transaction
# ============================================================================
print(f"\n{'='*100}")
print("EXAMPLE 1: SINGLE TRANSACTION PREDICTION")
print(f"{'='*100}\n")

# Get a fraud transaction
fraud_idx = y_test[y_test == 1].index[0]
transaction = X_test.iloc[fraud_idx:fraud_idx+1]
actual_label = y_test.iloc[fraud_idx]

print(f"Transaction Index: {fraud_idx}")
print(f"Actual Label: {'🚨 FRAUD' if actual_label == 1 else '✅ Legitimate'}")
print(f"\nPredictions from each model:\n")

# XGBoost prediction
xgb_pred = xgb_model.predict(transaction)[0]
xgb_proba = xgb_model.predict_proba(transaction)[0, 1]
print(f"1. XGBoost:")
print(f"   Prediction: {'🚨 FRAUD' if xgb_pred == 1 else '✅ Legitimate'}")
print(f"   Fraud Probability: {xgb_proba*100:.2f}%")
print(f"   Risk Score: {xgb_proba*100:.1f}/100")

# Isolation Forest prediction
iso_pred_raw = iso_model.predict(transaction)[0]
iso_pred = 1 if iso_pred_raw == -1 else 0
iso_score = iso_model.score_samples(transaction)[0]
print(f"\n2. Isolation Forest:")
print(f"   Prediction: {'🚨 FRAUD (Anomaly)' if iso_pred == 1 else '✅ Legitimate (Normal)'}")
print(f"   Anomaly Score: {iso_score:.4f}")

# Random Forest prediction
rf_pred = rf_model.predict(transaction)[0]
rf_proba = rf_model.predict_proba(transaction)[0, 1]
print(f"\n3. Random Forest:")
print(f"   Prediction: {'🚨 FRAUD' if rf_pred == 1 else '✅ Legitimate'}")
print(f"   Fraud Probability: {rf_proba*100:.2f}%")
print(f"   Risk Score: {rf_proba*100:.1f}/100")

# Ensemble prediction
iso_scores = iso_model.score_samples(X_test)
iso_score_norm = (iso_score - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
iso_score_norm = 1 - iso_score_norm

ensemble_proba = 0.70 * xgb_proba + 0.20 * iso_score_norm + 0.10 * rf_proba
ensemble_pred = 1 if ensemble_proba >= 0.5 else 0

print(f"\n4. Ensemble (70% XGB + 20% ISO + 10% RF):")
print(f"   Prediction: {'🚨 FRAUD' if ensemble_pred == 1 else '✅ Legitimate'}")
print(f"   Combined Fraud Probability: {ensemble_proba*100:.2f}%")
print(f"   Risk Score: {ensemble_proba*100:.1f}/100")

# ============================================================================
# EXAMPLE 2: Batch Prediction (10 transactions)
# ============================================================================
print(f"\n{'='*100}")
print("EXAMPLE 2: BATCH PREDICTION (10 Transactions)")
print(f"{'='*100}\n")

# Get 5 fraud and 5 legitimate transactions
fraud_indices = y_test[y_test == 1].index[:5]
legit_indices = y_test[y_test == 0].index[:5]
batch_indices = list(fraud_indices) + list(legit_indices)

batch_transactions = X_test.iloc[batch_indices]
batch_labels = y_test.iloc[batch_indices]

# Predict with XGBoost
xgb_batch_pred = xgb_model.predict(batch_transactions)
xgb_batch_proba = xgb_model.predict_proba(batch_transactions)[:, 1]

# Create results table
results = pd.DataFrame({
    'Transaction_ID': batch_indices,
    'Actual': ['FRAUD' if l == 1 else 'Legitimate' for l in batch_labels],
    'XGBoost_Prediction': ['FRAUD' if p == 1 else 'Legitimate' for p in xgb_batch_pred],
    'Fraud_Probability_%': [f"{p*100:.2f}" for p in xgb_batch_proba],
    'Risk_Score': [f"{p*100:.0f}/100" for p in xgb_batch_proba],
    'Correct': ['✅' if xgb_batch_pred[i] == batch_labels.iloc[i] else '❌' for i in range(len(batch_labels))]
})

print(results.to_string(index=False))

accuracy = (xgb_batch_pred == batch_labels).sum() / len(batch_labels) * 100
print(f"\nBatch Accuracy: {accuracy:.1f}%")

# ============================================================================
# EXAMPLE 3: Risk Level Classification
# ============================================================================
print(f"\n{'='*100}")
print("EXAMPLE 3: RISK LEVEL CLASSIFICATION")
print(f"{'='*100}\n")

# Get 20 random transactions
sample_indices = np.random.choice(X_test.index, 20, replace=False)
sample_transactions = X_test.iloc[sample_indices]

# Predict probabilities
sample_proba = xgb_model.predict_proba(sample_transactions)[:, 1]

# Classify risk levels
def get_risk_level(prob):
    if prob >= 0.80:
        return '🔴 HIGH RISK'
    elif prob >= 0.50:
        return '🟡 MEDIUM RISK'
    else:
        return '🟢 LOW RISK'

risk_classification = pd.DataFrame({
    'Transaction_ID': sample_indices,
    'Fraud_Probability_%': [f"{p*100:.2f}" for p in sample_proba],
    'Risk_Level': [get_risk_level(p) for p in sample_proba],
    'Recommended_Action': [
        'BLOCK & INVESTIGATE' if p >= 0.80 else
        'FLAG FOR REVIEW' if p >= 0.50 else
        'ALLOW'
        for p in sample_proba
    ]
})

# Sort by probability
risk_classification = risk_classification.sort_values('Fraud_Probability_%', ascending=False)

print(risk_classification.to_string(index=False))

# Count by risk level
print(f"\nRisk Distribution:")
print(f"  🔴 High Risk (≥80%): {sum(sample_proba >= 0.80)} transactions")
print(f"  🟡 Medium Risk (50-80%): {sum((sample_proba >= 0.50) & (sample_proba < 0.80))} transactions")
print(f"  🟢 Low Risk (<50%): {sum(sample_proba < 0.50)} transactions")

# ============================================================================
# EXAMPLE 4: Model Comparison on Same Transaction
# ============================================================================
print(f"\n{'='*100}")
print("EXAMPLE 4: MODEL COMPARISON - SAME TRANSACTION")
print(f"{'='*100}\n")

# Get a high-risk transaction
high_risk_idx = y_test[y_test == 1].index[10]
test_transaction = X_test.iloc[high_risk_idx:high_risk_idx+1]

print(f"Transaction ID: {high_risk_idx}")
print(f"Actual Label: {'🚨 FRAUD' if y_test.iloc[high_risk_idx] == 1 else '✅ Legitimate'}\n")

# All model predictions
xgb_p = xgb_model.predict_proba(test_transaction)[0, 1]
rf_p = rf_model.predict_proba(test_transaction)[0, 1]
iso_raw = iso_model.predict(test_transaction)[0]
iso_p = 1.0 if iso_raw == -1 else 0.0

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest', 'Isolation Forest', 'Ensemble'],
    'Fraud_Probability_%': [
        f"{xgb_p*100:.2f}",
        f"{rf_p*100:.2f}",
        f"{iso_p*100:.2f}",
        f"{(0.7*xgb_p + 0.1*rf_p + 0.2*iso_p)*100:.2f}"
    ],
    'Prediction': [
        'FRAUD' if xgb_p >= 0.5 else 'Legitimate',
        'FRAUD' if rf_p >= 0.5 else 'Legitimate',
        'FRAUD' if iso_p >= 0.5 else 'Legitimate',
        'FRAUD' if (0.7*xgb_p + 0.1*rf_p + 0.2*iso_p) >= 0.5 else 'Legitimate'
    ],
    'Confidence': [
        'High' if abs(xgb_p - 0.5) > 0.3 else 'Medium' if abs(xgb_p - 0.5) > 0.1 else 'Low',
        'High' if abs(rf_p - 0.5) > 0.3 else 'Medium' if abs(rf_p - 0.5) > 0.1 else 'Low',
        'High',
        'High' if abs((0.7*xgb_p + 0.1*rf_p + 0.2*iso_p) - 0.5) > 0.3 else 'Medium'
    ]
})

print(comparison.to_string(index=False))

# ============================================================================
# SAVE PREDICTION FUNCTION
# ============================================================================
print(f"\n{'='*100}")
print("CREATING REUSABLE PREDICTION FUNCTION")
print(f"{'='*100}\n")

prediction_function = """
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
"""

with open('predict_fraud.py', 'w') as f:
    f.write(prediction_function)

print("✅ Prediction function saved to: predict_fraud.py")

print(f"\n{'='*100}")
print("✅ PREDICTION EXAMPLES COMPLETE!")
print(f"{'='*100}\n")
print("📝 Summary:")
print("  ✅ Single transaction prediction demonstrated")
print("  ✅ Batch prediction (10 transactions) demonstrated")
print("  ✅ Risk level classification created")
print("  ✅ Model comparison shown")
print("  ✅ Reusable prediction function saved")
