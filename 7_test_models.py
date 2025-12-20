"""
1. Test All Models on Test Set
Comprehensive evaluation with fraud detection rate analysis
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPREHENSIVE MODEL TESTING ON TEST SET")
print("="*100)

# Load test data
print("\n📂 Loading test data...")
X_test = pd.read_csv('processed_data/X_test.csv')
y_test = pd.read_csv('processed_data/y_test.csv')['is_fraud']

print(f"✅ Test Set: {X_test.shape}")
print(f"   Total Samples: {len(y_test):,}")
print(f"   Fraud Cases: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
print(f"   Legitimate Cases: {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")

# Load models
print("\n📦 Loading trained models...")
xgb_model = joblib.load('models/xgboost_model.pkl')
iso_model = joblib.load('models/isolation_forest_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
print("✅ All models loaded")

# ============================================================================
# TEST MODEL 1: XGBOOST
# ============================================================================
print(f"\n{'='*100}")
print("TESTING MODEL 1: XGBOOST")
print(f"{'='*100}\n")

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate', 'Fraud']))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_xgb[0,0]:>8,}")
print(f"  False Positives: {cm_xgb[0,1]:>8,}")
print(f"  False Negatives: {cm_xgb[1,0]:>8,}")
print(f"  True Positives:  {cm_xgb[1,1]:>8,}")

# Fraud detection rate
xgb_fraud_rate = (cm_xgb[1,1] / (cm_xgb[1,0] + cm_xgb[1,1]) * 100)
xgb_fpr = (cm_xgb[0,1] / (cm_xgb[0,0] + cm_xgb[0,1]) * 100)

precision, recall, _ = precision_recall_curve(y_test, y_proba_xgb)
xgb_pr_auc = auc(recall, precision)

print(f"\n🎯 Performance Metrics:")
print(f"   PR-AUC: {xgb_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {xgb_fraud_rate:.2f}%")
print(f"   False Positive Rate: {xgb_fpr:.2f}%")

# ============================================================================
# TEST MODEL 2: ISOLATION FOREST
# ============================================================================
print(f"\n{'='*100}")
print("TESTING MODEL 2: ISOLATION FOREST")
print(f"{'='*100}\n")

y_pred_iso_raw = iso_model.predict(X_test)
y_pred_iso = (y_pred_iso_raw == -1).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_iso, target_names=['Legitimate', 'Fraud']))

cm_iso = confusion_matrix(y_test, y_pred_iso)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_iso[0,0]:>8,}")
print(f"  False Positives: {cm_iso[0,1]:>8,}")
print(f"  False Negatives: {cm_iso[1,0]:>8,}")
print(f"  True Positives:  {cm_iso[1,1]:>8,}")

iso_fraud_rate = (cm_iso[1,1] / (cm_iso[1,0] + cm_iso[1,1]) * 100)
iso_fpr = (cm_iso[0,1] / (cm_iso[0,0] + cm_iso[0,1]) * 100)

print(f"\n🎯 Performance Metrics:")
print(f"   Fraud Detection Rate: {iso_fraud_rate:.2f}%")
print(f"   False Positive Rate: {iso_fpr:.2f}%")

# ============================================================================
# TEST MODEL 3: RANDOM FOREST
# ============================================================================
print(f"\n{'='*100}")
print("TESTING MODEL 3: RANDOM FOREST")
print(f"{'='*100}\n")

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraud']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_rf[0,0]:>8,}")
print(f"  False Positives: {cm_rf[0,1]:>8,}")
print(f"  False Negatives: {cm_rf[1,0]:>8,}")
print(f"  True Positives:  {cm_rf[1,1]:>8,}")

rf_fraud_rate = (cm_rf[1,1] / (cm_rf[1,0] + cm_rf[1,1]) * 100)
rf_fpr = (cm_rf[0,1] / (cm_rf[0,0] + cm_rf[0,1]) * 100)

precision, recall, _ = precision_recall_curve(y_test, y_proba_rf)
rf_pr_auc = auc(recall, precision)

print(f"\n🎯 Performance Metrics:")
print(f"   PR-AUC: {rf_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {rf_fraud_rate:.2f}%")
print(f"   False Positive Rate: {rf_fpr:.2f}%")

# ============================================================================
# TEST ENSEMBLE MODEL
# ============================================================================
print(f"\n{'='*100}")
print("TESTING ENSEMBLE MODEL")
print(f"{'='*100}\n")

# Normalize Isolation Forest scores
iso_scores = iso_model.score_samples(X_test)
iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
iso_scores_norm = 1 - iso_scores_norm

# Ensemble prediction
ensemble_proba = (0.70 * y_proba_xgb + 0.20 * iso_scores_norm + 0.10 * y_proba_rf)
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Fraud']))

cm_ensemble = confusion_matrix(y_test, ensemble_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm_ensemble[0,0]:>8,}")
print(f"  False Positives: {cm_ensemble[0,1]:>8,}")
print(f"  False Negatives: {cm_ensemble[1,0]:>8,}")
print(f"  True Positives:  {cm_ensemble[1,1]:>8,}")

ensemble_fraud_rate = (cm_ensemble[1,1] / (cm_ensemble[1,0] + cm_ensemble[1,1]) * 100)
ensemble_fpr = (cm_ensemble[0,1] / (cm_ensemble[0,0] + cm_ensemble[0,1]) * 100)

precision, recall, _ = precision_recall_curve(y_test, ensemble_proba)
ensemble_pr_auc = auc(recall, precision)

print(f"\n🎯 Performance Metrics:")
print(f"   PR-AUC: {ensemble_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {ensemble_fraud_rate:.2f}%")
print(f"   False Positive Rate: {ensemble_fpr:.2f}%")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print(f"\n{'='*100}")
print("FINAL TEST SET COMPARISON - ALL MODELS")
print(f"{'='*100}\n")

test_results = pd.DataFrame({
    'Model': ['XGBoost', 'Isolation Forest', 'Random Forest', 'Ensemble'],
    'PR-AUC': [xgb_pr_auc, 0, rf_pr_auc, ensemble_pr_auc],
    'Fraud Detection Rate (%)': [xgb_fraud_rate, iso_fraud_rate, rf_fraud_rate, ensemble_fraud_rate],
    'False Positive Rate (%)': [xgb_fpr, iso_fpr, rf_fpr, ensemble_fpr],
    'True Positives': [cm_xgb[1,1], cm_iso[1,1], cm_rf[1,1], cm_ensemble[1,1]],
    'False Negatives': [cm_xgb[1,0], cm_iso[1,0], cm_rf[1,0], cm_ensemble[1,0]],
    'Precision (%)': [
        cm_xgb[1,1] / (cm_xgb[1,1] + cm_xgb[0,1]) * 100,
        cm_iso[1,1] / (cm_iso[1,1] + cm_iso[0,1]) * 100,
        cm_rf[1,1] / (cm_rf[1,1] + cm_rf[0,1]) * 100,
        cm_ensemble[1,1] / (cm_ensemble[1,1] + cm_ensemble[0,1]) * 100
    ]
})

print(test_results.to_string(index=False))

# Save results
test_results.to_csv('models/test_results.csv', index=False)
joblib.dump(test_results, 'models/test_results.pkl')

print(f"\n{'='*100}")
print("✅ TEST SET EVALUATION COMPLETE!")
print(f"{'='*100}")
print(f"\n🏆 Best Model on Test Set: {test_results.loc[test_results['Fraud Detection Rate (%)'].idxmax(), 'Model']}")
print(f"   Fraud Detection Rate: {test_results['Fraud Detection Rate (%)'].max():.2f}%")
print(f"\n💾 Results saved to: models/test_results.csv")
