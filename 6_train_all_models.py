"""
Train All 3 Models and Create Comprehensive Evaluation
With Fraud Detection Rate Analysis in Percentages
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPREHENSIVE ML MODEL TRAINING - ALL 3 MODELS")
print("="*100)

# Load data
print("\n📂 Loading data...")
X_train = pd.read_csv('processed_data/X_train.csv')
y_train = pd.read_csv('processed_data/y_train.csv')['is_fraud']
X_val = pd.read_csv('processed_data/X_val.csv')
y_val = pd.read_csv('processed_data/y_val.csv')['is_fraud']
X_test = pd.read_csv('processed_data/X_test.csv')
y_test = pd.read_csv('processed_data/y_test.csv')['is_fraud']

print(f"✅ Train: {X_train.shape} - Fraud: {y_train.mean()*100:.2f}%")
print(f"✅ Val:   {X_val.shape} - Fraud: {y_val.mean()*100:.2f}%")
print(f"✅ Test:  {X_test.shape} - Fraud: {y_test.mean()*100:.2f}%")

# ============================================================================
# MODEL 1: XGBOOST
# ============================================================================
print(f"\n{'='*100}")
print("MODEL 1: XGBOOST - PRIMARY CLASSIFIER")
print(f"{'='*100}\n")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    learning_rate=0.05,
    max_depth=8,
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_val_pred_xgb = xgb_model.predict(X_val)
y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

precision, recall, _ = precision_recall_curve(y_val, y_val_proba_xgb)
xgb_pr_auc = auc(recall, precision)

cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
xgb_fraud_rate = (cm_xgb[1,1] / (cm_xgb[1,0] + cm_xgb[1,1]) * 100) if (cm_xgb[1,0] + cm_xgb[1,1]) > 0 else 0
xgb_fpr = (cm_xgb[0,1] / (cm_xgb[0,0] + cm_xgb[0,1]) * 100) if (cm_xgb[0,0] + cm_xgb[0,1]) > 0 else 0

print(f"✅ XGBoost trained - PR-AUC: {xgb_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {xgb_fraud_rate:.2f}%")
print(f"   False Positive Rate: {xgb_fpr:.2f}%")

joblib.dump(xgb_model, 'models/xgboost_model.pkl')

# ============================================================================
# MODEL 2: ISOLATION FOREST
# ============================================================================
print(f"\n{'='*100}")
print("MODEL 2: ISOLATION FOREST - ANOMALY DETECTOR")
print(f"{'='*100}\n")

iso_model = IsolationForest(
    n_estimators=100,
    contamination=y_train.mean(),
    max_samples=256,
    random_state=42,
    n_jobs=-1
)

print("Training Isolation Forest...")
iso_model.fit(X_train)

y_val_pred_iso_raw = iso_model.predict(X_val)
y_val_pred_iso = (y_val_pred_iso_raw == -1).astype(int)

cm_iso = confusion_matrix(y_val, y_val_pred_iso)
iso_fraud_rate = (cm_iso[1,1] / (cm_iso[1,0] + cm_iso[1,1]) * 100) if (cm_iso[1,0] + cm_iso[1,1]) > 0 else 0
iso_fpr = (cm_iso[0,1] / (cm_iso[0,0] + cm_iso[0,1]) * 100) if (cm_iso[0,0] + cm_iso[0,1]) > 0 else 0

print(f"✅ Isolation Forest trained")
print(f"   Fraud Detection Rate: {iso_fraud_rate:.2f}%")
print(f"   False Positive Rate: {iso_fpr:.2f}%")

joblib.dump(iso_model, 'models/isolation_forest_model.pkl')

# ============================================================================
# MODEL 3: RANDOM FOREST
# ============================================================================
print(f"\n{'='*100}")
print("MODEL 3: RANDOM FOREST - BASELINE CLASSIFIER")
print(f"{'='*100}\n")

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
y_val_proba_rf = rf_model.predict_proba(X_val)[:, 1]

precision, recall, _ = precision_recall_curve(y_val, y_val_proba_rf)
rf_pr_auc = auc(recall, precision)

cm_rf = confusion_matrix(y_val, y_val_pred_rf)
rf_fraud_rate = (cm_rf[1,1] / (cm_rf[1,0] + cm_rf[1,1]) * 100) if (cm_rf[1,0] + cm_rf[1,1]) > 0 else 0
rf_fpr = (cm_rf[0,1] / (cm_rf[0,0] + cm_rf[0,1]) * 100) if (cm_rf[0,0] + cm_rf[0,1]) > 0 else 0

print(f"✅ Random Forest trained - PR-AUC: {rf_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {rf_fraud_rate:.2f}%")
print(f"   False Positive Rate: {rf_fpr:.2f}%")

joblib.dump(rf_model, 'models/random_forest_model.pkl')

# ============================================================================
# ENSEMBLE MODEL
# ============================================================================
print(f"\n{'='*100}")
print("ENSEMBLE MODEL - WEIGHTED VOTING (70% XGB, 20% ISO, 10% RF)")
print(f"{'='*100}\n")

# Normalize Isolation Forest scores to 0-1 range
iso_scores = iso_model.score_samples(X_val)
iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
iso_scores_norm = 1 - iso_scores_norm  # Invert so higher = more fraudulent

# Ensemble prediction
ensemble_proba = (0.70 * y_val_proba_xgb + 
                  0.20 * iso_scores_norm + 
                  0.10 * y_val_proba_rf)

ensemble_pred = (ensemble_proba >= 0.5).astype(int)

cm_ensemble = confusion_matrix(y_val, ensemble_pred)
ensemble_fraud_rate = (cm_ensemble[1,1] / (cm_ensemble[1,0] + cm_ensemble[1,1]) * 100) if (cm_ensemble[1,0] + cm_ensemble[1,1]) > 0 else 0
ensemble_fpr = (cm_ensemble[0,1] / (cm_ensemble[0,0] + cm_ensemble[0,1]) * 100) if (cm_ensemble[0,0] + cm_ensemble[0,1]) > 0 else 0

precision, recall, _ = precision_recall_curve(y_val, ensemble_proba)
ensemble_pr_auc = auc(recall, precision)

print(f"✅ Ensemble Model - PR-AUC: {ensemble_pr_auc:.4f}")
print(f"   Fraud Detection Rate: {ensemble_fraud_rate:.2f}%")
print(f"   False Positive Rate: {ensemble_fpr:.2f}%")

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print(f"\n{'='*100}")
print("COMPREHENSIVE MODEL COMPARISON - FRAUD DETECTION RATE ANALYSIS")
print(f"{'='*100}\n")

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'Isolation Forest', 'Random Forest', 'Ensemble'],
    'PR-AUC': [xgb_pr_auc, 0, rf_pr_auc, ensemble_pr_auc],
    'Fraud Detection Rate (%)': [xgb_fraud_rate, iso_fraud_rate, rf_fraud_rate, ensemble_fraud_rate],
    'False Positive Rate (%)': [xgb_fpr, iso_fpr, rf_fpr, ensemble_fpr],
    'True Positives': [cm_xgb[1,1], cm_iso[1,1], cm_rf[1,1], cm_ensemble[1,1]],
    'False Negatives': [cm_xgb[1,0], cm_iso[1,0], cm_rf[1,0], cm_ensemble[1,0]]
})

print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv('models/model_comparison.csv', index=False)
joblib.dump(comparison, 'models/model_comparison.pkl')

print(f"\n{'='*100}")
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print(f"{'='*100}")
print(f"\n📊 Best Model: {comparison.loc[comparison['Fraud Detection Rate (%)'].idxmax(), 'Model']}")
print(f"   Fraud Detection Rate: {comparison['Fraud Detection Rate (%)'].max():.2f}%")
print(f"\n💾 All models saved to: models/")
print(f"📈 Comparison saved to: models/model_comparison.csv")
