"""
Retrain Existing Models with User Data
Allows users to upload their own data and retrain the production models
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("🔄 RETRAIN EXISTING MODELS WITH YOUR DATA")
print("="*100)

# ============================================================================
# STEP 1: LOAD USER DATA
# ============================================================================
print("\n📂 STEP 1: Upload Your Data")
print("="*100)

print("\nProvide your fraud detection dataset to retrain the existing models.")
print("Your CSV should contain transaction features and a fraud label column.\n")

while True:
    file_path = input("Enter the path to your CSV file: ").strip()
    
    if os.path.exists(file_path):
        try:
            user_data = pd.read_csv(file_path)
            print(f"\n✅ Successfully loaded: {file_path}")
            print(f"   Shape: {user_data.shape}")
            break
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            print("Please try again.\n")
    else:
        print(f"❌ File not found: {file_path}")
        print("Please check the path and try again.\n")

# ============================================================================
# STEP 2: IDENTIFY FRAUD COLUMN
# ============================================================================
print("\n🔍 STEP 2: Identify Fraud Label Column")
print("="*100)

print(f"\nAvailable columns:")
for i, col in enumerate(user_data.columns, 1):
    print(f"  {i:2d}. {col}")

fraud_candidates = [col for col in user_data.columns 
                   if any(keyword in col.lower() for keyword in ['fraud', 'label', 'class', 'target'])]

if fraud_candidates:
    fraud_col = fraud_candidates[0]
    print(f"\n✅ Auto-detected fraud column: '{fraud_col}'")
    confirm = input(f"Is this correct? (y/n): ").strip().lower()
    if confirm != 'y':
        fraud_col = input("Enter the correct fraud column name: ").strip()
else:
    fraud_col = input("\nEnter the name of your fraud label column: ").strip()

if fraud_col not in user_data.columns:
    print(f"❌ Column '{fraud_col}' not found!")
    exit(1)

print(f"\n✅ Using '{fraud_col}' as fraud label")

# ============================================================================
# STEP 3: LOAD EXISTING MODELS & PREPROCESSING ARTIFACTS
# ============================================================================
print("\n📦 STEP 3: Loading Existing Models & Preprocessing Setup")
print("="*100)

try:
    # Load existing models
    xgb_model = joblib.load('models/xgboost_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    iso_model = joblib.load('models/isolation_forest_model.pkl')
    
    # Load preprocessing artifacts
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    print(f"✅ Loaded existing models:")
    print(f"   • XGBoost")
    print(f"   • Random Forest")
    print(f"   • Isolation Forest")
    print(f"✅ Loaded preprocessing setup:")
    print(f"   • Feature scaler")
    print(f"   • Feature names ({len(feature_names)} features)")
    
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    print("Please ensure models are in the 'models/' directory.")
    exit(1)

# ============================================================================
# STEP 4: PREPROCESS USER DATA TO MATCH EXISTING FEATURES
# ============================================================================
print("\n🔧 STEP 4: Preprocessing Your Data")
print("="*100)

# Standardize fraud column
if fraud_col != 'is_fraud':
    user_data['is_fraud'] = user_data[fraud_col]
    user_data = user_data.drop(columns=[fraud_col])

# Convert to binary
if user_data['is_fraud'].dtype == 'object':
    fraud_map = {
        'fraud': 1, 'Fraud': 1, 'FRAUD': 1, 1: 1, '1': 1,
        'legitimate': 0, 'Legitimate': 0, 0: 0, '0': 0
    }
    user_data['is_fraud'] = user_data['is_fraud'].map(fraud_map)

user_data['is_fraud'] = user_data['is_fraud'].astype(int)
user_data = user_data.dropna(subset=['is_fraud'])

print(f"✅ Data cleaned:")
print(f"   Total samples: {len(user_data):,}")
print(f"   Fraud cases: {user_data['is_fraud'].sum():,} ({user_data['is_fraud'].mean()*100:.2f}%)")

# Separate features and target
y = user_data['is_fraud']
X = user_data.drop(columns=['is_fraud'])

# Align features with existing model
print(f"\n🔄 Aligning features with existing model...")
print(f"   Your data has {X.shape[1]} features")
print(f"   Model expects {len(feature_names)} features")

# Create DataFrame with all expected features
X_aligned = pd.DataFrame(0, index=X.index, columns=feature_names)

# Fill in matching features
for col in X.columns:
    if col in feature_names:
        X_aligned[col] = X[col]

# Handle missing values
numeric_cols = X_aligned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X_aligned[col].isnull().sum() > 0:
        X_aligned[col] = X_aligned[col].fillna(X_aligned[col].median())

# Scale features using existing scaler
X_scaled = scaler.transform(X_aligned)
X_final = pd.DataFrame(X_scaled, columns=feature_names, index=X_aligned.index)

print(f"✅ Features aligned and scaled")
print(f"   Final shape: {X_final.shape}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n📊 STEP 5: Splitting Data")
print("="*100)

test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=test_size, random_state=42, stratify=y
)

print(f"✅ Data split:")
print(f"   Training: {len(X_train):,} samples - Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Testing:  {len(X_test):,} samples - Fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# STEP 6: RETRAIN MODELS
# ============================================================================
print("\n🔄 STEP 6: Retraining Models with Your Data")
print("="*100)

print("\nWhich models would you like to retrain?")
print("  1. XGBoost only (fastest)")
print("  2. All 3 models (XGBoost + Random Forest + Isolation Forest)")

model_choice = input("\nEnter choice (1-2): ").strip()

# Backup existing models
backup_dir = 'models/backups'
os.makedirs(backup_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n💾 Backing up existing models to {backup_dir}/")
joblib.dump(xgb_model, f"{backup_dir}/xgboost_model_backup_{timestamp}.pkl")
if model_choice == '2':
    joblib.dump(rf_model, f"{backup_dir}/random_forest_model_backup_{timestamp}.pkl")
    joblib.dump(iso_model, f"{backup_dir}/isolation_forest_model_backup_{timestamp}.pkl")
print("✅ Backups created")

# Retrain XGBoost
print("\n🚀 Retraining XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model_new = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    learning_rate=0.05,
    max_depth=8,
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model_new.fit(X_train, y_train, verbose=False)
print("✅ XGBoost retrained")

models_retrained = {'xgboost': xgb_model_new}

if model_choice == '2':
    # Retrain Random Forest
    print("\n🚀 Retraining Random Forest...")
    rf_model_new = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model_new.fit(X_train, y_train)
    models_retrained['random_forest'] = rf_model_new
    print("✅ Random Forest retrained")
    
    # Retrain Isolation Forest
    print("\n🚀 Retraining Isolation Forest...")
    iso_model_new = IsolationForest(
        n_estimators=100,
        contamination=y_train.mean(),
        max_samples=256,
        random_state=42,
        n_jobs=-1
    )
    iso_model_new.fit(X_train)
    models_retrained['isolation_forest'] = iso_model_new
    print("✅ Isolation Forest retrained")

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n📈 STEP 7: Evaluating Retrained Models")
print("="*100)

results = {}

for model_name, model in models_retrained.items():
    print(f"\n{model_name.upper()} Results:")
    print("-" * 50)
    
    if model_name == 'isolation_forest':
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    
    fraud_detection_rate = (cm[1,1] / (cm[1,0] + cm[1,1]) * 100) if (cm[1,0] + cm[1,1]) > 0 else 0
    false_positive_rate = (cm[0,1] / (cm[0,0] + cm[0,1]) * 100) if (cm[0,0] + cm[0,1]) > 0 else 0
    precision = (cm[1,1] / (cm[1,1] + cm[0,1]) * 100) if (cm[1,1] + cm[0,1]) > 0 else 0
    
    print(f"  Fraud Detection Rate: {fraud_detection_rate:.2f}%")
    print(f"  False Positive Rate:  {false_positive_rate:.2f}%")
    print(f"  Precision:            {precision:.2f}%")
    
    if model_name != 'isolation_forest':
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        print(f"  PR-AUC:              {pr_auc:.4f}")
    
    results[model_name] = {
        'fraud_detection_rate': fraud_detection_rate,
        'false_positive_rate': false_positive_rate,
        'precision': precision
    }

# ============================================================================
# STEP 8: SAVE RETRAINED MODELS
# ============================================================================
print("\n💾 STEP 8: Save Retrained Models")
print("="*100)

save_choice = input("\nReplace existing models with retrained versions? (y/n): ").strip().lower()

if save_choice == 'y':
    # Save retrained models
    joblib.dump(models_retrained['xgboost'], 'models/xgboost_model.pkl')
    print("  ✅ Saved: models/xgboost_model.pkl")
    
    if 'random_forest' in models_retrained:
        joblib.dump(models_retrained['random_forest'], 'models/random_forest_model.pkl')
        print("  ✅ Saved: models/random_forest_model.pkl")
    
    if 'isolation_forest' in models_retrained:
        joblib.dump(models_retrained['isolation_forest'], 'models/isolation_forest_model.pkl')
        print("  ✅ Saved: models/isolation_forest_model.pkl")
    
    # Save training metadata
    metadata = {
        'retrained_timestamp': timestamp,
        'training_data_size': len(user_data),
        'fraud_rate': float(y.mean() * 100),
        'results': results,
        'original_backups': f"{backup_dir}/*_backup_{timestamp}.pkl"
    }
    
    import json
    with open(f'models/retrain_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Saved: models/retrain_metadata_{timestamp}.json")
    
    print(f"\n✅ Models successfully retrained and saved!")
    print(f"   Original models backed up to: {backup_dir}/")
else:
    print("\n⚠️  Models not saved. Original models remain unchanged.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("✅ RETRAINING COMPLETE!")
print("="*100)

print(f"\n📊 Summary:")
print(f"  • Data processed: {len(user_data):,} transactions")
print(f"  • Models retrained: {len(models_retrained)}")
print(f"  • Best fraud detection rate: {max([r['fraud_detection_rate'] for r in results.values()]):.2f}%")

if save_choice == 'y':
    print(f"\n🎯 Next Steps:")
    print(f"  1. Restart your backend server to load the new models")
    print(f"  2. Test predictions with the retrained models")
    print(f"  3. Monitor performance on real transactions")
    print(f"  4. If needed, restore from backups in {backup_dir}/")

print(f"\n{'='*100}\n")
