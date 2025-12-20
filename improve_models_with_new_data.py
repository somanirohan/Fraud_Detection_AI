"""
Incremental Model Training - Add New Data to Existing Models
Combines uploaded data with existing training data to improve models
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
print("📈 INCREMENTAL MODEL TRAINING - IMPROVE EXISTING MODELS WITH NEW DATA")
print("="*100)

# ============================================================================
# STEP 1: LOAD NEW DATA
# ============================================================================
print("\n📂 STEP 1: Upload Your New Data")
print("="*100)

print("\nProvide new transaction data to improve the existing models.")
print("This data will be ADDED to the existing training data.\n")

while True:
    file_path = input("Enter the path to your CSV file: ").strip()
    
    if os.path.exists(file_path):
        try:
            new_data = pd.read_csv(file_path)
            print(f"\n✅ Successfully loaded: {file_path}")
            print(f"   Shape: {new_data.shape}")
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
for i, col in enumerate(new_data.columns, 1):
    print(f"  {i:2d}. {col}")

fraud_candidates = [col for col in new_data.columns 
                   if any(keyword in col.lower() for keyword in ['fraud', 'label', 'class', 'target'])]

if fraud_candidates:
    fraud_col = fraud_candidates[0]
    print(f"\n✅ Auto-detected fraud column: '{fraud_col}'")
    confirm = input(f"Is this correct? (y/n): ").strip().lower()
    if confirm != 'y':
        fraud_col = input("Enter the correct fraud column name: ").strip()
else:
    fraud_col = input("\nEnter the name of your fraud label column: ").strip()

if fraud_col not in new_data.columns:
    print(f"❌ Column '{fraud_col}' not found!")
    exit(1)

# Standardize fraud column
if fraud_col != 'is_fraud':
    new_data['is_fraud'] = new_data[fraud_col]
    new_data = new_data.drop(columns=[fraud_col])

# Convert to binary
if new_data['is_fraud'].dtype == 'object':
    fraud_map = {
        'fraud': 1, 'Fraud': 1, 'FRAUD': 1, 1: 1, '1': 1,
        'legitimate': 0, 'Legitimate': 0, 0: 0, '0': 0
    }
    new_data['is_fraud'] = new_data['is_fraud'].map(fraud_map)

new_data['is_fraud'] = new_data['is_fraud'].astype(int)
new_data = new_data.dropna(subset=['is_fraud'])

print(f"\n✅ New data cleaned:")
print(f"   Total samples: {len(new_data):,}")
print(f"   Fraud cases: {new_data['is_fraud'].sum():,} ({new_data['is_fraud'].mean()*100:.2f}%)")

# ============================================================================
# STEP 3: LOAD EXISTING TRAINING DATA
# ============================================================================
print("\n📦 STEP 3: Loading Existing Training Data")
print("="*100)

try:
    X_train_old = pd.read_csv('processed_data/X_train.csv')
    y_train_old = pd.read_csv('processed_data/y_train.csv')['is_fraud']
    
    print(f"✅ Loaded existing training data:")
    print(f"   Samples: {len(X_train_old):,}")
    print(f"   Fraud cases: {y_train_old.sum():,} ({y_train_old.mean()*100:.2f}%)")
    print(f"   Features: {X_train_old.shape[1]}")
    
except Exception as e:
    print(f"❌ Error loading existing training data: {str(e)}")
    print("Please ensure processed_data/ directory exists with training data.")
    exit(1)

# ============================================================================
# STEP 4: LOAD PREPROCESSING ARTIFACTS
# ============================================================================
print("\n🔧 STEP 4: Loading Preprocessing Setup")
print("="*100)

try:
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    print(f"✅ Loaded preprocessing artifacts:")
    print(f"   Feature scaler")
    print(f"   Feature names ({len(feature_names)} features)")
    
except Exception as e:
    print(f"❌ Error loading preprocessing artifacts: {str(e)}")
    exit(1)

# ============================================================================
# STEP 5: ALIGN NEW DATA WITH EXISTING FEATURES
# ============================================================================
print("\n🔄 STEP 5: Aligning New Data with Existing Features")
print("="*100)

# Separate features and target from new data
y_new = new_data['is_fraud']
X_new = new_data.drop(columns=['is_fraud'])

# Create DataFrame with all expected features
X_new_aligned = pd.DataFrame(0, index=X_new.index, columns=feature_names)

# Fill in matching features
for col in X_new.columns:
    if col in feature_names:
        X_new_aligned[col] = X_new[col]

# Handle missing values
numeric_cols = X_new_aligned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X_new_aligned[col].isnull().sum() > 0:
        X_new_aligned[col] = X_new_aligned[col].fillna(X_new_aligned[col].median())

print(f"✅ New data aligned:")
print(f"   Shape: {X_new_aligned.shape}")

# ============================================================================
# STEP 6: COMBINE OLD AND NEW DATA
# ============================================================================
print("\n➕ STEP 6: Combining Old and New Training Data")
print("="*100)

# Combine features
X_combined = pd.concat([X_train_old, X_new_aligned], ignore_index=True)
y_combined = pd.concat([y_train_old, y_new], ignore_index=True)

print(f"✅ Combined training data:")
print(f"   Total samples: {len(X_combined):,}")
print(f"   Old data: {len(X_train_old):,} samples")
print(f"   New data: {len(X_new_aligned):,} samples")
print(f"   Fraud cases: {y_combined.sum():,} ({y_combined.mean()*100:.2f}%)")

# ============================================================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================================================
print("\n📊 STEP 7: Creating Validation Set")
print("="*100)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.20, random_state=42, stratify=y_combined
)

print(f"✅ Data split:")
print(f"   Training: {len(X_train):,} samples - Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Validation: {len(X_test):,} samples - Fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# STEP 8: RETRAIN MODELS
# ============================================================================
print("\n🔄 STEP 8: Retraining Models with Combined Data")
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
xgb_model_old = joblib.load('models/xgboost_model.pkl')
joblib.dump(xgb_model_old, f"{backup_dir}/xgboost_model_backup_{timestamp}.pkl")

if model_choice == '2':
    rf_model_old = joblib.load('models/random_forest_model.pkl')
    iso_model_old = joblib.load('models/isolation_forest_model.pkl')
    joblib.dump(rf_model_old, f"{backup_dir}/random_forest_model_backup_{timestamp}.pkl")
    joblib.dump(iso_model_old, f"{backup_dir}/isolation_forest_model_backup_{timestamp}.pkl")

print("✅ Backups created")

# Retrain XGBoost
print("\n🚀 Retraining XGBoost with combined data...")
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
# STEP 9: EVALUATION
# ============================================================================
print("\n📈 STEP 9: Evaluating Improved Models")
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
# STEP 10: SAVE IMPROVED MODELS
# ============================================================================
print("\n💾 STEP 10: Save Improved Models")
print("="*100)

save_choice = input("\nReplace existing models with improved versions? (y/n): ").strip().lower()

if save_choice == 'y':
    # Save improved models
    joblib.dump(models_retrained['xgboost'], 'models/xgboost_model.pkl')
    print("  ✅ Saved: models/xgboost_model.pkl")
    
    if 'random_forest' in models_retrained:
        joblib.dump(models_retrained['random_forest'], 'models/random_forest_model.pkl')
        print("  ✅ Saved: models/random_forest_model.pkl")
    
    if 'isolation_forest' in models_retrained:
        joblib.dump(models_retrained['isolation_forest'], 'models/isolation_forest_model.pkl')
        print("  ✅ Saved: models/isolation_forest_model.pkl")
    
    # Update training data
    X_combined.to_csv('processed_data/X_train.csv', index=False)
    pd.DataFrame({'is_fraud': y_combined}).to_csv('processed_data/y_train.csv', index=False)
    print("  ✅ Updated: processed_data/X_train.csv")
    print("  ✅ Updated: processed_data/y_train.csv")
    
    # Save metadata
    metadata = {
        'improvement_timestamp': timestamp,
        'old_training_size': len(X_train_old),
        'new_data_added': len(X_new_aligned),
        'total_training_size': len(X_combined),
        'fraud_rate': float(y_combined.mean() * 100),
        'results': results,
        'original_backups': f"{backup_dir}/*_backup_{timestamp}.pkl"
    }
    
    import json
    with open(f'models/improvement_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Saved: models/improvement_metadata_{timestamp}.json")
    
    print(f"\n✅ Models successfully improved and saved!")
    print(f"   Original models backed up to: {backup_dir}/")
    print(f"   Training data updated with {len(X_new_aligned):,} new samples")
else:
    print("\n⚠️  Models not saved. Original models remain unchanged.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("✅ INCREMENTAL TRAINING COMPLETE!")
print("="*100)

print(f"\n📊 Summary:")
print(f"  • Old training data: {len(X_train_old):,} samples")
print(f"  • New data added: {len(X_new_aligned):,} samples")
print(f"  • Total training data: {len(X_combined):,} samples")
print(f"  • Models retrained: {len(models_retrained)}")
print(f"  • Best fraud detection rate: {max([r['fraud_detection_rate'] for r in results.values()]):.2f}%")

if save_choice == 'y':
    print(f"\n🎯 Next Steps:")
    print(f"  1. Test predictions with improved models")
    print(f"  2. Monitor performance on new transactions")
    print(f"  3. Continue adding more data to further improve")
    print(f"  4. If needed, restore from backups in {backup_dir}/")

print(f"\n{'='*100}\n")
