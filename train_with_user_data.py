"""
User Data Upload and Training Script
Allows users to upload their own data and train fraud detection models
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
print("🚀 FRAUD DETECTION - USER DATA UPLOAD & TRAINING")
print("="*100)

# ============================================================================
# STEP 1: USER DATA INPUT
# ============================================================================
print("\n📂 STEP 1: Upload Your Data")
print("="*100)

print("\nPlease provide your fraud detection dataset.")
print("Your CSV file should contain:")
print("  • Transaction features (amount, time, location, etc.)")
print("  • A fraud label column (e.g., 'is_fraud', 'fraud', 'label', 'class')")
print("  • Values: 0 = Legitimate, 1 = Fraud (or similar binary labels)\n")

# Get file path from user
while True:
    file_path = input("Enter the path to your CSV file (or 'demo' to use sample data): ").strip()
    
    if file_path.lower() == 'demo':
        # Use existing test data as demo
        print("\n✅ Using demo data from test set...")
        X_test = pd.read_csv('processed_data/X_test.csv')
        y_test = pd.read_csv('processed_data/y_test.csv')['is_fraud']
        user_data = X_test.copy()
        user_data['is_fraud'] = y_test
        print(f"   Loaded {len(user_data):,} demo transactions")
        break
    elif os.path.exists(file_path):
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

print(f"\nAvailable columns in your data:")
for i, col in enumerate(user_data.columns, 1):
    print(f"  {i:2d}. {col}")

# Auto-detect fraud column
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

# Validate fraud column
if fraud_col not in user_data.columns:
    print(f"❌ Column '{fraud_col}' not found!")
    exit(1)

print(f"\n✅ Using '{fraud_col}' as fraud label")
print(f"   Unique values: {user_data[fraud_col].unique()}")
print(f"   Value counts:\n{user_data[fraud_col].value_counts()}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n🔧 STEP 3: Preprocessing Your Data")
print("="*100)

# Standardize fraud column
if fraud_col != 'is_fraud':
    user_data['is_fraud'] = user_data[fraud_col]
    user_data = user_data.drop(columns=[fraud_col])

# Convert to binary
if user_data['is_fraud'].dtype == 'object':
    fraud_map = {
        'fraud': 1, 'Fraud': 1, 'FRAUD': 1, 1: 1, '1': 1, 'True': 1, 'true': 1, 'YES': 1, 'yes': 1,
        'legitimate': 0, 'Legitimate': 0, 0: 0, '0': 0, 'False': 0, 'false': 0, 'NO': 0, 'no': 0
    }
    user_data['is_fraud'] = user_data['is_fraud'].map(fraud_map)

user_data['is_fraud'] = user_data['is_fraud'].astype(int)

# Remove rows with null fraud labels
initial_rows = len(user_data)
user_data = user_data.dropna(subset=['is_fraud'])
if len(user_data) < initial_rows:
    print(f"⚠️  Removed {initial_rows - len(user_data)} rows with null fraud labels")

print(f"\n✅ Data cleaned:")
print(f"   Total samples: {len(user_data):,}")
print(f"   Fraud cases: {user_data['is_fraud'].sum():,} ({user_data['is_fraud'].mean()*100:.2f}%)")
print(f"   Legitimate cases: {(user_data['is_fraud']==0).sum():,} ({(user_data['is_fraud']==0).mean()*100:.2f}%)")

# Separate features and target
y = user_data['is_fraud']
X = user_data.drop(columns=['is_fraud'])

# Identify column types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Feature types:")
print(f"   Numeric features: {len(numeric_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Remove high-cardinality ID columns
id_patterns = ['id', 'ID', '_id', 'upi', 'transaction', 'customer']
cols_to_drop = []
for col in categorical_cols:
    if any(p in col.lower() for p in id_patterns):
        if X[col].nunique() > 100:
            cols_to_drop.append(col)

if cols_to_drop:
    print(f"\n🗑️  Dropping {len(cols_to_drop)} high-cardinality ID columns:")
    for col in cols_to_drop:
        print(f"   • {col} ({X[col].nunique()} unique values)")
    X = X.drop(columns=cols_to_drop)
    categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]

# Handle missing values
print(f"\n🔧 Handling missing values...")
for col in numeric_cols:
    if col in X.columns and X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
        print(f"   • Filled {col} with median")

for col in categorical_cols:
    if col in X.columns and X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        print(f"   • Filled {col} with mode")

# One-hot encode categorical features
if categorical_cols:
    print(f"\n🔄 Encoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        if col in X.columns:
            # Keep only top 10 categories
            top_cats = X[col].value_counts().head(10).index.tolist()
            X[col] = X[col].apply(lambda x: x if x in top_cats else 'Other')
    
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=np.int8)
    print(f"   ✅ Encoded to {X.shape[1]} features")

# Scale numeric features
numeric_cols_current = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[numeric_cols_current] = scaler.fit_transform(X[numeric_cols_current])

print(f"\n✅ Preprocessing complete!")
print(f"   Final features: {X.shape[1]}")
print(f"   Final samples: {len(X):,}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n📊 STEP 4: Splitting Data")
print("="*100)

# Ask user for split ratio
print("\nHow would you like to split your data?")
print("  1. 70% train, 30% test (recommended)")
print("  2. 80% train, 20% test")
print("  3. Custom split")

split_choice = input("\nEnter choice (1-3): ").strip()

if split_choice == '2':
    test_size = 0.20
elif split_choice == '3':
    test_size = float(input("Enter test size (e.g., 0.25 for 25%): ").strip())
else:
    test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

print(f"\n✅ Data split:")
print(f"   Training: {len(X_train):,} samples ({(1-test_size)*100:.0f}%) - Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Testing:  {len(X_test):,} samples ({test_size*100:.0f}%) - Fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================
print("\n🤖 STEP 5: Training Models")
print("="*100)

print("\nWhich models would you like to train?")
print("  1. XGBoost only (fastest, recommended)")
print("  2. All 3 models (XGBoost + Random Forest + Isolation Forest)")

model_choice = input("\nEnter choice (1-2): ").strip()

models_trained = {}

# Train XGBoost
print("\n🚀 Training XGBoost...")
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
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)
models_trained['xgboost'] = xgb_model
print("✅ XGBoost trained")

if model_choice == '2':
    # Train Random Forest
    print("\n🚀 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    models_trained['random_forest'] = rf_model
    print("✅ Random Forest trained")
    
    # Train Isolation Forest
    print("\n🚀 Training Isolation Forest...")
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=y_train.mean(),
        max_samples=256,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_train)
    models_trained['isolation_forest'] = iso_model
    print("✅ Isolation Forest trained")

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================
print("\n📈 STEP 6: Evaluating Models")
print("="*100)

results = {}

for model_name, model in models_trained.items():
    print(f"\n{model_name.upper()} Results:")
    print("-" * 50)
    
    if model_name == 'isolation_forest':
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Metrics
    fraud_detection_rate = (cm[1,1] / (cm[1,0] + cm[1,1]) * 100) if (cm[1,0] + cm[1,1]) > 0 else 0
    false_positive_rate = (cm[0,1] / (cm[0,0] + cm[0,1]) * 100) if (cm[0,0] + cm[0,1]) > 0 else 0
    precision = (cm[1,1] / (cm[1,1] + cm[0,1]) * 100) if (cm[1,1] + cm[0,1]) > 0 else 0
    
    print(f"  Fraud Detection Rate: {fraud_detection_rate:.2f}%")
    print(f"  False Positive Rate:  {false_positive_rate:.2f}%")
    print(f"  Precision:            {precision:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0,0]:,}")
    print(f"    False Positives: {cm[0,1]:,}")
    print(f"    False Negatives: {cm[1,0]:,}")
    print(f"    True Positives:  {cm[1,1]:,}")
    
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
# STEP 7: SAVE MODELS
# ============================================================================
print("\n💾 STEP 7: Save Trained Models")
print("="*100)

save_choice = input("\nWould you like to save the trained models? (y/n): ").strip().lower()

if save_choice == 'y':
    # Create user models directory
    user_models_dir = 'user_models'
    os.makedirs(user_models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model in models_trained.items():
        model_file = f"{user_models_dir}/{model_name}_user_{timestamp}.pkl"
        joblib.dump(model, model_file)
        print(f"  ✅ Saved: {model_file}")
    
    # Save scaler
    scaler_file = f"{user_models_dir}/scaler_user_{timestamp}.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"  ✅ Saved: {scaler_file}")
    
    # Save feature names
    feature_file = f"{user_models_dir}/features_user_{timestamp}.pkl"
    joblib.dump(list(X.columns), feature_file)
    print(f"  ✅ Saved: {feature_file}")
    
    # Save results
    results_file = f"{user_models_dir}/results_user_{timestamp}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'data_shape': {'samples': len(user_data), 'features': X.shape[1]},
            'fraud_rate': float(y.mean() * 100),
            'models': results
        }, f, indent=2)
    print(f"  ✅ Saved: {results_file}")
    
    print(f"\n✅ All models saved to: {user_models_dir}/")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================
print("\n" + "="*100)
print("✅ TRAINING COMPLETE!")
print("="*100)

print(f"\n📊 Summary:")
print(f"  • Data processed: {len(user_data):,} transactions")
print(f"  • Features created: {X.shape[1]}")
print(f"  • Models trained: {len(models_trained)}")
print(f"  • Best fraud detection rate: {max([r['fraud_detection_rate'] for r in results.values()]):.2f}%")

print(f"\n🎯 Next Steps:")
print(f"  1. Use your trained models for predictions")
print(f"  2. Fine-tune hyperparameters for better performance")
print(f"  3. Add more training data to improve accuracy")

print(f"\n{'='*100}\n")
