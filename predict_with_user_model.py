"""
Quick Prediction Script for User-Trained Models
Use your custom trained models to make predictions
"""

import pandas as pd
import joblib
import os
from datetime import datetime

print("="*100)
print("🔮 FRAUD PREDICTION - USER MODELS")
print("="*100)

# List available user models
user_models_dir = 'user_models'
if not os.path.exists(user_models_dir):
    print(f"\n❌ No user models found in '{user_models_dir}/'")
    print("Please train a model first using 'train_with_user_data.py'")
    exit(1)

# Find available models
model_files = [f for f in os.listdir(user_models_dir) if f.endswith('.pkl') and 'xgboost' in f]

if not model_files:
    print(f"\n❌ No trained models found")
    exit(1)

print(f"\n📦 Available models:")
for i, model_file in enumerate(model_files, 1):
    print(f"  {i}. {model_file}")

# Select model
if len(model_files) == 1:
    selected_model = model_files[0]
    print(f"\n✅ Using: {selected_model}")
else:
    choice = int(input(f"\nSelect model (1-{len(model_files)}): ").strip())
    selected_model = model_files[choice - 1]

# Extract timestamp
timestamp = selected_model.split('_')[-1].replace('.pkl', '')

# Load model and scaler
print(f"\n📂 Loading model...")
model = joblib.load(f"{user_models_dir}/{selected_model}")
scaler = joblib.load(f"{user_models_dir}/scaler_user_{timestamp}.pkl")
features = joblib.load(f"{user_models_dir}/features_user_{timestamp}.pkl")

print(f"✅ Model loaded with {len(features)} features")

# Get prediction data
print(f"\n📊 Prediction Options:")
print("  1. Predict from CSV file")
print("  2. Enter transaction manually")

pred_choice = input("\nEnter choice (1-2): ").strip()

if pred_choice == '1':
    # Load CSV
    csv_path = input("\nEnter path to CSV file: ").strip()
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        exit(1)
    
    pred_data = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(pred_data):,} transactions")
    
    # Ensure same features
    for feat in features:
        if feat not in pred_data.columns:
            pred_data[feat] = 0
    
    pred_data = pred_data[features]
    
    # Predict
    predictions = model.predict(pred_data)
    probabilities = model.predict_proba(pred_data)[:, 1]
    
    # Add results
    pred_data['prediction'] = ['FRAUD' if p == 1 else 'Legitimate' for p in predictions]
    pred_data['fraud_probability_%'] = probabilities * 100
    pred_data['risk_level'] = ['HIGH' if p >= 0.8 else 'MEDIUM' if p >= 0.5 else 'LOW' for p in probabilities]
    
    # Save results
    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pred_data.to_csv(output_file, index=False)
    
    print(f"\n✅ Predictions saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total: {len(predictions):,}")
    print(f"  Fraud: {sum(predictions):,} ({sum(predictions)/len(predictions)*100:.2f}%)")
    print(f"  Legitimate: {len(predictions)-sum(predictions):,}")

else:
    print("\n⚠️  Manual entry requires all {len(features)} features")
    print("This feature is best used with CSV upload")

print(f"\n{'='*100}\n")
