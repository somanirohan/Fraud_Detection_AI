"""
Memory-Efficient Preprocessing - Select Best Datasets
Focus on largest, highest-quality datasets
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

print("="*100)
print("MEMORY-EFFICIENT PREPROCESSING - BEST DATASETS ONLY")
print("="*100)

# Load dataset info
all_dataframes = joblib.load('processed_data/loaded_datasets.pkl')

# Select top datasets by size and quality
selected_datasets = [
    'dataset.csv',  # 1M+ rows
    'dataset (1).csv',  # 200K+ rows
    'upi_transactions_2024.csv',  # 250K rows
    'fraud_dataset.csv',  # 26K rows
    'Fraud Detection Dataset.csv',  # 10K rows
]

print(f"\n📊 Processing {len(selected_datasets)} high-quality datasets\n")

combined_data = []

for item in all_dataframes:
    if item['name'] in selected_datasets:
        name = item['name']
        df = item['data'].copy()
        fraud_col = item['fraud_col']
        
        print(f"Processing: {name}")
        print(f"  Shape: {df.shape}")
        
        # Standardize fraud column
        if fraud_col != 'is_fraud':
            df['is_fraud'] = df[fraud_col]
        
        # Convert to binary
        if df['is_fraud'].dtype == 'object':
            fraud_map = {'fraud': 1, 'Fraud': 1, 1: 1, '1': 1, 'True': 1, 'true': 1,
                        'legitimate': 0, 'Legitimate': 0, 0: 0, '0': 0, 'False': 0, 'false': 0}
            df['is_fraud'] = df['is_fraud'].map(fraud_map)
        
        df['is_fraud'] = df['is_fraud'].astype(int)
        df = df.dropna(subset=['is_fraud'])
        
        # Keep only numeric and low-cardinality categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')
        
        # Drop ID columns
        id_patterns = ['id', 'ID', '_id', 'upi']
        cols_to_drop = []
        for col in df.columns:
            if any(p in col.lower() for p in id_patterns):
                if df[col].nunique() > 100:
                    cols_to_drop.append(col)
        
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} ID columns")
        
        # Update column lists
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        # Fill missing values
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        
        # Encode categorical (limit to top 10 categories per column)
        if categorical_cols:
            for col in categorical_cols:
                top_cats = df[col].value_counts().head(10).index.tolist()
                df[col] = df[col].apply(lambda x: x if x in top_cats else 'Other')
            
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=np.int8)
        
        fraud_rate = df['is_fraud'].mean() * 100
        print(f"  Final shape: {df.shape}, Fraud rate: {fraud_rate:.2f}%\n")
        
        combined_data.append(df)

# Combine datasets
print("="*100)
print("Combining Datasets...")
print("="*100 + "\n")

# Get all columns
all_cols = set()
for df in combined_data:
    all_cols.update(df.columns)

all_cols.discard('is_fraud')
all_cols = sorted(list(all_cols))

print(f"Total features: {len(all_cols)}")

# Align
aligned = []
for df in combined_data:
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0
    aligned.append(df[all_cols + ['is_fraud']])

# Concatenate
final_df = pd.concat(aligned, axis=0, ignore_index=True)

print(f"\nFinal Dataset:")
print(f"  Rows: {len(final_df):,}")
print(f"  Features: {len(all_cols)}")
print(f"  Fraud cases: {final_df['is_fraud'].sum():,} ({final_df['is_fraud'].mean()*100:.2f}%)")

# Split features and target
X = final_df[all_cols]
y = final_df['is_fraud']

# Scale numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-val-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nData Split:")
print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"  Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

# Save
os.makedirs('processed_data', exist_ok=True)
os.makedirs('models', exist_ok=True)

X_train.to_csv('processed_data/X_train.csv', index=False)
X_val.to_csv('processed_data/X_val.csv', index=False)
X_test.to_csv('processed_data/X_test.csv', index=False)
y_train.to_csv('processed_data/y_train.csv', index=False, header=['is_fraud'])
y_val.to_csv('processed_data/y_val.csv', index=False, header=['is_fraud'])
y_test.to_csv('processed_data/y_test.csv', index=False, header=['is_fraud'])

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(all_cols, 'models/feature_names.pkl')

# Save fraud analysis
fraud_analysis = {
    'total_samples': len(final_df),
    'total_fraud': final_df['is_fraud'].sum(),
    'overall_fraud_rate_percent': final_df['is_fraud'].mean() * 100,
    'train_fraud_rate_percent': y_train.mean() * 100,
    'val_fraud_rate_percent': y_val.mean() * 100,
    'test_fraud_rate_percent': y_test.mean() * 100,
}
joblib.dump(fraud_analysis, 'models/fraud_analysis.pkl')

print(f"\n✅ Preprocessing complete!")
print(f"✅ Files saved to processed_data/ and models/")
