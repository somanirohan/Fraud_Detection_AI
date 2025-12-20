# 🛡️ AI-Based Fraud Detection System

<div align="center">

![Fraud Detection](https://img.shields.io/badge/Fraud%20Detection-94.79%25-success)
![False Positive Rate](https://img.shields.io/badge/False%20Positive%20Rate-1.41%25-blue)
![Models](https://img.shields.io/badge/Models-3-orange)
![Samples](https://img.shields.io/badge/Samples-787K+-purple)

**A comprehensive machine learning system for detecting fraudulent transactions with high accuracy and explainability**

**Tech Stack**: Python + XGBoost + Random Forest + Isolation Forest + scikit-learn

[Features](#-features) • [Quick Start](#-quick-start) • [Performance](#-performance-metrics) • [Documentation](#-complete-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Usage Examples](#-usage-examples)
- [Training Your Own Models](#-train-models-with-your-own-data)
- [Deployment](#-deployment)

---

## 🎯 Overview

This project implements a **state-of-the-art fraud detection system** using ensemble machine learning:

- ✅ **Advanced AI/ML Models**: Ensemble of XGBoost, Random Forest, and Isolation Forest
- ✅ **High Accuracy**: 94.79% fraud detection rate with only 1.41% false positives
- ✅ **Explainable AI**: Feature importance analysis for every prediction
- ✅ **Production Ready**: Complete with training scripts, prediction tools, and model retraining
- ✅ **Flexible**: Train on your own data or use pre-trained models

### Key Highlights

```
┌─────────────────────────────────────────────────────────────┐
│                    Project Highlights                       │
├─────────────────────────────────────────────────────────────┤
│  📊 Data Processed:           787,108 transactions          │
│  🤖 Models Trained:           3 (XGBoost, RF, IF)          │
│  🎯 Best Detection Rate:      94.79%                        │
│  ⚡ False Positive Rate:      1.41%                         │
│  📈 PR-AUC Score:             0.9614                        │
│  ⏱️  Inference Speed:          <50ms per transaction        │
│  💾 Total Features:           243                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎨 Web Application Features

#### 1. **Dashboard (📊)**
- Real-time statistics (total transactions, fraud count, today's stats)
- Risk distribution pie chart (HIGH/MEDIUM/LOW)
- Fraud trend line chart (last 30 days)
- Hourly fraud pattern bar chart
- Auto-refresh capability

#### 2. **Single Transaction Prediction (🔮)**
- Interactive form for transaction details
- Ensemble prediction using all 3 models
- Risk classification (HIGH/MEDIUM/LOW)
- Individual model scores (XGBoost, RF, IF)
- Top contributing features displayed
- Clear recommended actions

#### 3. **Batch CSV Upload (📤)**
- Drag & drop interface
- Bulk processing (thousands of transactions)
- Real-time progress tracking
- Summary results with fraud statistics
- Sample results table view

#### 4. **Transaction Management (📋)**
- View all transactions with pagination
- Filter by risk level and prediction
- Delete suspicious entries
- Real-time updates
- Detailed transaction metadata

#### 5. **Architecture Visualization (🏗️)**
- All 7 system flowchart images:
  1. System Architecture
  2. Website Architecture
  3. AI Model Workflow
  4. Data Pipeline
  5. Dashboard UI
  6. Comparison Table
  7. Citizen Impact
- Beautiful image gallery layout
- Detailed descriptions for each diagram

#### 6. **Model Information (🤖)**
- Performance metrics dashboard
- Model comparison table
- Feature importance chart (top 20 features)
- Training data statistics
- Ensemble strategy details

### 🤖 Machine Learning Features

| Model | Type | Detection Rate | False Positive Rate | Use Case |
|-------|------|----------------|---------------------|----------|
| **XGBoost** | Supervised Classifier | **94.79%** | **1.41%** | Primary fraud detection |
| **Random Forest** | Supervised Classifier | 94.71% | 1.47% | Baseline & validation |
| **Isolation Forest** | Unsupervised Anomaly Detector | 39.52% | 17.07% | Novel pattern detection |
| **Ensemble** | Weighted Voting | **94.79%** | **1.41%** | Production system |

### 🔧 Core Capabilities

- **Real-time Fraud Detection**: Process transactions in <50ms
- **Risk Scoring**: 0-100 risk score with HIGH/MEDIUM/LOW classification
- **Batch Processing**: Handle thousands of transactions efficiently
- **Explainable AI**: Feature importance analysis for every prediction
- **User Feedback System**: Collect feedback for continuous improvement
- **Adaptive Learning**: Models can be retrained with new data
- **Zero Data Loss**: Comprehensive preprocessing with no information loss

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Required packages (see requirements.txt)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Making Predictions

```bash
# Run prediction examples
python 8_prediction_examples.py
```

This will demonstrate:
- Single transaction prediction
- Batch prediction
- Risk classification
- Model comparison

---

## 🎓 Training Models with Your Own Data

You have **TWO options** for training models with your data:

### Option 1: Improve Existing Models (Recommended) ⭐

**Add your data to existing training data for continuous improvement**

```bash
python improve_models_with_new_data.py
```

**What it does:**
1. Loads your new CSV data
2. Loads existing training data (550K+ samples)
3. **Combines old + new data**
4. Retrains models on combined dataset
5. Backs up old models
6. Updates training data for future improvements

**Example:**
```
Before: 550,975 training samples → 94.79% detection
After adding 10,000 new samples: 560,975 samples → 95.12% detection ⬆️
```

**Benefits:**
- ✅ Continuous improvement over time
- ✅ No data loss - keeps all historical data
- ✅ Better generalization with more examples
- ✅ Incremental learning

### Option 2: Retrain from Scratch

**Replace training data completely with your data**

```bash
python retrain_existing_models.py
```

**What it does:**
1. Uploads your CSV file
2. Aligns your data with existing 243 features
3. Retrains XGBoost, Random Forest, Isolation Forest
4. Backs up old models to `models/backups/`
5. Replaces production models in `models/`

**Use when:**
- You have completely different data
- You want to start fresh
- Your data is from a different domain

### CSV Requirements

Your CSV file should contain:
- **Transaction features**: amount, time, location, user info, etc.
- **Fraud label column**: Binary column (e.g., `is_fraud`, `fraud`, `label`)
  - Values: 0 = Legitimate, 1 = Fraud

**Example CSV:**
```csv
amount,hour,user_age,account_age,device_type,is_fraud
5000,14,35,365,mobile,0
15000,2,28,30,web,1
3000,10,45,1000,tablet,0
```

### Training Workflow

```
1. Prepare CSV → 2. Run Script → 3. Follow Prompts → 4. Models Updated!
```

**Interactive prompts will guide you through:**
- Uploading your CSV file
- Confirming fraud label column
- Choosing which models to train
- Saving the trained models

### After Training

Your updated models are ready to use:
```bash
python 8_prediction_examples.py
```

### Restore from Backup

If needed, restore original models:
```bash
copy models\backups\xgboost_model_backup_*.pkl models\xgboost_model.pkl
```

---

## 📁 Project Structure

```
AI Fraud Detection/
│
├── 📊 Data/                              # Raw datasets (16 CSV files)
│   ├── dataset.csv                       # 250K transactions
│   ├── dataset (1).csv                   # 209K transactions
│   ├── upi_transactions_2024.csv         # 250K transactions
│   ├── fraud_dataset.csv                 # 26K transactions
│   ├── Fraud Detection Dataset.csv       # 51K transactions
│   └── ... (11 more datasets)
│
├── 🔧 processed_data/                    # Preprocessed data
│   ├── X_train.csv                       # 550,975 samples
│   ├── X_val.csv                         # 118,066 samples
│   ├── X_test.csv                        # 118,067 samples
│   └── y_train.csv, y_val.csv, y_test.csv
│
├── 🤖 models/                            # Trained models & results
│   ├── xgboost_model.pkl                 # XGBoost (94.79% detection)
│   ├── random_forest_model.pkl           # Random Forest (94.71%)
│   ├── isolation_forest_model.pkl        # Isolation Forest (39.52%)
│   ├── scaler.pkl                        # Feature scaler
│   ├── feature_names.pkl                 # 243 feature names
│   ├── test_results.csv                  # Test set results
│   ├── model_comparison.csv              # Model comparison
│   └── combined_feature_importance.csv   # Feature importance
│
├── 🌐 backend/                           # FastAPI Backend
│   ├── app.py                            # Main application
│   └── requirements.txt                  # Python dependencies
│
├── ⚛️ frontend/                          # React Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx             # Statistics & charts
│   │   │   ├── Predict.jsx               # Single prediction
│   │   │   ├── BatchUpload.jsx           # CSV upload
│   │   │   ├── Transactions.jsx          # Transaction management
│   │   │   ├── Architecture.jsx          # Flowcharts display
│   │   │   └── ModelInfo.jsx             # Model performance
│   │   ├── components/
│   │   │   └── StatCard.jsx              # Reusable components
│   │   ├── services/
│   │   │   └── api.js                    # API service layer
│   │   ├── App.jsx                       # Main app component
│   │   └── main.jsx                      # Entry point
│   ├── public/
│   │   └── images/                       # All 7 flowchart images
│   ├── package.json
│   └── vite.config.js
│
├── 🐍 Essential Scripts/                 # ML Pipeline
│   ├── 1_load_all_data.py               # Load & combine datasets
│   ├── 2_preprocess_data.py             # Preprocess & feature engineering
│   ├── 6_train_all_models.py            # Train all 3 models + ensemble
│   ├── 7_test_models.py                 # Test on test set
│   ├── 8_prediction_examples.py         # Prediction examples
│   ├── 9_feature_importance.py          # Feature importance analysis
│   ├── generate_json_report.py          # Generate JSON report
│   ├── train_with_user_data.py          # Train on custom data
│   └── predict_with_user_model.py       # Predict with custom models
│
├── 🔮 Reusable Functions/
│   ├── predict_fraud.py                  # Prediction function
│   └── explain_fraud.py                  # Explanation function
│
└── 📖 Documentation/
    ├── README.md                         # This file
    ├── fraud_detection_report.json       # JSON report (20 KB)
    ├── fraud_detection_report_compact.json
    └── requirements.txt                  # Python dependencies
```

---

## 📊 Performance Metrics

### Test Set Results (118,067 transactions)

```
┌─────────────────────────────────────────────────────────────┐
│                    XGBoost Performance                      │
├─────────────────────────────────────────────────────────────┤
│  Fraud Detection Rate:        94.79%  ████████████████████  │
│  False Positive Rate:          1.41%  █                     │
│  Precision:                   94.90%  ████████████████████  │
│  PR-AUC:                      0.9614  ████████████████████  │
│  True Positives:             24,312                         │
│  False Negatives:             1,337                         │
│  True Negatives:             91,111                         │
│  False Positives:             1,307                         │
└─────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Metric | XGBoost | Random Forest | Isolation Forest | Ensemble |
|--------|---------|---------------|------------------|----------|
| **Fraud Detection Rate** | **94.79%** | 94.71% | 39.52% | **94.79%** |
| **False Positive Rate** | **1.41%** | 1.47% | 17.07% | **1.41%** |
| **Precision** | **94.90%** | 94.71% | 37.16% | **94.90%** |
| **PR-AUC** | **0.9614** | 0.9460 | N/A | 0.9569 |

### Business Impact

```
For every 100 transactions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fraud Attempts (22):
  ✅ Detected: 21 (94.79%)
  ❌ Missed: 1 (5.21%)

Legitimate Transactions (78):
  ✅ Approved: 77 (98.59%)
  ❌ Flagged: 1 (1.41%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📚 Complete Documentation

### Dataset Information

Successfully loaded and combined **5 high-quality datasets**:

| Dataset | Rows | Features | Fraud Rate | Description |
|---------|------|----------|------------|-------------|
| `dataset.csv` | 250,000 | 70 | 50.01% | Large UPI transaction dataset |
| `dataset (1).csv` | 209,715 | 33 | 0.11% | Real-world transaction data |
| `upi_transactions_2024.csv` | 250,000 | 70 | 0.19% | 2024 UPI transactions |
| `fraud_dataset.csv` | 26,393 | 117 | 17.22% | Dedicated fraud dataset |
| `Fraud Detection Dataset.csv` | 51,000 | 28 | 9.65% | Fraud detection benchmark |

**Combined Dataset Statistics:**

```
┌────────────────────────────────────────────────────────┐
│              Final Combined Dataset                    │
├────────────────────────────────────────────────────────┤
│  Total Samples:           787,108 transactions         │
│  Total Features:          243 (after preprocessing)    │
│  Fraud Cases:             170,989 (21.72%)            │
│  Legitimate Cases:        616,119 (78.28%)            │
│  Missing Values:          0 (zero data loss ✓)        │
│                                                        │
│  Training Set:            550,975 (70%)               │
│  Validation Set:          118,066 (15%)               │
│  Test Set:                118,067 (15%)               │
└────────────────────────────────────────────────────────┘
```

### Model Architecture

#### Ensemble Strategy

```
Transaction → Preprocess → 3 Models → Weighted Vote → Risk Score
                           ├─ XGBoost (70%)
                           ├─ Isolation Forest (20%)
                           └─ Random Forest (10%)
```

**Ensemble Logic:**
```python
ensemble_score = (0.70 * xgb_prob + 
                  0.20 * iso_prob + 
                  0.10 * rf_prob)

if ensemble_score >= 0.80:
    risk_level = "HIGH RISK"
    action = "BLOCK & INVESTIGATE"
elif ensemble_score >= 0.50:
    risk_level = "MEDIUM RISK"
    action = "FLAG FOR REVIEW"
else:
    risk_level = "LOW RISK"
    action = "ALLOW"
```

#### Model Specifications

**XGBoost (Primary Classifier)**
```python
Parameters:
  - learning_rate: 0.05
  - max_depth: 8
  - n_estimators: 200
  - scale_pos_weight: 4.39 (class imbalance handling)
  - subsample: 0.8
  - colsample_bytree: 0.8
  - objective: binary:logistic
  - eval_metric: aucpr
```

**Random Forest (Baseline)**
```python
Parameters:
  - n_estimators: 150
  - max_depth: 15
  - min_samples_split: 10
  - class_weight: balanced
  - random_state: 42
```

**Isolation Forest (Anomaly Detector)**
```python
Parameters:
  - n_estimators: 100
  - contamination: 0.2172 (auto-calculated)
  - max_samples: 256
  - random_state: 42
```

### Feature Importance

**Top 10 Most Important Features:**

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **Previous_Fraudulent_Transactions** | 23.82% | Historical fraud count |
| 2 | **risk_score** | 22.05% | Pre-computed risk indicator |
| 3 | **Account_Age** | 10.38% | Age of user account |
| 4 | **amount (INR)** | 7.22% | Transaction amount |
| 5 | **step** | 5.69% | Transaction sequence |
| 6 | **nameOrig_Other** | 5.26% | Sender account type |
| 7 | **unusual_transaction_amount_flag** | 2.39% | Amount anomaly flag |
| 8 | **amount** | 2.21% | Base transaction amount |
| 9 | **nameDest_Other** | 1.34% | Receiver account type |
| 10 | **Transaction_Amount** | 1.34% | Normalized amount |

**Key Insights:**
- Historical fraud behavior is the strongest predictor (23.82%)
- Pre-computed risk scores are highly valuable (22.05%)
- Account age matters - newer accounts are riskier (10.38%)
- Transaction amount patterns contribute ~13% total importance

### Web Application

#### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                           │
│  • Web Interface (React)                                    │
│  • Dashboard, Predict, Batch Upload, Transactions          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  • FastAPI Backend                                          │
│  • REST APIs                                                │
│  • Request Validation                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI/ML LAYER                           │
│  • XGBoost Classifier (70% weight)                         │
│  • Isolation Forest (20% weight)                           │
│  • Random Forest (10% weight)                              │
│  • Ensemble Voting                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│  • MongoDB Database                                         │
│  • Transaction Storage                                      │
│  • Statistics Caching                                       │
└─────────────────────────────────────────────────────────────┘
```

#### Tech Stack

**Frontend:**
- React 18
- Vite (build tool)
- React Router (navigation)
- Chart.js (visualizations)
- React Dropzone (file upload)
- Axios (HTTP client)
- React Toastify (notifications)

**Backend:**
- FastAPI (async Python framework)
- Motor (async MongoDB driver)
- Pydantic (data validation)
- Pandas (data processing)
- XGBoost, scikit-learn (ML models)

**Database:**
- MongoDB (NoSQL database)

### API Reference

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/api/predict` | POST | Single transaction prediction |
| `/api/batch-predict` | POST | CSV batch upload |
| `/api/stats` | GET | Global statistics |
| `/api/transactions` | GET | List transactions (with filters) |
| `/api/transactions/{id}` | GET | Get specific transaction |
| `/api/transactions/{id}` | DELETE | Delete transaction |
| `/api/feedback` | POST | Submit user feedback |
| `/api/charts/risk-distribution` | GET | Risk distribution data |
| `/api/charts/fraud-trend` | GET | Fraud trend data (last N days) |
| `/api/charts/hourly-pattern` | GET | Hourly pattern data |
| `/api/model-performance` | GET | Model performance metrics |
| `/api/feature-importance` | GET | Feature importance data |
| `/api/images` | GET | Architecture images list |

#### Example: Single Prediction

**Request:**
```bash
POST /api/predict
Content-Type: application/json

{
  "amount": 15000,
  "hour": 2,
  "user_age": 28,
  "account_age": 30,
  "device_type": "mobile",
  "transaction_type": "transfer"
}
```

**Response:**
```json
{
  "transaction_id": "TXN20251220210530123456",
  "fraud_probability": 0.8573,
  "risk_level": "HIGH",
  "prediction": "FRAUD",
  "xgboost_score": 0.89,
  "random_forest_score": 0.82,
  "isolation_forest_score": 1.0,
  "ensemble_score": 0.8573,
  "top_features": [
    {"Feature": "Previous_Fraudulent_Transactions", "Average_Importance_%": 23.82},
    {"Feature": "risk_score", "Average_Importance_%": 22.05}
  ],
  "recommended_action": "BLOCK & INVESTIGATE - Immediate action required",
  "timestamp": "2025-12-20T21:05:30"
}
```

### Database Schema

#### Transactions Collection
```javascript
{
  transaction_id: String,              // Unique transaction ID
  input_data: Object,                  // Original transaction data
  fraud_probability: Number,           // Ensemble fraud probability
  xgboost_score: Number,              // XGBoost prediction score
  random_forest_score: Number,        // Random Forest score
  isolation_forest_score: Number,     // Isolation Forest score
  ensemble_score: Number,             // Final ensemble score
  risk_level: String,                 // HIGH/MEDIUM/LOW
  prediction: String,                 // FRAUD/Legitimate
  recommended_action: String,         // Action recommendation
  timestamp: Date,                    // Prediction timestamp
  feedback_received: Boolean,         // User feedback flag
  actual_fraud: Boolean,              // Actual fraud (from feedback)
  user_comment: String                // User comment (optional)
}
```

#### Statistics Collection
```javascript
{
  _id: "global",
  total_transactions: Number,
  fraud_count: Number,
  today_transactions: Number,
  today_fraud: Number,
  last_updated: Date
}
```

#### Feedback Collection
```javascript
{
  transaction_id: String,
  actual_fraud: Boolean,
  user_comment: String,
  timestamp: Date
}
```

---

## 💻 Usage Examples

### 1. Single Transaction Prediction

**Via Web Interface:**
1. Navigate to "Predict" tab
2. Enter transaction details
3. Click "Analyze Transaction"
4. View fraud probability and risk level

**Via API:**
```python
import requests

response = requests.post('http://localhost:8000/api/predict', json={
    'amount': 15000,
    'hour': 2,
    'user_age': 28,
    'account_age': 30,
    'device_type': 'mobile'
})

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
print(f"Risk Level: {result['risk_level']}")
```

### 2. Batch CSV Upload

**Via Web Interface:**
1. Navigate to "Batch Upload" tab
2. Drag & drop CSV file
3. Click "Upload & Analyze"
4. View results summary

**CSV Format:**
```csv
amount,hour,user_age,account_age,device_type
5000,14,35,365,mobile
15000,2,28,30,web
3000,10,45,1000,tablet
```

### 3. View Dashboard Statistics

**Via Web Interface:**
1. Navigate to "Dashboard" tab
2. View real-time statistics
3. Analyze charts
4. Click refresh to update

**Via API:**
```python
response = requests.get('http://localhost:8000/api/stats')
stats = response.json()

print(f"Total Transactions: {stats['total_transactions']}")
print(f"Fraud Count: {stats['fraud_count']}")
print(f"Fraud Rate: {stats['fraud_percentage']:.2f}%")
```

### 4. Train on Custom Data (New Models)

```bash
# Interactive training script - creates NEW custom models
python train_with_user_data.py

# Follow prompts to:
# 1. Upload your CSV file
# 2. Select fraud label column
# 3. Choose models to train
# 4. Save trained models to user_models/
```

### 5. Retrain Existing Models with Your Data

```bash
# Retrain the EXISTING production models with your data
python retrain_existing_models.py

# This will:
# 1. Upload your CSV file
# 2. Align your data with existing 243 features
# 3. Retrain XGBoost, RF, IF models
# 4. Backup old models
# 5. Replace production models
# 6. Restart backend to use new models
```

**Key Difference:**
- `train_with_user_data.py` → Creates **new custom models** in `user_models/`
- `retrain_existing_models.py` → **Updates production models** in `models/`

### 6. Make Predictions with Custom Models

```bash
# Use your trained models
python predict_with_user_model.py

# Upload new transactions and get predictions
```

---

## 🚀 Deployment

### Backend (FastAPI)

**Development:**
```bash
python backend/app.py
```

**Production:**
```bash
pip install gunicorn
gunicorn backend.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend (React)

**Development:**
```bash
cd frontend
npm run dev
```

**Production:**
```bash
cd frontend
npm run build

# Serve with nginx, Apache, or any static server
# Build output is in frontend/dist/
```

### MongoDB

**Local:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**Cloud (MongoDB Atlas):**
1. Create free cluster at https://www.mongodb.com/cloud/atlas
2. Get connection string
3. Set environment variable:
```bash
export MONGODB_URL="mongodb+srv://user:pass@cluster.mongodb.net/fraud_detection"
```

### Environment Variables

Create `.env` file in `backend/`:

```
MONGODB_URL=mongodb://localhost:27017
# or
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/fraud_detection
```

---

## 🐛 Troubleshooting

### MongoDB Connection Error

```bash
# Check if MongoDB is running
mongosh

# Or check Docker container
docker ps

# Restart MongoDB
docker restart mongodb
```

### Backend Not Loading Models

- Ensure `models/` directory exists with all .pkl files:
  - xgboost_model.pkl
  - random_forest_model.pkl
  - isolation_forest_model.pkl
  - scaler.pkl
  - feature_names.pkl
- Run backend from project root directory
- Check file permissions

### Frontend API Errors

- Verify backend is running on port 8000
- Check CORS is enabled in backend
- Open browser console for detailed errors
- Verify API base URL in `frontend/src/services/api.js`

### Images Not Loading

- Ensure images are in `frontend/public/images/`
- Check image paths in API response
- Verify backend static files serving
- Check browser network tab for 404 errors

### Port Already in Use

```bash
# Backend (port 8000)
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Frontend (port 3000)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 📊 Performance Benchmarks

### Inference Speed

| Batch Size | XGBoost | Random Forest | Isolation Forest | Ensemble |
|------------|---------|---------------|------------------|----------|
| 1 | 15ms | 12ms | 8ms | 18ms |
| 10 | 18ms | 15ms | 10ms | 22ms |
| 100 | 35ms | 28ms | 18ms | 42ms |
| 1,000 | 120ms | 95ms | 65ms | 145ms |

✅ **All models meet <50ms requirement for single transactions**

### Memory Usage

| Component | Memory |
|-----------|--------|
| XGBoost Model | 45 MB |
| Random Forest Model | 120 MB |
| Isolation Forest Model | 25 MB |
| Feature Scaler | 2 MB |
| **Total** | **~200 MB** |

---

## ✅ Features Checklist

### Data Processing
- [x] Load and combine 5 datasets (787K transactions)
- [x] Zero data loss preprocessing
- [x] Feature engineering (243 features)
- [x] Stratified train/val/test split (70/15/15)
- [x] Handle class imbalance

### Machine Learning
- [x] Train XGBoost (94.79% detection)
- [x] Train Random Forest (94.71% detection)
- [x] Train Isolation Forest (anomaly detection)
- [x] Ensemble model with weighted voting
- [x] Feature importance analysis
- [x] Model performance evaluation

### Backend (FastAPI)
- [x] REST API endpoints
- [x] MongoDB integration
- [x] Async operations
- [x] Request validation
- [x] Error handling
- [x] CORS configuration
- [x] Health check endpoint
- [x] API documentation (Swagger)

### Frontend (React)
- [x] Dashboard with statistics
- [x] Real-time charts (Chart.js)
- [x] Single prediction form
- [x] CSV batch upload (drag & drop)
- [x] Transaction management
- [x] Architecture visualization
- [x] Model information page
- [x] Responsive design
- [x] Toast notifications
- [x] Loading states

### Additional Features
- [x] User feedback system
- [x] Custom data training
- [x] Prediction with custom models
- [x] JSON report generation
- [x] Comprehensive documentation
- [x] All 7 architecture images integrated

---

## 🎓 Key Learnings

### What Worked Well ✅

1. **Ensemble Approach**: Combining multiple models improved robustness
2. **Feature Engineering**: Creating 243 features captured complex patterns
3. **Class Imbalance Handling**: `scale_pos_weight` significantly improved detection
4. **Zero Data Loss**: Careful preprocessing preserved all information
5. **Modern Tech Stack**: React + FastAPI + MongoDB = fast and scalable
6. **Explainability**: Feature importance made predictions interpretable

### Challenges Overcome 💪

1. **Large Dataset**: Optimized preprocessing for 787K samples
2. **Memory Issues**: Limited categorical encoding to top 10 categories
3. **Multiclass Labels**: Converted to binary for consistent training
4. **Async MongoDB**: Implemented async operations for better performance
5. **Feature Alignment**: Ensured all datasets had same 243 features

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **AI Fraud Detection Team**

---

## 🙏 Acknowledgments

- Kaggle for providing fraud detection datasets
- scikit-learn, XGBoost communities
- FastAPI and React communities
- All contributors and researchers in fraud detection field

---

## 📞 Contact & Support

For questions or support:
- Create an issue in the repository
- Email: support@frauddetection.com

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

**🛡️ Built for Excellence | 🎯 Designed for Impact | 🚀 Ready for Production**

*AI-Based Fraud Detection & Risk Management System*

Made with ❤️ for fraud prevention

</div>
