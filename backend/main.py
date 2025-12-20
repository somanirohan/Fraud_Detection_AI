from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd

from backend.database import SessionLocal, engine
from backend.models import Transaction
from backend.schemas import TransactionInput
from backend.dummy_model import fake_fraud_probability
from backend.risk_engine import get_risk_level

Transaction.metadata.create_all(bind=engine)

app = FastAPI(title="Fraud Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- SINGLE TRANSACTION ----------------
@app.post("/analyze")
def analyze_transaction(txn: TransactionInput, db: Session = Depends(get_db)):

    features = [
        txn.amount,
        txn.transaction_hour,
        txn.is_new_device,
        txn.location_change,
        txn.daily_txn_count
    ]

    prob = fake_fraud_probability(features)
    risk = get_risk_level(prob)

    record = Transaction(
        amount=txn.amount,
        transaction_hour=txn.transaction_hour,
        is_new_device=txn.is_new_device,
        location_change=txn.location_change,
        daily_txn_count=txn.daily_txn_count,
        fraud_probability=round(prob, 2),
        risk_level=risk
    )

    db.add(record)
    db.commit()

    return {
        "fraud_probability": round(prob, 2),
        "risk_level": risk
    }

# ---------------- CSV UPLOAD ----------------
@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_csv(file.file)

    for _, row in df.iterrows():
        prob = fake_fraud_probability(list(row))
        risk = get_risk_level(prob)

        db.add(Transaction(
            amount=row["amount"],
            transaction_hour=row["transaction_hour"],
            is_new_device=row["is_new_device"],
            location_change=row["location_change"],
            daily_txn_count=row["daily_txn_count"],
            fraud_probability=round(prob, 2),
            risk_level=risk
        ))

    db.commit()
    return {"message": "CSV data stored successfully"}

# ---------------- ALL TRANSACTIONS ----------------
@app.get("/transactions")
def get_transactions(db: Session = Depends(get_db)):
    return db.query(Transaction).all()

# ---------------- DATA FOR GRAPHS ----------------
@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Transaction).count()
    high = db.query(Transaction).filter(Transaction.risk_level == "High").count()
    medium = db.query(Transaction).filter(Transaction.risk_level == "Medium").count()
    low = db.query(Transaction).filter(Transaction.risk_level == "Low").count()

    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low
    }

#-------------------delete----------------

@app.delete("/delete/{txn_id}")
def delete_transaction(txn_id: int, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.id == txn_id).first()

    if not txn:
        return {"error": "Transaction not found"}

    db.delete(txn)
    db.commit()
    return {"message": "Transaction deleted successfully"}

#------------------GRAPH 1: Transactions by Risk Level (Bar Chart)---------

@app.get("/risk-bar")
def risk_bar(db: Session = Depends(get_db)):
    return {
        "High": db.query(Transaction).filter(Transaction.risk_level == "High").count(),
        "Medium": db.query(Transaction).filter(Transaction.risk_level == "Medium").count(),
        "Low": db.query(Transaction).filter(Transaction.risk_level == "Low").count(),
    }

#--------------------GRAPH 2: Risk Trend Over Time (Line Chart)----------------
@app.get("/risk-trend")
def risk_trend(db: Session = Depends(get_db)):
    txns = db.query(Transaction).order_by(Transaction.id).all()
    return [
        {"id": t.id, "risk": t.fraud_probability}
        for t in txns
    ]

#------------------GRAPH 3: Amount vs Risk (Scatter Plot)-----------

@app.get("/amount-risk")
def amount_vs_risk(db: Session = Depends(get_db)):
    return [
        {"amount": t.amount, "risk": t.fraud_probability}
        for t in db.query(Transaction).all()
    ]

#-----------------GRAPH 4: Fraud by Hour (Bar Chart)-------------------

@app.get("/hourly-fraud")
def hourly_fraud(db: Session = Depends(get_db)):
    result = {}
    for h in range(24):
        result[h] = db.query(Transaction).filter(Transaction.transaction_hour == h).count()
    return result
