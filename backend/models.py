from sqlalchemy import Column, Integer, Float, String
from backend.database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    transaction_hour = Column(Integer)
    is_new_device = Column(Integer)
    location_change = Column(Integer)
    daily_txn_count = Column(Integer)
    fraud_probability = Column(Float)
    risk_level = Column(String)
