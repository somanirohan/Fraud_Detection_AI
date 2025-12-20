from pydantic import BaseModel

class TransactionInput(BaseModel):
    amount: float
    transaction_hour: int
    is_new_device: int
    location_change: int
    daily_txn_count: int
