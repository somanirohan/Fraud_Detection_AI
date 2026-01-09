from fastapi import FastAPI, UploadFile, File
from risk_engine import analyze_transaction

app = FastAPI()

@app.post("/analyze")
def analyze(data: dict):
    risk = analyze_transaction(data)
    return {"risk": risk}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    contents = file.file.read().decode("utf-8")
    # parse CSV here
    return {"status": "ok"}
