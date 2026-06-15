# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel
from tradingview_screener import Query
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn
import os

app = FastAPI(
    title="TradingView Flexible Proxy API",
    description="جسر بيانات يدعم GET و POST معاً لحل مشكلة 404 على ريندر"
)

class TickerRequest(BaseModel):
    tickers: List[str]

def chunk_list(lst, n):
    """تقسيم القائمة الكبيرة إلى دفعات صغيرة"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def execute_scan(tickers: List[str]) -> List[dict]:
    """الدالة الأساسية لمعالجة وجلب البيانات الفنية"""
    print(f"⚙️ جاري معالجة طلب فحص لـ {len(tickers)} عملة...")
    combined_records = []
    batches = list(chunk_list(tickers, 30))
    
    for idx, batch in enumerate(batches):
        try:
            q = Query().select(
                'close', 'change', 'volume', 'RSI', 'ADX', 
                'EMA9', 'EMA21', 'BB.lower', 'BB.upper'
            )
            q.set_tickers(*batch)
            _, data = q.get_scanner_data()
            
            if data is not None and not data.empty:
                cleaned_data = data.replace([np.inf, -np.inf], np.nan)
                cleaned_data = cleaned_data.astype(object).where(pd.notnull(cleaned_data), None)
                records = cleaned_data.to_dict(orient="records")
                combined_records.extend(records)
                print(f"✅ تم بنجاح جلب الدفعة {idx+1}/{len(batches)}")
        except Exception as e:
            print(f"⚠️ خطأ في جلب الدفعة {idx+1}: {e}")
            continue
            
    return combined_records

@app.get("/")
def health_check():
    return {"status": "online", "message": "Proxy is active and healthy!"}

# --- دعم مسارات الـ POST (مع وبدون شرطة مائلة) ---
@app.post("/scan")
@app.post("/scan/")
def scan_tickers_post(payload: TickerRequest):
    print("📥 تم استقبال طلب POST على مسار /scan")
    results = execute_scan(payload.tickers)
    if not results:
        raise HTTPException(status_code=404, detail="No data could be fetched")
    return {"success": True, "data": results}

# --- دعم مسارات الـ GET كبديل احتياطي أوتوماتيكي ممتاز ---
@app.get("/scan")
@app.get("/scan/")
def scan_tickers_get(tickers: Optional[List[str]] = FastAPIQuery(None)):
    print("📥 تم استقبال طلب GET على مسار /scan")
    if not tickers:
        # عملات افتراضية في حال طلب فحص فارغ
        tickers = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:SOLUSDT"]
    results = execute_scan(tickers)
    if not results:
        raise HTTPException(status_code=404, detail="No data could be fetched")
    return {"success": True, "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
