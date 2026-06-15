# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tradingview_screener import Query
import pandas as pd
import numpy as np
from typing import List
import uvicorn
import os

app = FastAPI(
    title="TradingView Secure Proxy API",
    description="جسر بيانات مطور يعتمد على معالجة الدفعات وتصفية البيانات الفارغة لتجنب خطأ 500"
)

# نموذج استقبال البيانات عبر طلب POST لمنع مشاكل طول الرابط
class TickerRequest(BaseModel):
    tickers: List[str]

def chunk_list(lst, n):
    """تقسيم القائمة الكبيرة إلى دفعات صغيرة الحجم"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@app.get("/")
def health_check():
    return {"status": "online", "message": "Secure Proxy is fully active and stable!"}

@app.post("/scan")
def scan_tickers(payload: TickerRequest):
    tickers = payload.tickers
    if not tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty")
    
    combined_records = []
    
    # تقسيم العملات إلى دفعات (30 عملة لكل دفعة) لمعالجة مستقرة وسريعة
    batches = list(chunk_list(tickers, 30))
    
    for batch in batches:
        try:
            q = Query().select(
                'close', 'change', 'volume', 'RSI', 'ADX', 
                'EMA9', 'EMA21', 'BB.lower', 'BB.upper'
            )
            q.set_tickers(*batch)
            _, data = q.get_scanner_data()
            
            if data is not None and not data.empty:
                # تحويل البيانات إلى تنسيق بايثون وتصفية قيم NaN/Inf التي تسبب انهيار الـ JSON
                cleaned_data = data.replace([np.inf, -np.inf], np.nan)
                # استبدال جميع الـ NaN بـ None لترسل كـ null متوافقة مع JSON
                cleaned_data = cleaned_data.astype(object).where(pd.notnull(cleaned_data), None)
                
                records = cleaned_data.to_dict(orient="records")
                combined_records.extend(records)
        except Exception as batch_error:
            # تخطي الدفعة التي تحتوي على أخطاء دون التسبب في انهيار الفحص بالكامل
            continue
            
    if not combined_records:
        raise HTTPException(status_code=404, detail="Failed to fetch active data for all batches")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
