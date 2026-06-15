# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tradingview_screener import Query
import pandas as pd
import numpy as np
from typing import List
import uvicorn
import os

# تهيئة تطبيق FastAPI مع إضافة معلومات توضيحية
app = FastAPI(
    title="TradingView Secure Proxy API",
    description="جسر بيانات مطور يعتمد على معالجة الدفعات وتصفية البيانات الفارغة لتجنب خطأ 500"
)

# نموذج استقبال البيانات لضمان صحة المدخلات عبر طلبات POST
class TickerRequest(BaseModel):
    tickers: List[str]

def chunk_list(lst, n):
    """تقسيم القائمة الكبيرة من العملات إلى دفعات صغيرة الحجم لتجنب الضغط والمهلة الزمنية"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@app.get("/")
def health_check():
    """نقطة فحص السلامة للتأكد من أن الخادم يعمل على ريندر"""
    return {"status": "online", "message": "Secure Proxy is fully active and stable!"}

@app.post("/scan")
def scan_tickers(payload: TickerRequest):
    """نقطة الاتصال الرئيسية لجلب ومعالجة مؤشرات التداول"""
    tickers = payload.tickers
    if not tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty")
    
    combined_records = []
    
    # تقسيم العملات إلى دفعات (30 عملة لكل دفعة) لمعالجة مستقرة وسريعة
    batches = list(chunk_list(tickers, 30))
    
    for batch in batches:
        try:
            # صياغة الاستعلام لمكتبة TradingView مع تحديد المؤشرات بدقة عالية
            q = Query().select(
                'close', 'change', 'volume', 'RSI', 'ADX', 
                'EMA9', 'EMA21', 'BB.lower', 'BB.upper'
            )
            q.set_tickers(*batch)
            _, data = q.get_scanner_data()
            
            if data is not None and not data.empty:
                # استبدال قيم المالانهاية (inf) بقيم NaN العادية لمنع الانهيار
                cleaned_data = data.replace([np.inf, -np.inf], np.nan)
                
                # استبدال جميع قيم NaN بكائنات None ليتم تحويلها إلى null القياسية في JSON
                cleaned_data = cleaned_data.astype(object).where(pd.notnull(cleaned_data), None)
                
                # تحويل الجدول إلى قائمة من القواميس وإضافتها للنتائج المدمجة
                records = cleaned_data.to_dict(orient="records")
                combined_records.extend(records)
        except Exception:
            # تخطي أي دفعة تحتوي على عملات غير موجودة دون التسبب في إيقاف الفحص بالكامل
            continue
            
    if not combined_records:
        raise HTTPException(status_code=404, detail="Failed to fetch active data for all batches")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    # تشغيل الخادم والارتباط بالمنفذ الذي تخصصه منصة الاستضافة تلقائياً
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
