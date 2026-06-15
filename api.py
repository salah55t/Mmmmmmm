# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from tradingview_screener import Query
import uvicorn
from typing import List

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="TradingView API Proxy for Crypto Radar",
    description="جسر بيانات وسيط لجلب مؤشرات التداول من TradingView دون حظر"
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Crypto Radar Proxy API is running smoothly!"}

@app.get("/scan")
def scan_tickers(tickers: List[str] = FastAPIQuery(None)):
    """
    نقطة اتصال تقوم باستقبال قائمة العملات وجلب مؤشراتها فورا من TradingView
    """
    if not tickers:
        # قائمة افتراضية في حال عدم إرسال عملات محددة
        tickers = [
            "BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:SOLUSDT", 
            "BINANCE:BNBUSDT", "BINANCE:ADAUSDT", "BINANCE:AVAXUSDT"
        ]
    
    try:
        # إعداد طلب المؤشرات الفنية الأساسية من TradingView
        q = Query().select(
            'close', 'change', 'volume', 'RSI', 'ADX', 
            'EMA9', 'EMA21', 'BB.lower', 'BB.upper'
        )
        q.set_tickers(*tickers)
        _, data = q.get_scanner_data()
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No data received from TradingView")
            
        # تحويل البيانات إلى قاموس JSON متوافق لإرساله عبر الشبكة
        result = data.to_dict(orient="records")
        return {"success": True, "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    # تشغيل الخادم محليا أو عبر منفذ الاستضافة
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
