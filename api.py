from fastapi import FastAPI
from binance import AsyncClient
import os
import uvicorn

app = FastAPI()

# استخدام المتغيرات البيئية لضمان الأمان
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

@app.get("/fetch_data/{symbol}")
async def fetch_data(symbol: str):
    try:
        # الاتصال المباشر بجلب الشموع (OHLCV)
        client = await AsyncClient.create(API_KEY, API_SECRET)
        klines = await client.get_klines(symbol=symbol, interval='15m', limit=100)
        await client.close_connection()
        
        # إرجاع البيانات بتنسيق نظيف للرادار
        return {"symbol": symbol, "data": klines}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
