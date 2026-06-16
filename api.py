# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
import asyncio
from typing import List, Optional
import uvicorn
import os

app = FastAPI(title="Cyber-Radar Raw Data Proxy")

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

class TickerRequest(BaseModel):
    tickers: List[str]

async def fetch_raw_klines(client: AsyncClient, ticker_raw: str) -> Optional[dict]:
    symbol_clean = ticker_raw.replace("BINANCE:", "").replace(":", "")
    if any(stable in symbol_clean for stable in ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "EURUSDT"]):
        return None
    try:
        # جلب آخر 100 شمعة فريش بإطار 15 دقيقة
        klines = await client.get_klines(symbol=symbol_clean, interval='15m', limit=100)
        if not klines:
            return None
        return {"ticker": ticker_raw, "klines": klines}
    except BinanceAPIException as e:
        if e.code == -1003:
            await asyncio.sleep(5)
        return None
    except Exception:
        return None

@app.get("/")
def health_check():
    return {"status": "online", "message": "Proxy is ready to stream raw data."}

@app.post("/scan")
@app.post("/scan/")
async def scan_tickers(payload: TickerRequest):
    client = await AsyncClient.create(API_KEY, API_SECRET)
    try:
        tasks = [fetch_raw_klines(client, t) for t in payload.tickers]
        results = await asyncio.gather(*tasks)
    finally:
        await client.close_connection()
    combined_records = [r for r in results if r is not None]
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
