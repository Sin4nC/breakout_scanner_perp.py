# breakout_scanner_v3.py
import argparse
import asyncio
import time
import os
import aiohttp
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import requests

# --- CONFIG MATRIX ---
CONFIG_MATRIX = {
    "WEIGHT_VOLUME": 25.0,
    "WEIGHT_MOMENTUM": 20.0,
    "WEIGHT_COILING": 15.0,
    "WEIGHT_TREND": 15.0,
    "WEIGHT_OI": 10.0,
    "WEIGHT_ALPHA": 10.0,
    "WEIGHT_FUNDING": 5.0,
    "HARD_LIQUIDITY_FLOOR_USD": 5_000_000.0
}

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="https://contract.mexc.com")
parser.add_argument("--concurrency-limit", type=int, default=20)
parser.add_argument("--alpha-threshold", type=float, default=65.0)
args = parser.parse_args()

# دریافت توکن و چت‌آیدی تلگرام از محیط گیت‌هاب اکشنز
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """ارسال نتایج به تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not found. Skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

class VectorizedQuantEngine:
    @staticmethod
    def calculate_atr_array(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        if len(closes) <= period: return np.zeros_like(closes)
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = np.zeros_like(closes)
        tr_pd = pd.Series(tr)
        atr[period:] = tr_pd.rolling(window=period).mean().values[period-1:]
        return atr

    @classmethod
    def evaluate_market_data(cls, h4_data: dict, daily_data: dict, btc_returns: np.ndarray) -> Optional[dict]:
        c_h4, h_h4, l_h4, v_h4, o_h4 = h4_data['close'], h4_data['high'], h4_data['low'], h4_data['volume'], h4_data['open']
        oi_h4 = h4_data.get('open_interest', np.zeros_like(c_h4))
        c_d = daily_data['close']
        
        if len(c_h4) < 100 or len(c_d) < 40: return None
        idx = -1
        
        avg_dollar_vol_4d = float(np.mean(v_h4[-24:] * c_h4[-24:]))
        if avg_dollar_vol_4d < CONFIG_MATRIX["HARD_LIQUIDITY_FLOOR_USD"]: return None
            
        lookback_block = h_h4[-41:-1]
        try:
            kde = stats.gaussian_kde(lookback_block[2:-2])
            x_eval = np.linspace(np.min(lookback_block), np.max(lookback_block), 50)
            resistance_baseline = float(x_eval[np.argmax(kde(x_eval))])
        except Exception:
            resistance_baseline = float(np.max(lookback_block))
            
        if c_h4[idx] <= resistance_baseline: return None
            
        breakout_midpoint = o_h4[idx] + ((c_h4[idx] - o_h4[idx]) / 2.0)
        structural_floor = min(breakout_midpoint, resistance_baseline)
        if c_h4[idx] < structural_floor: return None

        atr = cls.calculate_atr_array(h_h4, l_h4, c_h4, period=14)
        candle_range = h_h4[idx] - l_h4[idx]
        atr_mult = candle_range / atr[idx-1] if atr[idx-1] > 0 else 1.0
        if atr_mult < 1.5: return None
        
        historical_vols = v_h4[-120:-1]
        vol_percentile = float(stats.percentileofscore(historical_vols, v_h4[idx]) / 100.0)
        
        h4_series = pd.Series(c_h4)
        bbw_history = (h4_series.rolling(20).std() * 4.0) / h4_series.rolling(20).mean()
        bbw_block = bbw_history.values[-200:-1]
        bbw_block = bbw_block[~np.isnan(bbw_block)]
        bbw_percentile = float(stats.percentileofscore(bbw_block, bbw_history.values[-1]) / 100.0) if len(bbw_block) > 0 else 0.5
        
        oi_change_pct = (oi_h4[idx] - oi_h4[idx-4]) / oi_h4[idx-4] if oi_h4[idx-4] > 0 else 0.0
        funding_rate = float(h4_data.get('funding_rate', 0.0))
        
        asset_returns = np.diff(c_h4[-21:]) / c_h4[-22:-1]
        alpha_score_metric = 5.0
        if len(asset_returns) == len(btc_returns):
            try:
                cov = np.cov(asset_returns, btc_returns)[0][1]
                btc_var = np.var(btc_returns)
                beta = cov / btc_var if btc_var > 0 else 1.0
                alpha_raw = np.mean(asset_returns) - (beta * np.mean(btc_returns))
                alpha_score_metric = max(0.0, min(10.0, alpha_raw * 1500.0))
            except Exception: pass

        score_volume = vol_percentile * CONFIG_MATRIX["WEIGHT_VOLUME"]
        score_momentum = min((atr_mult / 3.5), 1.0) * CONFIG_MATRIX["WEIGHT_MOMENTUM"]
        score_coiling = (1.0 - bbw_percentile) * CONFIG_MATRIX["WEIGHT_COILING"]
        
        daily_ema21 = pd.Series(c_d).ewm(span=21, adjust=False).mean().values
        dist_ema21 = (c_d[idx] - daily_ema21[idx]) / daily_ema21[idx]
        score_trend = min(max(0.0, (dist_ema21 + 0.05) / 0.15), 1.0) * CONFIG_MATRIX["WEIGHT_TREND"]
        
        score_oi = 0.0
        if oi_change_pct > 0.08: score_oi = CONFIG_MATRIX["WEIGHT_OI"]
        elif oi_change_pct >= 0.0: score_oi = CONFIG_MATRIX["WEIGHT_OI"] * 0.5
        
        annualized_funding = funding_rate * 3 * 365
        score_funding = 0.0
        if annualized_funding <= 0.06: score_funding = CONFIG_MATRIX["WEIGHT_FUNDING"]
        elif annualized_funding <= 0.22: score_funding = (1.0 - ((annualized_funding - 0.06) / 0.16)) * CONFIG_MATRIX["WEIGHT_FUNDING"]
        else: score_funding = -10.0
        
        score_alpha = (alpha_score_metric / 10.0) * CONFIG_MATRIX["WEIGHT_ALPHA"]
        
        composite_alpha_score = score_volume + score_momentum + score_coiling + score_trend + score_oi + score_funding + score_alpha
        if composite_alpha_score < args.alpha_threshold: return None
        
        return {
            "score": round(composite_alpha_score, 1),
            "close": float(c_h4[idx]),
            "rvol_pct": round(vol_percentile * 100, 1),
            "atr_mult": round(atr_mult, 2),
            "oi_growth": round(oi_change_pct * 100, 1),
            "annualized_funding_pct": round(annualized_funding * 100, 2),
            "timestamp": datetime.fromtimestamp(int(h4_data['time'][idx]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        }

class ProductionDataPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(args.concurrency_limit)

    async def retrieve_active_universe(self, session: aiohttp.ClientSession) -> List[str]:
        url = f"{self.base_url}/api/v1/contract/detail"
        try:
            async with session.get(url, timeout=10) as response:
                payload = await response.json()
                data = payload.get("data") or []
                return [str(item.get("symbol")).upper() for item in data 
                        if str(item.get("settleCoin")).upper() == "USDT" and int(item.get("state", 0)) == 0]
        except Exception:
            return []

    async def fetch_clean_kline(self, session: aiohttp.ClientSession, symbol: str, interval: str, duration_days: int) -> Optional[dict]:
        end_time = int(time.time())
        start_time = end_time - (duration_days * 86400)
        url = f"{self.base_url}/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "start": start_time, "end": end_time}
        
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with session.get(url, params=params, timeout=12) as response:
                        if response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if response.status != 200: return None
                        raw = await response.json()
                        d = raw.get("data") or {}
                        if not d or 'time' not in d or len(d['time']) == 0: return None
                        
                        c_arr = np.array(d['close'], dtype=np.float64)
                        mock_oi = np.linspace(100000, 120000, len(c_arr)) * (c_arr / np.mean(c_arr))
                        
                        return {
                            'time': np.array(d['time'], dtype=np.int64),
                            'open': np.array(d['open'], dtype=np.float64),
                            'high': np.array(d['high'], dtype=np.float64),
                            'low': np.array(d['low'], dtype=np.float64),
                            'close': c_arr,
                            'volume': np.array(d['volume'], dtype=np.float64),
                            'open_interest': mock_oi,
                            'funding_rate': 0.0001
                        }
                except Exception:
                    await asyncio.sleep(0.5)
            return None

    async def extract_and_score_asset(self, session: aiohttp.ClientSession, symbol: str, btc_returns: np.ndarray) -> Optional[dict]:
        h4_task = self.fetch_clean_kline(session, symbol, "Hour4", duration_days=25)
        daily_task = self.fetch_clean_kline(session, symbol, "Day1", duration_days=60)
        h4_data, daily_data = await asyncio.gather(h4_task, daily_task)
        
        if not h4_data or not daily_data: return None
        metrics = VectorizedQuantEngine.evaluate_market_data(h4_data, daily_data, btc_returns)
        if metrics: metrics["symbol"] = symbol
        return metrics

async def main():
    pipeline = ProductionDataPipeline(args.api_url)
    async with aiohttp.ClientSession() as session:
        universe = await pipeline.retrieve_active_universe(session)
        if not universe: return
        
        btc_data = await pipeline.fetch_clean_kline(session, "BTC_USDT", "Hour4", duration_days=25)
        if btc_data:
            btc_closes = btc_data['close'][-21:]
            btc_returns = np.diff(btc_closes) / btc_closes[:-1]
        else:
            btc_returns = np.zeros(20)

        tasks = [pipeline.extract_and_score_asset(session, sym, btc_returns) for sym in universe]
        raw_results = await asyncio.gather(*tasks)
        
        validated_signals = [res for res in raw_results if res is not None]
        validated_signals.sort(key=lambda x: x['score'], reverse=True)
        
        if validated_signals:
            msg_lines = ["🚀 *MEXC 4H Breakout Signals* 🚀\n"]
            for rank, sig in enumerate(validated_signals[:15], 1):
                line = f"{rank}. *{sig['symbol']}* | Score: {sig['score']} | Price: {sig['close']}\n(RVOL: {sig['rvol_pct']}% | ATR Mult: {sig['atr_mult']})"
                msg_lines.append(line)
            
            full_message = "\n".join(msg_lines)
            print(full_message)
            send_telegram_message(full_message)
        else:
            print("No high-alpha breakout signals detected.")

if __name__ == "__main__":
    asyncio.run(main())
