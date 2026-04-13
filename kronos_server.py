"""
Kronos AI Forecast Microservice — Render Deployment
Merlijn Signaal Labo — Camelot Finance

Endpoint: GET /forecast?symbol=BTCUSDT
Health:   GET /health
"""

import os
from flask import Flask, jsonify, request
import ccxt
import pandas as pd
import time

app = Flask(__name__)

# ── Model laden (eenmalig bij opstart) ───────────────────────────────
print("[Kronos] Model laden...")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model_k    = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor  = KronosPredictor(model_k, tokenizer, max_context=512)
    print("[Kronos] Model geladen")
    KRONOS_OK = True
except Exception as e:
    print(f"[Kronos] Model fout: {e}")
    KRONOS_OK = False

# ── Exchange ─────────────────────────────────────────────────────────
# data-api.binance.vision = publieke marktdata zonder geo-restricties
exchange = ccxt.binance({
    'enableRateLimit': True,
    'hostname': 'data-api.binance.vision',
})

# ── Cache: max 1 forecast per symbol per 30 min ─────────────────────
_cache = {}
CACHE_TTL = 1800

SYMBOL_MAP = {
    'XAGUSD': 'PAXGUSDT',
    'XAUUSD': 'PAXGUSDT',
}

def get_binance_symbol(symbol):
    return SYMBOL_MAP.get(symbol.upper(), symbol.upper())


def fetch_ohlcv(symbol, lookback=450):
    binance_sym = get_binance_symbol(symbol)
    pair = binance_sym.replace('USDT', '/USDT')
    ohlcv = exchange.fetch_ohlcv(pair, timeframe='4h', limit=lookback)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def compute_forecast(symbol):
    now = time.time()

    if symbol in _cache and now - _cache[symbol]['ts'] < CACHE_TTL:
        return _cache[symbol]['data']

    if not KRONOS_OK:
        return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0}

    try:
        df = fetch_ohlcv(symbol)

        lookback = 400
        pred_len = 24

        if len(df) < lookback + pred_len:
            return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0,
                    'error': f'Onvoldoende data: {len(df)} rijen'}

        df = df.iloc[-(lookback + pred_len):].reset_index(drop=True)

        x_df        = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
        x_timestamp = df.iloc[:lookback]['timestamps']
        y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps']

        pred_df = predictor.predict(
            df           = x_df,
            x_timestamp  = x_timestamp,
            y_timestamp  = y_timestamp,
            pred_len     = pred_len,
            T            = 1.0,
            top_p        = 0.9,
            sample_count = 3
        )

        current_close  = df.iloc[lookback - 1]['close']
        forecast_close = pred_df['close'].iloc[-1]
        pct_change     = (forecast_close - current_close) / current_close * 100

        raw_score = pct_change * 1.5
        score     = int(max(-15, min(15, round(raw_score))))

        if pct_change > 1.5:
            direction = 'bullish'
        elif pct_change < -1.5:
            direction = 'bearish'
        else:
            direction = 'neutral'

        result = {
            'symbol':    symbol,
            'direction': direction,
            'pct':       round(pct_change, 2),
            'score':     score,
            'forecast':  round(float(forecast_close), 6),
            'current':   round(float(current_close), 6),
        }

        _cache[symbol] = {'ts': now, 'data': result}
        print(f"[Kronos] {symbol}: {direction} {pct_change:+.2f}% score {score:+d}")
        return result

    except Exception as e:
        print(f"[Kronos] Fout {symbol}: {e}")
        return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0, 'error': str(e)}


# ── Routes ───────────────────────────────────────────────────────────

@app.route('/forecast')
def forecast():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    result = compute_forecast(symbol)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'kronos_loaded': KRONOS_OK, 'cached': list(_cache.keys())})

@app.route('/cache/clear')
def clear_cache():
    _cache.clear()
    return jsonify({'status': 'cache cleared'})

@app.route('/')
def index():
    return jsonify({'service': 'Kronos AI Forecast', 'status': 'ok', 'endpoints': ['/forecast?symbol=BTCUSDT', '/health']})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
