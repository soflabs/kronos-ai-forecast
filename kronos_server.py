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

# ── Exchanges (3 grootste, met fallback) ─────────────────────────────
exchanges = {
    'bybit':   ccxt.bybit({'enableRateLimit': True}),
    'binance': ccxt.binance({'enableRateLimit': True}),
    'kraken':  ccxt.kraken({'enableRateLimit': True}),
}
# Primaire exchange voor OHLCV (Bybit werkt wereldwijd, geen geo-restricties)
PRIMARY_EXCHANGE = 'bybit'

# ── Cache: max 1 forecast per symbol per 30 min ─────────────────────
_cache = {}
CACHE_TTL = 1800

SYMBOL_MAP = {
    'XAGUSD':  'PAXGUSDT',
    'XAUUSD':  'PAXGUSDT',
}

# Kraken gebruikt andere symboolnamen
KRAKEN_MAP = {
    'BTC/USDT': 'BTC/USDT',
    'XRP/USDT': 'XRP/USDT',
    'HBAR/USDT': 'HBAR/USDT',
    'VET/USDT': 'VET/USDT',
    'PAXG/USDT': 'PAXG/USDT',
}

def get_trading_pair(symbol):
    mapped = SYMBOL_MAP.get(symbol.upper(), symbol.upper())
    return mapped.replace('USDT', '/USDT')


def fetch_ohlcv(symbol, lookback=450):
    """Haal OHLCV data op met fallback over 3 exchanges."""
    pair = get_trading_pair(symbol)
    errors = []

    # Probeer exchanges in volgorde
    for name in [PRIMARY_EXCHANGE, 'binance', 'kraken', 'bybit']:
        ex = exchanges.get(name)
        if not ex:
            continue
        try:
            ohlcv = ex.fetch_ohlcv(pair, timeframe='4h', limit=lookback)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"[Kronos] OHLCV via {name}: {pair} ({len(df)} candles)")
                return df
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue

    raise Exception(f"Alle exchanges gefaald voor {pair}: {'; '.join(errors)}")


def fetch_multi_exchange_price(symbol):
    """Haal huidige prijs op van 3 exchanges en vergelijk."""
    pair = get_trading_pair(symbol)
    prices = {}

    for name, ex in exchanges.items():
        try:
            ticker = ex.fetch_ticker(pair)
            if ticker and ticker.get('last'):
                prices[name] = round(float(ticker['last']), 6)
        except Exception:
            continue

    if not prices:
        return None

    avg = round(sum(prices.values()) / len(prices), 6)
    spread = round(max(prices.values()) - min(prices.values()), 6)
    spread_pct = round(spread / avg * 100, 3) if avg > 0 else 0

    return {
        'prices': prices,
        'avg': avg,
        'spread': spread,
        'spread_pct': spread_pct,
    }


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

        # Multi-exchange prijsvergelijking
        multi = fetch_multi_exchange_price(symbol)

        result = {
            'symbol':    symbol,
            'direction': direction,
            'pct':       round(pct_change, 2),
            'score':     score,
            'forecast':  round(float(forecast_close), 6),
            'current':   round(float(current_close), 6),
            'exchanges': multi,
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

@app.route('/prices')
def prices():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    result = fetch_multi_exchange_price(symbol)
    if result:
        return jsonify({'symbol': symbol, **result})
    return jsonify({'symbol': symbol, 'error': 'Geen prijzen beschikbaar'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'kronos_loaded': KRONOS_OK, 'cached': list(_cache.keys()),
                    'exchanges': list(exchanges.keys()), 'primary': PRIMARY_EXCHANGE})

@app.route('/cache/clear')
def clear_cache():
    _cache.clear()
    return jsonify({'status': 'cache cleared'})

@app.route('/')
def index():
    return jsonify({'service': 'Kronos AI Forecast', 'status': 'ok',
                    'endpoints': ['/forecast?symbol=BTCUSDT', '/prices?symbol=BTCUSDT', '/health']})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
