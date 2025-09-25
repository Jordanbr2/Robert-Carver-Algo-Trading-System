import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.utils import resample

# === Download Data ===
tickers = ["SPY","GDX", "TLT"]
data = yf.download(tickers, start="2010-01-01")
price_data = data["Close"][tickers].ffill().dropna()
high_data = data["High"][tickers]
low_data = data["Low"][tickers]

volume_data = data["Volume"][tickers].ffill().dropna()

# === Base Class ===
class ForecastRule:
    def __init__(self, name):
        self.name = name

    def normalize_forecast(self, raw):
        avg_abs = raw.abs().mean()
        if avg_abs != 0:
            scaled = 10 * raw / avg_abs
        else:
            scaled = raw
        return scaled.clip(-20, 20)

    def forecast(self, prices, high=None, low=None, volume=None):
        raise NotImplementedError

# === Trend Following Rule ===
class TrendFollowing(ForecastRule):
    def __init__(self, ema_span, atr_window):
        name = f"Trend_EMA{ema_span}_ATR{atr_window}"
        super().__init__(name)
        self.ema_span = ema_span
        self.atr_window = atr_window

    def forecast(self, prices, high, low, volume=None):
        ema = prices.ewm(self.ema_span).mean()
        high = high.shift(1)
        low = low.shift(1)
        prev_close = prices.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()
        raw_forecast = 20 * (prices - ema) / (2 * atr)
        return self.normalize_forecast(raw_forecast)

# === EMASlope Rule (keep only longer versions in rule list) ===
class EMASlope(ForecastRule):
    def __init__(self, ema_span, slope_window):
        name = f"EMASLOPE_EMA{ema_span}_W{slope_window}"
        super().__init__(name)
        self.ema_span = ema_span
        self.slope_window = slope_window

    def forecast(self, prices, high=None, low=None, volume=None):
        ema = prices.ewm(self.ema_span).mean()
        ema_shifted = ema.shift(self.slope_window)
        slope = 1000 * (ema - ema_shifted) / prices
        return self.normalize_forecast(slope)

# === EMACrossoverStrength Rule ===
class EMACrossoverStrength(ForecastRule):
    def __init__(self, fast, slow):
        name = f"EMACrossover_{fast}_{slow}"
        super().__init__(name)
        self.fast = fast
        self.slow = slow

    def forecast(self, prices, high=None, low=None, volume=None):
        fast_ema = prices.ewm(self.fast).mean()
        slow_ema = prices.ewm(self.slow).mean()
        raw = 100 * (fast_ema - slow_ema) / prices
        return self.normalize_forecast(raw)

# === TwoATRReversal ===
class TwoATRReversal(ForecastRule):
    def __init__(self, ema_window, atr_window):
        name = f"MR_Z2ATR_{ema_window}_{atr_window}"
        super().__init__(name)
        self.ema_window = ema_window
        self.atr_window = atr_window

    def forecast(self, prices, high, low, volume=None):
        ema = prices.ewm(span=self.ema_window).mean()
        prev_close = prices.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()

        raw_forecast = -10 * (prices - ema) / (2 * atr)
        return self.normalize_forecast(raw_forecast)

# === RSIMeanReversion ===
class RSIMeanReversion(ForecastRule):
    def __init__(self, rsi_window=14):
        name = f"MR_RSI_{rsi_window}"
        super().__init__(name)
        self.rsi_window = rsi_window

    def forecast(self, prices, high=None, low=None, volume=None):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_window).mean()
        avg_loss = loss.rolling(self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        raw_forecast = 0.5 - rsi / 100
        return self.normalize_forecast(raw_forecast)

# === BollingerReversion ===
class BollingerReversion(ForecastRule):
    def __init__(self, window=20):
        name = f"MR_Bollinger_{window}"
        super().__init__(name)
        self.window = window

    def forecast(self, prices, high=None, low=None, volume=None):
        ma = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        raw_forecast = (prices - ma) / (2 * std)
        return self.normalize_forecast(raw_forecast)


# === 3. Volume Spike + Reversal Candle ===
class VolumeReversal(ForecastRule):
    def __init__(self, volume_window=20):
        name = f"VolumeReversal_{volume_window}"
        super().__init__(name)
        self.volume_window = volume_window

    def forecast(self, prices, high, low, volume):
        prices, volume = prices.ffill().bfill(), volume.ffill().bfill()
        if len(volume) < self.volume_window + 5:
            return pd.Series(np.nan, index=prices.index)

        avg_vol = volume.rolling(self.volume_window, min_periods=1).mean()
        vol_spike = volume > 1.5 * avg_vol

        body = (prices - prices.shift(1)).fillna(0)
        reversal = (body * body.shift(1) < 0)

        signal = vol_spike & reversal
        forecast = pd.Series(0, index=prices.index, dtype=float)
        forecast[signal & (body < 0)] = -20
        forecast[signal & (body > 0)] = 20
        return forecast.replace(0, np.nan).ffill(limit=1).fillna(0)






# === 4. Bollinger Band Break ===
class BollingerBreakout(ForecastRule):
    def __init__(self, window=20):
        name = f"BollingerBreakout_{window}"
        super().__init__(name)
        self.window = window

    def forecast(self, prices, high=None, low=None, volume=None):
        prices = prices.ffill().bfill()
        ma = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()

        upper = ma + 2 * std
        lower = ma - 2 * std
        body = (prices - prices.shift(1)).abs()

        signal_up = (prices > upper) & (body > std)
        signal_down = (prices < lower) & (body > std)

        raw = pd.Series(0, index=prices.index, dtype=float)
        raw[signal_up] = 20
        raw[signal_down] = -20

        return self.normalize_forecast(raw.replace(0, np.nan).ffill(limit=1).fillna(0))


# === 5. NR7 + Breakout ===
class NR7Breakout(ForecastRule):
    def __init__(self, lookback=7):
        name = f"NR7Breakout_{lookback}"
        super().__init__(name)
        self.lookback = lookback

    def forecast(self, prices, high, low, volume=None):
        high = high.ffill().bfill()
        low = low.ffill().bfill()
        prices = prices.ffill().bfill()

        range_ = high - low
        nr7 = range_ == range_.rolling(self.lookback).min()

        breakout_up = (prices > high.shift(1)) & nr7.shift(1)
        breakout_down = (prices < low.shift(1)) & nr7.shift(1)

        raw = pd.Series(0, index=prices.index, dtype=float)
        raw[breakout_up] = 15
        raw[breakout_down] = -15

        return self.normalize_forecast(raw.replace(0, np.nan).ffill(limit=1).fillna(0))


# === Rule 3: Bollinger Band Squeeze ===
class BollingerSqueeze(ForecastRule):
    def __init__(self, window=20):
        name = f"BollingerSqueeze_{window}"
        super().__init__(name)
        self.window = window

    def forecast(self, prices, high=None, low=None, volume=None):
        prices = prices.ffill().bfill()
        std = prices.rolling(self.window, min_periods=1).std()
        width = 2 * std
        avg_width = width.rolling(self.window, min_periods=1).mean().replace(0, np.nan)

        narrow = width < avg_width
        momentum = (prices - prices.shift(5)).fillna(0)

        raw = pd.Series(0, index=prices.index, dtype=float)
        raw[narrow & (momentum > 0)] = 10
        raw[narrow & (momentum < 0)] = -10

        return self.normalize_forecast(raw.fillna(0))






class MR_Bollinger_Reentry_Strength(ForecastRule):
    def __init__(self, window=20, std_dev=2):
        super().__init__("MR_Bollinger_Reentry_Strength")
        self.window = window
        self.std_dev = std_dev

    def forecast(self, prices, high, low, volume=None):
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std().replace(0, np.nan)

        dist = prices - mean
        strength = dist / std
        forecast = -20 * (strength / self.std_dev)

        return self.normalize_forecast(forecast)







# Rule 4: Low-volume rejection trap
class SR_LowVolumeTrap(ForecastRule):
    def __init__(self, lookback=20):
        super().__init__(f"SR_LowVolumeTrap_{lookback}")
        self.lookback = lookback

    def forecast(self, prices, high, low, volume):
        avg_vol = volume.rolling(self.lookback).mean()
        low_vol = volume < avg_vol * 0.7

        zone_high = high.shift(1).rolling(self.lookback).max()
        zone_low = low.shift(1).rolling(self.lookback).min()

        strength_high = (prices - zone_high) / prices
        strength_low = (zone_low - prices) / prices

        forecast = pd.Series(0.0, index=prices.index)
        forecast[(prices >= zone_high * 0.99) & low_vol] = -10 * strength_high[(prices >= zone_high * 0.99) & low_vol]
        forecast[(prices <= zone_low * 1.01) & low_vol] = 10 * strength_low[(prices <= zone_low * 1.01) & low_vol]

        return self.normalize_forecast(forecast)






# Rule 4: Volume Dry-Up Before Breakout
class VolumeDryUp(ForecastRule):
    def __init__(self, lookback=10):
        super().__init__(f"VolumeDryUp_{lookback}")
        self.lookback = lookback

    def forecast(self, prices, high, low, volume):
        avg_vol = volume.rolling(self.lookback).mean().replace(0, np.nan)
        vol_ratio = volume / avg_vol

        compression = 1 - vol_ratio
        price_range = high - low
        norm_range = (price_range / prices).clip(0, 1)

        signal = compression * norm_range
        direction = prices.diff().apply(np.sign)

        forecast = 10 * signal * direction
        return self.normalize_forecast(forecast)





# === Bootstrap Sharpe ===
def bootstrap_sharpe(forecast, returns, n_samples=1000):
    forecast = forecast.shift(1).dropna()
    returns = returns.loc[forecast.index]
    pnl = forecast * returns

    sr_list = []
    for _ in range(n_samples):
        sample = resample(pnl)
        if sample.std() != 0:
            daily_sr = sample.mean() / sample.std()
            sr_list.append(daily_sr * np.sqrt(252))  # Annualize
    return np.mean(sr_list)

# === Run Forecast & Evaluation ===
rules = [

    # === TrendFollowing (Price vs EMA normalized by ATR) ===
    TrendFollowing(50, 20), TrendFollowing(100, 30), TrendFollowing(200, 50), TrendFollowing(300, 70),

    # === EMASlope (EMA slope over lag window) ===
    EMASlope(100, 20), EMASlope(200, 40), EMASlope(300, 60),

    # === EMACrossoverStrength (Fast EMA vs Slow EMA spread) ===
    EMACrossoverStrength(20, 50), EMACrossoverStrength(50, 100), EMACrossoverStrength(100, 200), EMACrossoverStrength(150, 300),

    # === TwoATRReversal (Sharp reversal after 2x ATR move) ===
    TwoATRReversal(10, 10), TwoATRReversal(20, 20), TwoATRReversal(50, 30),

    # === RSIMeanReversion (Reversion based on RSI extremes) ===
    RSIMeanReversion(7), RSIMeanReversion(14), RSIMeanReversion(30),

    # === BollingerReversion (Price deviation from Bollinger mean) ===
    BollingerReversion(50),

    # === Volume Spike + Reversal Rules ===
    VolumeReversal(10), VolumeReversal(15), VolumeReversal(20), VolumeReversal(30),

    # === Bollinger Band Break ===
    BollingerBreakout(10), BollingerBreakout(15), BollingerBreakout(20), BollingerBreakout(30), BollingerBreakout(50),

    # === NR7 + Breakout ===
    NR7Breakout(3), NR7Breakout(5), NR7Breakout(7), NR7Breakout(10), NR7Breakout(15),

    # === Bollinger Band Squeeze ===
    BollingerSqueeze(20), BollingerSqueeze(30), BollingerSqueeze(50),
    
    # === Bollinger Band Squeeze ===
    BollingerSqueeze(20), BollingerSqueeze(30), BollingerSqueeze(50),
    
    # === MR_Bollinger_Reentry_Strength (Deviation from Bollinger mean) ===
    MR_Bollinger_Reentry_Strength(10), MR_Bollinger_Reentry_Strength(15), MR_Bollinger_Reentry_Strength(20),
    MR_Bollinger_Reentry_Strength(25), MR_Bollinger_Reentry_Strength(30),
    
    # === SR_LowVolumeTrap (rejection happens on low volume = trap) ===
    SR_LowVolumeTrap(10), SR_LowVolumeTrap(15), SR_LowVolumeTrap(20), SR_LowVolumeTrap(25), SR_LowVolumeTrap(30),

    # === VolumeDryUp (Low volume before breakout / compression signal) ===
    VolumeDryUp(10), VolumeDryUp(15), VolumeDryUp(20), VolumeDryUp(25), VolumeDryUp(30),
]
