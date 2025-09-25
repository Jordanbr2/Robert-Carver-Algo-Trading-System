# === Robust Sharpe that always returns a dictionary ===
def bootstrap_sharpe_dict(forecast, returns, n_samples=1000):
    forecast = forecast.shift(1).dropna()
    returns = returns.loc[forecast.index]
    pnl = forecast * returns

    sr_list = []
    for _ in range(n_samples):
        sample = pnl.sample(frac=1, replace=True)
        if sample.std() != 0:
            daily_sr = sample.mean() / sample.std()
            sr_list.append(daily_sr * np.sqrt(252))
    if len(sr_list) == 0:
        return {
            "mean_sharpe": np.nan,
            "std_sharpe": np.nan,
            "5%_quantile": np.nan,
            "95%_quantile": np.nan
        }
    return {
        "mean_sharpe": np.mean(sr_list),
        "std_sharpe": np.std(sr_list),
        "5%_quantile": np.percentile(sr_list, 5),
        "95%_quantile": np.percentile(sr_list, 95)
    }

# === Combine forecasts and evaluate ===
combined_results = []

for ticker in tickers:
    prices = price_data[ticker]
    high = high_data[ticker]
    low = low_data[ticker]
    volume = volume_data[ticker]
    returns = prices.pct_change().dropna()

    rule_forecasts = {}

    for rule in rules:
        try:
            fcast = rule.forecast(prices, high, low, volume) * 1.73
            if fcast.isna().all() or (fcast == 0).all():
                continue
            rule_forecasts[rule.name] = fcast
        except Exception:
            continue

    combined_forecast = pd.Series(0.0, index=prices.index)

    for rule_name, weight in forecast_weights.items():
        if rule_name in rule_forecasts:
            combined_forecast = combined_forecast.add(weight * rule_forecasts[rule_name], fill_value=0)

    combined_forecast = combined_forecast.dropna()
    common = combined_forecast.index.intersection(returns.index)
    if len(common) == 0:
        continue

    sr_dict = bootstrap_sharpe_dict(combined_forecast.loc[common], returns.loc[common])
    combined_results.append({
        "Asset": ticker,
        **sr_dict
    })

# === Create DataFrame and display ===
df_combined = pd.DataFrame(combined_results)
df_combined = df_combined.sort_values(by="mean_sharpe", ascending=False)
print("\n=== Combined Forecast Sharpe Ratios ===\n")
print(df_combined)
