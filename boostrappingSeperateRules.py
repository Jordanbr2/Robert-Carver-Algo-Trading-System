results = []

for ticker in tickers:
    prices = price_data[ticker]
    high = high_data[ticker]
    low = low_data[ticker]
    volume = volume_data[ticker]
    returns = prices.pct_change().dropna()

    for rule in rules:
        fcast = rule.forecast(prices,high,low,volume).dropna()
        common = fcast.index.intersection(returns.index)
        sr = bootstrap_sharpe(fcast.loc[common], returns.loc[common])
        results.append({
            "Asset": ticker,
            "Rule": rule.name,
            "Sharpe": sr
        })

# === Analyze Results ===
df_results = pd.DataFrame(results)

if not df_results.empty:
    avg_sr = df_results.groupby("Rule")["Sharpe"].mean().sort_values(ascending=False)
    df_results["Average_SR"] = df_results["Rule"].map(avg_sr)

    print("\n=== Average Sharpe Ratios (Across Assets) ===\n")
    print(avg_sr)

    print("\n=== Full Sharpe Ratio Table ===\n")
    print(df_results.sort_values(by=["Average_SR", "Asset"], ascending=[False, True]))
else:
    print("No results to display. Make sure rules and data are correctly set up.")
