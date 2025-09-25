def estimate_diversification_multiplier(price_data, tickers):
    returns = price_data[tickers].pct_change().dropna()
    corr_matrix = returns.corr()

    # Remove self-correlation, floor at 0
    avg_corr = corr_matrix.where(~np.eye(len(tickers), dtype=bool)).stack().clip(lower=0).mean()

    # Use Carver's table approximation
    if avg_corr >= 0.75:
        dm = 1.10
    elif avg_corr >= 0.5:
        dm = 1.27
    elif avg_corr >= 0.25:
        dm = 1.51
    else:
        dm = 2.00

    # Apply cap
    return min(dm, 2.5), avg_corr
