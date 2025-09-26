# Mean Reversion Algo Trading Strategy

## Brief Description / Motivation
This project applies the principles of *Systematic Trading* by Robert Carver using the assistance of AI tools to code different rules based on multiple hypotheses, aiming to achieve a diversified rule-based strategy. It also uses **bootstrapping** and **average correlation** to determine forecast weights and the diversification number, reducing the risk of overfitting. The approach is fully mechanical and backtested over historical market data. Finally, all rules are combined to form a working strategy with a positive Sharpe ratio.

---

## Technologies / Tools Used
- **Python**  
- **Python Libraries:**  
  - `yfinance` → Market data  
  - `pandas` → Data manipulation  
  - `numpy` → Numerical calculations  
  - `resample` (`sklearn.utils`) → Bootstrapping

---

## Getting Started / Usage
1. Install required libraries:
   ```bash
   pip install yfinance pandas numpy scikit-learn

---

## Results / Performance
![image alt](https://github.com/Jordanbr2/Robert-Carver-Algo-Trading-System/blob/d43dbbb80b943ec06c0a9f387ab09559d82f63de/Combined%20Forecast.jpg)

---
![image alt](https://github.com/Jordanbr2/Robert-Carver-Algo-Trading-System/blob/d43dbbb80b943ec06c0a9f387ab09559d82f63de/Rules%20SR.jpg)

---

## References/Links
[Systematic Trading by Robert Carver](https://www.amazon.ca/Systematic-Trading-designing-trading-investing/dp/0857194453)
