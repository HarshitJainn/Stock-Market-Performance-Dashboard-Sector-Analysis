full_cum = ((1 + sector_returns).cumprod().iloc[-1] - 1).sort_values(ascending=False)
print("=== Full-period cumulative returns (all data) ===")
print(full_cum.to_string())
