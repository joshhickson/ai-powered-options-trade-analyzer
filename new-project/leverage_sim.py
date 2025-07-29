#!/usr/bin/env python3
"""
leverage_sim.py
Simulate a repeated BTC-collateral loan strategy with dynamic sizing.

User-defined rules (2025-07-29):
    • Start with 0.24 BTC, price = $118 000.
    • Post 0.12 BTC as collateral; 0.12 BTC stays in reserve.
    • Borrow $10 000 at 11.5 % APR (lump-sum interest at exit).
    • A loan cycle ends when price = entry_price + $30 000.
    • One margin call *always* happens per cycle at the worst drawdown
      (we size it using the 99 th-percentile historical drop).
    • If LTV would hit 90 %, move reserve BTC to cure (locked until exit).
    • Reserve size must equal collateral and remains locked until exit.
    • Each new loan’s principal is capped so that the reserve covers
      the fitted 99 % drawdown at that entry price.
    • Stop when free BTC ≥ 1.0.

Outputs:
    cycles_log.csv, btc_price_over_cycles.png, btc_owned_over_cycles.png
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & helpers
# ──────────────────────────────────────────────────────────────────────────────
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import yfinance; fall back to CSV if offline
try:
    import yfinance as yf
    ONLINE = True
except ImportError:
    ONLINE = False
    print("⚠️  yfinance not found.  Expecting btc_history.csv in this folder.")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 · Pull / load BTC price history
# ──────────────────────────────────────────────────────────────────────────────
def load_btc_history() -> pd.Series:
    """Return a daily Close price series indexed by date."""
    if ONLINE:
        btc = yf.download("BTC-USD", start="2010-07-17", progress=False)
        return btc["Adj Close"].dropna()
    else:
        csv = "btc_history.csv"
        if not os.path.exists(csv):
            sys.exit("BTC history CSV missing and yfinance unavailable.")
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
        return df["Close"].dropna()

prices = load_btc_history()

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 · For every day, compute “worst drawdown before +$30 K recovery”
# ──────────────────────────────────────────────────────────────────────────────
def worst_drop_until_recovery(price_series: pd.Series,
                              jump: float = 30000.0) -> pd.Series:
    """
    For each start date i, find j > i s.t. price_j ≥ price_i + jump.
    Record min(price_i…j) ÷ price_i − 1  (negative % drop).
    If recovery never occurs, drop the sample.
    """
    res = []
    dates = price_series.index
    p = price_series.values
    n = len(p)
    for i in range(n):
        target = p[i] + jump
        # Scan forward until recovery
        j = i
        min_p = p[i]
        while j < n and p[j] < target:
            min_p = min(min_p, p[j])
            j += 1
        if j == n:
            continue  # never recovered
        draw = (min_p / p[i]) - 1.0  # negative value
        res.append((dates[i], p[i], draw))
    df = pd.DataFrame(res, columns=["date", "price", "draw"])
    return df

draw_df = worst_drop_until_recovery(prices)

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 · Fit 99 th-percentile drawdown curve  draw99(price) = a·p^(-b)
# ──────────────────────────────────────────────────────────────────────────────
# Get 99th-percentile (largest magnitude) drop for price bins
bins = np.logspace(np.log10(draw_df.price.min()),
                   np.log10(draw_df.price.max()), 40)
draw_df["bin"] = pd.cut(draw_df.price, bins=bins)
bin_stats = draw_df.groupby("bin").agg(
    p=("price", "median"),
    d99=("draw", lambda x: np.percentile(x, 1))  # 1st percentile = worst 1%
).dropna()

# Fit a log-log linear regression  ln|d| = ln a  – b ln p
y = np.log(np.abs(bin_stats.d99.values))
x = np.log(bin_stats.p.values)
b, ln_a = np.polyfit(x, y, 1) * np.array([-1, 1])  # adjust sign
a = math.exp(ln_a)

def draw99(price: float) -> float:
    """Return the 99 % worst expected drawdown (as +fraction) for a price."""
    return a * (price ** (-b))  # positive number like 0.25 → 25 %

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 · Simulation parameters & state
# ──────────────────────────────────────────────────────────────────────────────
params = {
    "start_price": 118_000.0,
    "start_btc_free": 0.24,
    "start_collateral_btc": 0.12,
    "loan_rate": 0.115,        # APR lump-sum
    "exit_jump": 30_000.0,
    "margin_LTV": 0.90,
}

state = {
    "cycle": 0,
    "price": params["start_price"],
    "free_btc": params["start_btc_free"] - params["start_collateral_btc"],
    "collat_btc": params["start_collateral_btc"],  # locked reserve = collat
    "loan": 10_000.0,
    "btc_goal": 1.0,
}

records = []

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 · Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def btc_needed_for_cure(price: float, loan: float) -> float:
    """
    Given entry price and loan size, compute how much BTC must be added
    to keep LTV ≤ 90 % in a 99 % drawdown.
    """
    drop_pct = draw99(price)        # e.g. 0.25 → –25 %
    worst_price = price * (1 - drop_pct)
    # LTV = loan / (collateral * worst_price)
    # Want collateral s.t. loan / (collat * worst_price) ≤ 0.90
    return loan / (0.90 * worst_price)

def cap_next_loan(price: float, reserve_btc: float) -> float:
    """
    Find the max loan principal such that reserve ≥ needed cure BTC.
    Solve inverse of btc_needed_for_cure.
    """
    drop_pct = draw99(price)
    worst_price = price * (1 - drop_pct)
    return reserve_btc * worst_price * 0.90

# ──────────────────────────────────────────────────────────────────────────────
# Section 6 · Event-driven cycle loop
# ──────────────────────────────────────────────────────────────────────────────
while state["free_btc"] < state["btc_goal"]:
    state["cycle"] += 1
    entry_price = state["price"]
    loan = state["loan"]

    # Step 1: compute cure requirement & check reserve
    needed_btc = btc_needed_for_cure(entry_price, loan)
    if needed_btc > state["collat_btc"]:
        # Reserve insufficient: shrink loan
        loan = cap_next_loan(entry_price, state["collat_btc"])
        state["loan"] = loan

    # Step 2: immediate margin call at worst drawdown
    state["collat_btc"] += needed_btc
    state["free_btc"] -= needed_btc  # moved to collateral locker

    # Step 3: price appreciation +$30 K
    exit_price = entry_price + params["exit_jump"]
    state["price"] = exit_price

    # BTC purchased with loan (assuming instant buy at entry price)
    btc_bought = loan / entry_price

    # Interest lump-sum
    interest = loan * state["cycle"] * 0  # placeholder (simple loop-year calc)
    payoff = loan + loan * params["loan_rate"]  # 1-yr APR lump

    # Sell BTC equal to payoff
    btc_sold = payoff / exit_price
    gain_btc = btc_bought - btc_sold

    # Step 4: release all collateral back to free BTC
    state["free_btc"] += gain_btc + state["collat_btc"]
    state["collat_btc"] = params["start_collateral_btc"]  # reset reserve

    # Step 5: set next loan equal to cap for new price
    state["loan"] = cap_next_loan(exit_price, state["collat_btc"])

    # Log cycle
    records.append({
        "cycle": state["cycle"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "loan": loan,
        "needed_cure_btc": needed_btc,
        "btc_free": state["free_btc"],
    })

    # Safety break
    if state["cycle"] > 500:
        print("Aborting after 500 cycles (should never happen).")
        break

# ──────────────────────────────────────────────────────────────────────────────
# Section 7 · Save results
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv("cycles_log.csv", index=False)

plt.figure()
plt.plot(df.cycle, df.exit_price, marker="o")
plt.xlabel("Cycle")
plt.ylabel("BTC Exit Price ($)")
plt.title("BTC Price per Cycle")
plt.savefig("btc_price_over_cycles.png", dpi=150)

plt.figure()
plt.plot(df.cycle, df.btc_free, marker="o", color="orange")
plt.xlabel("Cycle")
plt.ylabel("Unencumbered BTC")
plt.title("BTC Held vs. Cycle")
plt.savefig("btc_owned_over_cycles.png", dpi=150)

print("Simulation complete.  Files saved:\n"
      "  • cycles_log.csv\n"
      "  • btc_price_over_cycles.png\n"
      "  • btc_owned_over_cycles.png")