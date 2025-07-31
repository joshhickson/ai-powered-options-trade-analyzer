#!/usr/bin/env python3
"""
btc_volatility_analyzer_multi_res.py

Analyzes Bitcoin volatility across multiple time resolutions (Daily, Hourly,
and 15-Minute) to create a correlated, high-dynamic-range view of market
crashes. This "blade of the knife" analysis helps identify if high-frequency
micro-crashes are predictive of larger, macro-level downturns.

Includes "V-Shape Analysis" to compare the velocity of a crash to its
subsequent recovery, and "Run-up Analysis" to measure the velocity of the
price increase leading into a crash peak.

Also includes a GARCH(1,1) model to analyze and forecast volatility regimes,
adding a layer of predictive context to the descriptive analysis.
"""

import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
# Simple volatility analysis without external GARCH library

def get_btc_price_history(interval: int, periods: int = 720) -> pd.Series:
    """
    Fetches high-resolution Bitcoin price history from the Kraken API.
    """
    timeframe_map = {1440: "days", 60: "hours", 15: "minutes"}
    timeframe_unit = timeframe_map.get(interval, "periods")
    duration = periods
    if timeframe_unit == "days": duration = periods
    elif timeframe_unit == "hours": duration = (interval * periods) / 60
    elif timeframe_unit == "minutes": duration = (interval * periods)

    print(f"📈 Fetching {periods} data points at {interval}-min resolution ({duration:.0f} {timeframe_unit})...")

    url = "https://api.kraken.com/0/public/OHLC"
    params = {'pair': 'XBTUSD', 'interval': interval}

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get('error'):
            print(f"   ❌ Kraken API Error: {', '.join(data['error'])}")
            return pd.Series(dtype=float)

        pair_name = list(data['result'].keys())[0]
        ohlc_data = data['result'][pair_name]

        df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time').sort_index().tail(periods)

        btc_series = df['close'].astype(float)
        btc_series.name = 'Close'

        print(f"   ✅ Successfully loaded {len(btc_series)} data points.")
        return btc_series

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network or HTTP Error: {e}")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"   ❌ An unexpected error occurred: {e}")
        return pd.Series(dtype=float)


def find_crash_events(prices: pd.Series, resolution_str: str) -> pd.DataFrame:
    """
    Identifies crash events using the HDR method for a given price series.
    """
    sigma_levels = [3.5, 3.0, 2.5, 2.0]
    window_days = 90 if resolution_str == 'Daily' else 7

    if len(prices.index) < 2: return pd.DataFrame()

    seconds_per_period = (prices.index[1] - prices.index[0]).total_seconds()
    periods_per_day = 86400 / seconds_per_period
    window_periods = int(window_days * periods_per_day)

    print(f"   🔬 Analyzing {resolution_str} data (Window: {window_days} days / {window_periods} periods)...")

    returns = prices.pct_change()
    rolling_mean = returns.rolling(window=window_periods, min_periods=int(window_periods/2)).mean()
    rolling_std = returns.rolling(window=window_periods, min_periods=int(window_periods/2)).std()

    all_events = {}
    for sigma in sorted(sigma_levels, reverse=True):
        crash_threshold = rolling_mean - (sigma * rolling_std)
        crash_periods = returns[returns < crash_threshold].index
        for period in crash_periods:
            if period not in all_events: all_events[period] = sigma

    if not all_events: return pd.DataFrame()

    distinct_events = []
    sorted_periods = sorted(all_events.keys())
    grouping_timedelta = pd.Timedelta(days=7) if resolution_str == 'Daily' else pd.Timedelta(hours=24)

    if sorted_periods:
        current_event_start = sorted_periods[0]
        max_sigma_in_event = all_events[current_event_start]
        for i in range(1, len(sorted_periods)):
            period = sorted_periods[i]
            if (period - sorted_periods[i-1]) <= grouping_timedelta:
                max_sigma_in_event = max(max_sigma_in_event, all_events[period])
            else:
                distinct_events.append({'TriggerDate': current_event_start, 'Severity(Sigma)': max_sigma_in_event})
                current_event_start = period
                max_sigma_in_event = all_events[period]
        distinct_events.append({'TriggerDate': current_event_start, 'Severity(Sigma)': max_sigma_in_event})

    return pd.DataFrame(distinct_events)


def characterize_crashes(prices: pd.Series, events_df: pd.DataFrame, resolution_str: str) -> pd.DataFrame:
    """
    Analyzes each crash event to determine its key characteristics, including
    the run-up before the peak and the recovery after the trough.
    """
    print(f"   📊 Characterizing {len(events_df)} {resolution_str} crash triggers...")
    crash_data = []

    for _, event in events_df.iterrows():
        trigger_date = event['TriggerDate']
        lookback_days = 90 if resolution_str == 'Daily' else 30

        # Find the peak before the crash trigger
        peak_lookback = prices.loc[trigger_date - pd.Timedelta(days=lookback_days):trigger_date]
        if peak_lookback.empty: continue
        peak_date = peak_lookback.idxmax()
        peak_price = peak_lookback.max()

        # --- 1. Run-up Analysis (The "Reverse Derivative") ---
        runup_lookback = prices.loc[peak_date - pd.Timedelta(days=30):peak_date]
        if runup_lookback.empty: continue
        runup_start_date = runup_lookback.idxmin()
        runup_start_price = runup_lookback.min()

        runup_duration_hours = (peak_date - runup_start_date).total_seconds() / 3600
        runup_amplitude = (peak_price / runup_start_price) - 1.0
        runup_velocity = (runup_amplitude * 100) / runup_duration_hours if runup_duration_hours > 0 else 0

        # --- 2. Crash Analysis ---
        search_period = prices.loc[peak_date:]
        trough_price, trough_date = peak_price, peak_date
        temp_recovery_price = peak_price * 0.95
        for date, price in search_period.items():
            if price < trough_price:
                trough_price, trough_date = price, date
            if price >= temp_recovery_price and trough_date > peak_date:
                break

        if trough_date <= peak_date: continue

        crash_amplitude = (trough_price / peak_price) - 1.0
        crash_duration_hours = (trough_date - peak_date).total_seconds() / 3600
        crash_velocity = (crash_amplitude * 100) / crash_duration_hours if crash_duration_hours > 0 else (crash_amplitude * 100)

        # --- 3. Recovery (V-Shape) Analysis ---
        recovery_date, recovery_duration_hours, recovery_velocity, velocity_ratio = None, np.nan, np.nan, np.nan
        recovery_search_period = prices.loc[trough_date:]
        for date, price in recovery_search_period.items():
            if price >= peak_price:
                recovery_date = date
                break

        if recovery_date:
            recovery_duration_hours = (recovery_date - trough_date).total_seconds() / 3600
            recovery_amplitude = (peak_price / trough_price) - 1.0
            if recovery_duration_hours > 0:
                recovery_velocity = (recovery_amplitude * 100) / recovery_duration_hours
                if crash_velocity != 0:
                    velocity_ratio = abs(recovery_velocity / crash_velocity)

        crash_data.append({
            'Resolution': resolution_str, 'PeakDate': peak_date, 'TroughDate': trough_date, 'RecoveryDate': recovery_date,
            'Severity(Sigma)': event['Severity(Sigma)'], 'PeakPrice': peak_price,
            'RunUpVelocity(%/Hour)': runup_velocity, 'CrashVelocity(%/Hour)': crash_velocity,
            'RecoveryVelocity(%/Hour)': recovery_velocity, 'VelocityRatio': velocity_ratio,
            'Amplitude(%)': crash_amplitude * 100, 'CrashDuration(Hours)': crash_duration_hours,
            'RecoveryDuration(Hours)': recovery_duration_hours, 'RunUpDuration(Hours)': runup_duration_hours
        })

    if not crash_data: return pd.DataFrame()

    df = pd.DataFrame(crash_data)
    df = df.sort_values('Severity(Sigma)', ascending=False).drop_duplicates(subset=['PeakDate'], keep='first')
    df = df.sort_values(by='PeakDate').reset_index(drop=True)

    print(f"   ✅ Found {len(df)} unique crash events.")
    return df


def run_analysis_for_resolution(interval: int, resolution_str: str) -> pd.DataFrame:
    """
    Orchestrates the analysis for a single time resolution.
    """
    prices = get_btc_price_history(interval=interval)
    if prices is None or prices.empty:
        return pd.DataFrame()

    crash_events_df = find_crash_events(prices, resolution_str)
    if crash_events_df.empty:
        return pd.DataFrame()

    crashes_df = characterize_crashes(prices, crash_events_df, resolution_str)
    return crashes_df


def analyze_volatility_regime(prices: pd.Series, crashes_df: pd.DataFrame, export_dir: Path):
    """
    Simple volatility analysis using rolling standard deviation (replaces GARCH).
    """
    print("🧠 Performing simple volatility regime analysis...")
    returns = prices.pct_change().dropna() * 100
    
    # Calculate rolling volatility (30-day window)
    rolling_volatility = returns.rolling(window=30).std()
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(18, 9))

    # Plot price on the left y-axis
    ax1.plot(prices.index, prices, color='black', alpha=0.7, linewidth=1.0, label='BTC Price (Log Scale)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Price (USD, Log Scale)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for volatility
    ax2 = ax1.twinx()
    ax2.plot(rolling_volatility.index, rolling_volatility, color='purple', alpha=0.6, label='30-Day Rolling Volatility')
    ax2.set_ylabel('Rolling Daily Volatility (%)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Highlight our detected crash events
    for _, row in crashes_df.iterrows():
        if row['Resolution'] == 'Daily':
            ax1.axvspan(row['PeakDate'], row['TroughDate'], color='red', alpha=0.3, zorder=0)

    ax1.set_title('Simple Volatility Regime Analysis vs. Detected Crashes', fontsize=18)
    fig.tight_layout()
    fig.savefig(export_dir / '5_simple_volatility_regime.png', dpi=200)
    plt.close(fig)
    print("   - Saved: 5_simple_volatility_regime.png")


def generate_visualizations(daily_prices: pd.Series, all_crashes_df: pd.DataFrame, export_dir: Path):
    """
    Generates all visualizations for the analysis.
    """
    print("🎨 Generating visualizations...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Composite "Blade of the Knife" Chart
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(daily_prices.index, daily_prices, color='black', alpha=0.7, linewidth=1.0, label='Daily Price')
    ax.set_yscale('log')
    ax.set_title('Multi-Resolution Volatility Analysis', fontsize=18)
    colors = {'Daily': 'red', 'Hourly': 'orange', '15-Minute': 'yellow'}
    alphas = {'Daily': 0.2, 'Hourly': 0.4, '15-Minute': 0.6}
    for res, df in all_crashes_df.groupby('Resolution'):
        for _, row in df.iterrows():
            ax.axvspan(row['PeakDate'], row['TroughDate'], color=colors[res], alpha=alphas[res], label=f'_{res}')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[res], alpha=min(1.0, alphas[res] + 0.3), label=f'{res} Crashes')
                       for res in ['Daily', 'Hourly', '15-Minute'] if res in all_crashes_df['Resolution'].unique()]
    if legend_elements: ax.legend(handles=legend_elements, loc='upper left')
    fig.tight_layout()
    fig.savefig(export_dir / '1_composite_volatility_analysis.png', dpi=200)
    plt.close(fig)
    print("   - Saved: 1_composite_volatility_analysis.png")

    # 2. V-Shape Velocity Ratio Chart
    v_shape_df = all_crashes_df.dropna(subset=['VelocityRatio'])
    if not v_shape_df.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        cmap = plt.get_cmap('coolwarm')
        scatter = ax.scatter(v_shape_df['PeakDate'], v_shape_df['VelocityRatio'], c=v_shape_df['VelocityRatio'], cmap=cmap, s=100, alpha=0.8, vmin=0, vmax=2)
        ax.axhline(y=1.0, color='grey', linestyle='--', label='Symmetric Recovery (Ratio=1.0)')
        ax.set_yscale('log')
        ax.set_title('V-Shape Analysis: Recovery vs. Crash Velocity', fontsize=16)
        cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Ratio (<1 = Slow Recovery, >1 = Fast Recovery)')
        ax.legend()
        fig.tight_layout(); fig.savefig(export_dir / '2_v_shape_velocity_ratio.png', dpi=150)
        plt.close(fig)
        print("   - Saved: 2_v_shape_velocity_ratio.png")

    # 3. Run-up vs. Crash Velocity Chart
    runup_df = all_crashes_df.dropna(subset=['RunUpVelocity(%/Hour)', 'CrashVelocity(%/Hour)'])
    if not runup_df.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        cmap = plt.get_cmap('viridis_r')
        scatter = ax.scatter(runup_df['RunUpVelocity(%/Hour)'], runup_df['CrashVelocity(%/Hour)'], c=runup_df['Severity(Sigma)'], cmap=cmap, s=100, alpha=0.8)
        ax.set_title('Run-up Velocity vs. Crash Velocity ("Blow-off Top" Analysis)', fontsize=16)
        ax.set_xlabel('Pre-Crash Run-up Velocity (% per Hour)')
        ax.set_ylabel('Crash Velocity (% per Hour)')
        cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Severity (Sigma)')
        fig.tight_layout(); fig.savefig(export_dir / '3_runup_vs_crash_velocity.png', dpi=150)
        plt.close(fig)
        print("   - Saved: 3_runup_vs_crash_velocity.png")

    # 4. Detailed V-Shape Bubble Chart
    bubble_df = all_crashes_df.dropna(subset=['CrashVelocity(%/Hour)', 'RecoveryVelocity(%/Hour)', 'PeakPrice'])
    if not bubble_df.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        cmap = plt.get_cmap('plasma')
        scatter = ax.scatter(bubble_df['CrashVelocity(%/Hour)'], bubble_df['RecoveryVelocity(%/Hour)'], 
                             s=bubble_df['PeakPrice']/100, # Scale bubble size by price
                             c=bubble_df['Severity(Sigma)'], cmap=cmap, alpha=0.7)
        ax.set_title('Detailed V-Shape Analysis', fontsize=16)
        ax.set_xlabel('Crash Velocity (% per Hour)')
        ax.set_ylabel('Recovery Velocity (% per Hour)')
        cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Severity (Sigma)')
        handles, labels = scatter.legend_elements("sizes", num=5, func=lambda s: s*100)
        ax.legend(handles, labels, title="Peak Price (USD)")
        fig.tight_layout(); fig.savefig(export_dir / '4_detailed_v_shape_bubble_chart.png', dpi=150)
        plt.close(fig)
        print("   - Saved: 4_detailed_v_shape_bubble_chart.png")

    print("   ✅ Visualization generation complete.")


def main():
    """Main function to run the multi-resolution volatility analysis."""
    print("🚀 Starting Multi-Resolution Bitcoin Volatility Analyzer")
    print("=" * 60)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("exports") / f"multi_res_analysis_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created export directory: {export_dir}")
    print("-" * 60)

    daily_crashes = run_analysis_for_resolution(interval=1440, resolution_str='Daily')
    print("-" * 60)
    hourly_crashes = run_analysis_for_resolution(interval=60, resolution_str='Hourly')
    print("-" * 60)
    minute_crashes = run_analysis_for_resolution(interval=15, resolution_str='15-Minute')
    print("-" * 60)

    all_crashes_df = pd.concat([daily_crashes, hourly_crashes, minute_crashes], ignore_index=True)

    if all_crashes_df.empty:
        print("✅ Analysis complete. No significant crashes found at any resolution.")
        return

    all_crashes_df = all_crashes_df.sort_values('Severity(Sigma)', ascending=False)
    all_crashes_df['PeakDay'] = all_crashes_df['PeakDate'].dt.floor('24h')
    all_crashes_df = all_crashes_df.drop_duplicates(subset=['PeakDay'], keep='first')
    all_crashes_df = all_crashes_df.drop(columns=['PeakDay'])
    all_crashes_df = all_crashes_df.sort_values(by='PeakDate').reset_index(drop=True)

    print("\n--- Combined Crash Analysis Summary ---")
    display_cols = ['Resolution', 'PeakDate', 'Severity(Sigma)', 'RunUpVelocity(%/Hour)', 'CrashVelocity(%/Hour)', 'VelocityRatio']
    print(all_crashes_df[display_cols].round(2))
    print("-" * 60)

    print("💾 Exporting combined data...")
    all_crashes_df.to_csv(export_dir / 'combined_crash_data.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
    all_crashes_df.to_json(export_dir / 'combined_crash_data.json', orient='records', indent=4, date_format='iso')
    print("   ✅ Saved combined_crash_data.csv and .json")
    print("-" * 60)

    daily_prices = get_btc_price_history(interval=1440)
    if not daily_prices.empty:
        generate_visualizations(daily_prices, all_crashes_df, export_dir)
        analyze_volatility_regime(daily_prices, all_crashes_df[all_crashes_df['Resolution'] == 'Daily'], export_dir)
    else:
        print("⚠️ Could not generate composite chart or GARCH analysis due to daily price data failure.")

    print("\n" + "=" * 60)
    print("✅ Analysis finished successfully!")
    print(f"   All results have been saved to the '{export_dir}' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
