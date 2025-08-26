import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load backtest results
def load_backtest_results(file_path: str):
    """Load backtest results from JSON."""
    with open(file_path, 'r') as f:
        results = json.load(f)
    # Convert trade logs to DataFrames
    # The trade logs are now under 'trade_logs' key by strategy name
    trade_logs = {}
    if 'trade_logs' in results:
        for strategy_name, log in results['trade_logs'].items():
            if log is not None:
                trade_logs[strategy_name] = pd.DataFrame(log)
    results['trade_logs'] = trade_logs
    return results

# Main analysis function
def analyze_trade_logs(results: dict):
    """Analyze trade logs from backtest results."""
    # Create a figure for subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get trade logs
    trade_logs = results.get('trade_logs', {})
    
    # Plot 1: Win rate by strategy
    win_rates = {}
    for strategy_name, trades_df in trade_logs.items():
        if not trades_df.empty:
            # For simplicity, assume we have a 'profit' column
            if 'profit' in trades_df.columns:
                winning_trades = trades_df[trades_df['profit'] > 0]
                win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                win_rates[strategy_name] = win_rate
            else:
                win_rates[strategy_name] = 0.0
    
    if win_rates:
        pd.Series(win_rates).plot.bar(ax=axes[0,0], title='Win Rate by Strategy')
    
    # Plot 2: Confidence distribution
    for strategy_name, trades_df in trade_logs.items():
        if not trades_df.empty:
            if 'profit' in trades_df.columns and 'confidence' in trades_df.columns:
                # Mark winning trades
                trades_df['outcome'] = trades_df['profit'].apply(lambda x: 'win' if x > 0 else 'loss')
                sns.kdeplot(data=trades_df, x='confidence', hue='outcome', ax=axes[0,1], common_norm=False)
    
    # Plot 3: Action distribution
    for strategy_name, trades_df in trade_logs.items():
        if not trades_df.empty:
            if 'action' in trades_df.columns:
                action_counts = trades_df['action'].value_counts()
                action_counts.plot.pie(ax=axes[1,0], autopct='%1.1f%%', title='Action Distribution')
    
    # Plot 4: Profit distribution
    for strategy_name, trades_df in trade_logs.items():
        if not trades_df.empty:
            if 'profit' in trades_df.columns:
                sns.histplot(trades_df['profit'], ax=axes[1,1], kde=True)
                axes[1,1].set_title('Profit Distribution')
    
    plt.tight_layout()
    plt.savefig('trade_log_analysis.png')
    print("Saved analysis to trade_log_analysis.png")

if __name__ == "__main__":
    results = load_backtest_results("backtest_report.json")
    analyze_trade_logs(results)
