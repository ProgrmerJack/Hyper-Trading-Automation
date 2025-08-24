#!/usr/bin/env python3
"""Technical indicator backtesting with realistic market data."""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_realistic_data(periods=2000):
    """Create realistic OHLCV data."""
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='h')
    
    # Generate realistic price movement
    returns = np.random.normal(0.0001, 0.02, periods)  # Slight upward bias
    price = 50000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame(index=dates)
    df['close'] = price
    df['open'] = df['close'].shift(1).fillna(50000)
    
    # Realistic high/low
    noise = np.random.uniform(0.002, 0.008, periods)
    df['high'] = np.maximum(df['open'], df['close']) * (1 + noise)
    df['low'] = np.minimum(df['open'], df['close']) * (1 - noise)
    
    # Volume with some correlation to price movement
    df['volume'] = np.random.uniform(500, 2000, periods) * (1 + np.abs(returns) * 20)
    
    return df

# Technical Indicators
def sma(series, period):
    """Simple Moving Average."""
    return series.rolling(period).mean()

def ema(series, period):
    """Exponential Moving Average."""
    return series.ewm(span=period).mean()

def rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands."""
    sma_val = sma(series, period)
    std_val = series.rolling(period).std()
    upper = sma_val + (std_val * std_dev)
    lower = sma_val - (std_val * std_dev)
    return upper, sma_val, lower

def macd(series, fast=12, slow=26, signal=9):
    """MACD Indicator."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Strategy Classes
class MovingAverageCrossover:
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow
        self.name = f"MA_{fast}_{slow}"
    
    def generate_signals(self, df):
        fast_ma = sma(df['close'], self.fast)
        slow_ma = sma(df['close'], self.slow)
        
        signals = pd.Series(0, index=df.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals

class RSIStrategy:
    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI_{period}"
    
    def generate_signals(self, df):
        rsi_vals = rsi(df['close'], self.period)
        
        signals = pd.Series(0, index=df.index)
        signals[rsi_vals < self.oversold] = 1
        signals[rsi_vals > self.overbought] = -1
        
        return signals

class BollingerBandsStrategy:
    def __init__(self, period=20, std_dev=2):
        self.period = period
        self.std_dev = std_dev
        self.name = f"BB_{period}_{std_dev}"
    
    def generate_signals(self, df):
        upper, middle, lower = bollinger_bands(df['close'], self.period, self.std_dev)
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] < lower] = 1  # Buy when price below lower band
        signals[df['close'] > upper] = -1  # Sell when price above upper band
        
        return signals

class MACDStrategy:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD_{fast}_{slow}_{signal}"
    
    def generate_signals(self, df):
        macd_line, signal_line, histogram = macd(df['close'], self.fast, self.slow, self.signal)
        
        signals = pd.Series(0, index=df.index)
        signals[(macd_line > signal_line) & (macd_line.shift() <= signal_line.shift())] = 1
        signals[(macd_line < signal_line) & (macd_line.shift() >= signal_line.shift())] = -1
        
        return signals

class MeanReversionStrategy:
    def __init__(self, lookback=20, threshold=2):
        self.lookback = lookback
        self.threshold = threshold
        self.name = f"MeanRev_{lookback}_{threshold}"
    
    def generate_signals(self, df):
        mean_price = sma(df['close'], self.lookback)
        std_price = df['close'].rolling(self.lookback).std()
        
        z_score = (df['close'] - mean_price) / std_price
        
        signals = pd.Series(0, index=df.index)
        signals[z_score < -self.threshold] = 1  # Buy when oversold
        signals[z_score > self.threshold] = -1   # Sell when overbought
        
        return signals

class MomentumStrategy:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.name = f"Momentum_{lookback}"
    
    def generate_signals(self, df):
        momentum = df['close'].pct_change(self.lookback)
        
        signals = pd.Series(0, index=df.index)
        signals[momentum > 0.05] = 1   # Buy on strong positive momentum
        signals[momentum < -0.05] = -1  # Sell on strong negative momentum
        
        return signals

class BreakoutStrategy:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.name = f"Breakout_{lookback}"
    
    def generate_signals(self, df):
        high_max = df['high'].rolling(self.lookback).max()
        low_min = df['low'].rolling(self.lookback).min()
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > high_max.shift()] = 1   # Buy on breakout above resistance
        signals[df['close'] < low_min.shift()] = -1   # Sell on breakdown below support
        
        return signals

class VolumeWeightedStrategy:
    def __init__(self, period=20):
        self.period = period
        self.name = f"VWAP_{period}"
    
    def generate_signals(self, df):
        # Simple VWAP calculation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(self.period).sum() / df['volume'].rolling(self.period).sum()
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > vwap] = 1
        signals[df['close'] < vwap] = -1
        
        return signals

# Backtesting Engine
class Backtester:
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def backtest(self, df, strategy):
        """Run backtest for a strategy."""
        signals = strategy.generate_signals(df)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        position_value = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Calculate current equity
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
            
            # Execute trades based on signals
            if current_signal == 1 and position <= 0:  # Buy signal
                # Close short position if any
                if position < 0:
                    trade_value = -position * current_price
                    commission_cost = trade_value * self.commission
                    capital += trade_value - commission_cost
                    trades.append({
                        'type': 'cover',
                        'price': current_price,
                        'quantity': -position,
                        'value': trade_value,
                        'commission': commission_cost
                    })
                    position = 0
                
                # Open long position
                if capital > 0:
                    position_size = capital * 0.95  # Use 95% of capital
                    shares = position_size / current_price
                    commission_cost = position_size * self.commission
                    
                    capital -= (position_size + commission_cost)
                    position = shares
                    position_value = position_size
                    
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'quantity': shares,
                        'value': position_size,
                        'commission': commission_cost
                    })
            
            elif current_signal == -1 and position >= 0:  # Sell signal
                # Close long position if any
                if position > 0:
                    trade_value = position * current_price
                    commission_cost = trade_value * self.commission
                    capital += trade_value - commission_cost
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'quantity': position,
                        'value': trade_value,
                        'commission': commission_cost
                    })
                    position = 0
                
                # Open short position
                if capital > 0:
                    position_size = capital * 0.95
                    shares = position_size / current_price
                    commission_cost = position_size * self.commission
                    
                    capital += position_size - commission_cost
                    position = -shares
                    position_value = -position_size
                    
                    trades.append({
                        'type': 'short',
                        'price': current_price,
                        'quantity': -shares,
                        'value': position_size,
                        'commission': commission_cost
                    })
        
        # Close final position
        if position != 0:
            final_price = df['close'].iloc[-1]
            trade_value = position * final_price
            commission_cost = abs(trade_value) * self.commission
            capital += trade_value - commission_cost
            
            trades.append({
                'type': 'close',
                'price': final_price,
                'quantity': position,
                'value': trade_value,
                'commission': commission_cost
            })
        
        final_equity = capital
        equity_curve.append(final_equity)
        
        return self.calculate_metrics(final_equity, equity_curve, trades)
    
    def calculate_metrics(self, final_equity, equity_curve, trades):
        """Calculate performance metrics."""
        # Basic metrics
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Equity curve analysis
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24)  # Hourly data
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity_series.expanding().max()
        drawdown = (peak - equity_series) / peak
        max_drawdown = drawdown.max() * 100
        
        # Trade analysis
        if trades:
            trade_returns = []
            for trade in trades:
                if trade['type'] in ['sell', 'cover', 'close']:
                    # Calculate return for this trade
                    trade_returns.append(trade['value'] - trade['commission'])
            
            winning_trades = len([r for r in trade_returns if r > 0])
            total_trades = len(trade_returns)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'final_equity': final_equity
        }

def main():
    """Run comprehensive backtesting."""
    print("Comprehensive Strategy Backtesting")
    print("=" * 60)
    
    # Generate market data
    print("Generating market data...")
    df = create_realistic_data(2000)
    print(f"Data: {len(df)} periods from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Initialize all strategies
    strategies = [
        MovingAverageCrossover(5, 20),
        MovingAverageCrossover(10, 30),
        MovingAverageCrossover(20, 50),
        RSIStrategy(14, 30, 70),
        RSIStrategy(21, 25, 75),
        BollingerBandsStrategy(20, 2),
        BollingerBandsStrategy(20, 1.5),
        MACDStrategy(12, 26, 9),
        MeanReversionStrategy(20, 2),
        MeanReversionStrategy(10, 1.5),
        MomentumStrategy(10),
        MomentumStrategy(20),
        BreakoutStrategy(20),
        BreakoutStrategy(50),
        VolumeWeightedStrategy(20),
    ]
    
    # Run backtests
    backtester = Backtester()
    results = {}
    
    print(f"\nTesting {len(strategies)} strategies...")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Return%':<8} {'Sharpe':<7} {'MaxDD%':<7} {'WinRate%':<9} {'Trades':<7}")
    print("-" * 80)
    
    for strategy in strategies:
        try:
            result = backtester.backtest(df, strategy)
            results[strategy.name] = result
            
            print(f"{strategy.name:<20} {result['total_return']:>7.2f} {result['sharpe_ratio']:>6.2f} "
                  f"{result['max_drawdown']:>6.2f} {result['win_rate']:>8.1f} {result['num_trades']:>6d}")
        
        except Exception as e:
            print(f"{strategy.name:<20} ERROR: {str(e)[:40]}")
            results[strategy.name] = {'error': str(e)}
    
    # Analysis and summary
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_results:
        returns = [r['total_return'] for r in successful_results.values()]
        sharpes = [r['sharpe_ratio'] for r in successful_results.values()]
        
        print("-" * 80)
        print(f"{'AVERAGE':<20} {np.mean(returns):>7.2f} {np.mean(sharpes):>6.2f}")
        
        # Best performers
        best_return = max(successful_results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(successful_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        lowest_dd = min(successful_results.items(), key=lambda x: x[1]['max_drawdown'])
        
        print(f"\nTop Performers:")
        print(f"Best Return:    {best_return[0]} ({best_return[1]['total_return']:.2f}%)")
        print(f"Best Sharpe:    {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
        print(f"Lowest Drawdown: {lowest_dd[0]} ({lowest_dd[1]['max_drawdown']:.2f}%)")
        
        # Save results
        import json
        with open('comprehensive_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to comprehensive_backtest_results.json")
    
    else:
        print("No successful backtests completed.")

if __name__ == "__main__":
    main()