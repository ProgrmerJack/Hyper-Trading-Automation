#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework
Advanced backtesting with multiple strategies, risk metrics, and performance analysis
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

# Import available modules
try:
    from hypertrader.backtester.advanced_engine import AdvancedBacktestEngine
except ImportError:
    AdvancedBacktestEngine = None

try:
    from hypertrader.strategies.advanced_ml import create_ml_strategy
except ImportError:
    create_ml_strategy = None

try:
    from hypertrader.strategies.alpha_strategies import create_alpha_strategy
except ImportError:
    create_alpha_strategy = None

class ComprehensiveBacktester:
    """Advanced backtesting framework with multiple strategies and analysis."""
    
    def __init__(self, initial_capital: float = 100.0):
        self.initial_capital = initial_capital
        self.results = {}
        self.strategies = {}
        self.performance_metrics = {}
        
    def add_strategy(self, name: str, strategy_type: str, strategy_category: str = "technical"):
        """Add strategy to backtesting suite."""
        if strategy_category == "ml":
            self.strategies[name] = create_ml_strategy(strategy_type)
        elif strategy_category == "alpha":
            self.strategies[name] = create_alpha_strategy(strategy_type)
        else:
            # Default technical strategies handled by main system
            self.strategies[name] = {"type": strategy_type, "category": strategy_category}
    
    def generate_synthetic_data(self, days: int = 365, symbol: str = "BTC-USD") -> pd.DataFrame:
        """Generate synthetic market data for backtesting."""
        np.random.seed(42)  # For reproducible results
        
        # Start with a base price
        start_price = 50000
        dates = pd.date_range(start='2023-01-01', periods=days*24, freq='H')  # Hourly data
        
        # Generate price movements with trend and volatility
        returns = np.random.normal(0.0001, 0.02, len(dates))  # Small positive drift
        
        # Add some trend and cycles
        trend = np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7)) * 0.005  # Weekly cycle
        macro_trend = np.arange(len(dates)) * 0.00001  # Small upward trend
        
        returns += trend + macro_trend
        
        # Calculate prices
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices)))),
            'close': prices,
            'volume': np.random.lognormal(15, 1, len(prices))  # Realistic volume distribution
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    def calculate_performance_metrics(self, equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        if len(equity_curve) < 2:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annual_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365*24 / len(equity_curve)) - 1) * 100
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(365*24) * 100  # Annualized
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(365*24)) if returns.std() > 0 else 0
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Trade analysis
        if len(trades) > 0:
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Advanced metrics
        sortino_ratio = returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2),
            'total_trades': len(trades),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'final_equity': round(equity_curve.iloc[-1], 2)
        }
    
    def simulate_strategy_performance(self, data: pd.DataFrame, strategy_name: str) -> dict:
        """Simulate individual strategy performance."""
        equity = self.initial_capital
        equity_curve = [equity]
        trades = []
        position = 0
        entry_price = 0
        trade_log = []  # Detailed log of trade decisions
        
        # Risk management and execution parameters (Medium risk for $1000 target)
        risk_pct = 0.03  # allocate up to 3% of equity per trade (medium risk)
        stop_loss_pct = 0.015  # 1.5% stop
        take_profit_pct = 0.04  # 4% take profit
        fee_bps = 5 / 10000.0  # 5 bps per trade
        slippage_bps = 2 / 10000.0  # 2 bps per trade
        txn_cost = fee_bps + slippage_bps
        vol_window = 20
        
        # Instantiate strategy objects if available
        ml_strategy = None
        alpha_strategy = None
        if strategy_name == "ml_ensemble" and create_ml_strategy is not None:
            try:
                ml_strategy = create_ml_strategy('ensemble')
            except Exception:
                ml_strategy = None
        if create_alpha_strategy is not None and strategy_name in ("mean_reversion", "momentum_breakout", "volatility_trading"):
            try:
                alpha_strategy = create_alpha_strategy(strategy_name)
            except Exception:
                alpha_strategy = None
        
        # Warmup periods for indicators/features
        warmup = 0
        if strategy_name == "ml_ensemble":
            warmup = 50  # extended for ML feature stability
        elif strategy_name == "mean_reversion":
            warmup = 60
        elif strategy_name in ("momentum_breakout", "volatility_trading"):
            warmup = 20  # reduced to activate strategies
        
        # Simple simulation logic
        for i in range(warmup, len(data)):  # Skip warmup for indicators
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            
            # Volatility-adjusted sizing factor (reduce size in high vol)
            recent_rets = current_data['close'].pct_change().tail(vol_window)
            vol = recent_rets.std() if len(recent_rets) > 0 else 0.0
            vol_adj = max(0.5, 1.0 - (vol * 10))  # between 0.5 and 1.0 typically
            
            # Manage stops/takes for open positions before new signals
            if position != 0 and entry_price > 0:
                if position > 0:
                    ret_since_entry = (current_price - entry_price) / entry_price
                    if ret_since_entry <= -stop_loss_pct or ret_since_entry >= take_profit_pct:
                        exit_price = current_price * (1 - txn_cost)  # selling
                        pnl = (exit_price - entry_price) * position
                        equity += pnl
                        trades.append({'type': 'exit', 'reason': 'stop/take', 'price': exit_price, 'pnl': pnl})
                        position = 0
                        entry_price = 0
                else:  # short
                    ret_since_entry = (entry_price - current_price) / entry_price
                    if ret_since_entry <= -stop_loss_pct or ret_since_entry >= take_profit_pct:
                        exit_price = current_price * (1 + txn_cost)  # buying to cover
                        pnl = (entry_price - exit_price) * abs(position)
                        equity += pnl
                        trades.append({'type': 'exit', 'reason': 'stop/take', 'price': exit_price, 'pnl': pnl})
                        position = 0
                        entry_price = 0
            
            # Signal generation based on strategy type
            if strategy_name == "enhanced_rsi":
                # RSI-based signals
                rsi_period = 14
                if i >= rsi_period:
                    price_changes = current_data['close'].diff().tail(rsi_period)
                    gains = price_changes.where(price_changes > 0, 0)
                    losses = -price_changes.where(price_changes < 0, 0)
                    avg_gain = gains.rolling(rsi_period).mean().iloc[-1]
                    avg_loss = losses.rolling(rsi_period).mean().iloc[-1]
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Ultra-conservative RSI: only trade extreme conditions with cooldown
                        last_trade_idx = getattr(self, f'{strategy_name}_last_trade', -100)
                        cooldown_period = 50  # Minimum 50 bars between trades
                        
                        if rsi < 15 and position == 0 and vol_adj > 0.8 and (i - last_trade_idx) > cooldown_period:
                            position_value = equity * risk_pct * 0.3  # Very small position
                            position = position_value / (current_price * (1 + txn_cost))
                            entry_price = current_price * (1 + txn_cost)
                            trades.append({'type': 'buy', 'price': entry_price, 'size': position})
                            setattr(self, f'{strategy_name}_last_trade', i)
                            
                        elif rsi > 85 and position == 0 and vol_adj > 0.8 and (i - last_trade_idx) > cooldown_period:
                            position_value = equity * risk_pct * 0.3  # Very small position
                            position = -(position_value / (current_price * (1 - txn_cost)))
                            entry_price = current_price * (1 - txn_cost)
                            trades.append({'type': 'short', 'price': entry_price, 'size': abs(position)})
                            setattr(self, f'{strategy_name}_last_trade', i)
                        
                        # Exit conditions: RSI crosses back to neutral zone
                        elif position > 0 and rsi > 55:  # Exit long when RSI > 55
                            exit_price = current_price * (1 - txn_cost)
                            pnl = (exit_price - entry_price) * position
                            equity += pnl
                            trades.append({'type': 'sell', 'price': exit_price, 'pnl': pnl})
                            position = 0
                            entry_price = 0
                            
                        elif position < 0 and rsi < 45:  # Exit short when RSI < 45
                            exit_price = current_price * (1 + txn_cost)
                            pnl = (entry_price - exit_price) * abs(position)
                            equity += pnl
                            trades.append({'type': 'cover', 'price': exit_price, 'pnl': pnl})
                            position = 0
                            entry_price = 0
            
            elif strategy_name == "ml_ensemble":
                # Prefer real ML strategy if available; otherwise fallback heuristic
                action = None
                sig_conf = 0.6
                ml_signal_type = "real"
                if ml_strategy is not None:
                    try:
                        ml_sig = ml_strategy.generate_signal(current_data)
                        action = ml_sig.action
                        sig_conf = getattr(ml_sig, 'confidence', 0.6)
                        ml_signal_type = "real"
                    except Exception:
                        action = None
                if action is None:
                    # Heuristic fallback
                    returns = current_data['close'].pct_change().tail(20)
                    volatility = returns.std()
                    momentum = returns.sum()
                    score = momentum * 10 - volatility * 5
                    if score > 0.5:
                        action = "BUY"
                        sig_conf = 0.65
                    elif score < -0.5:
                        action = "SELL"
                        sig_conf = 0.65
                    else:
                        action = "HOLD"
                        sig_conf = 0.5
                        ml_signal_type = "heuristic"
                # Execute action
                size_factor = max(0.5, min(1.0, sig_conf)) * vol_adj
                if action == "BUY" and position <= 0:
                    # Close short if any
                    if position < 0:
                        exit_price = current_price * (1 + txn_cost)  # buy to cover
                        pnl = (entry_price - exit_price) * abs(position)
                        equity += pnl
                        trades.append({'type': 'cover', 'price': exit_price, 'pnl': pnl})
                        position = 0
                    # Open long
                    units = (equity * risk_pct * size_factor) / (current_price * (1 + txn_cost))
                    entry_price = current_price * (1 + txn_cost)
                    position = units
                    trades.append({'type': 'buy', 'price': entry_price, 'size': units})
                elif action == "SELL" and position >= 0:
                    # Close long if any
                    if position > 0:
                        exit_price = current_price * (1 - txn_cost)  # sell
                        pnl = (exit_price - entry_price) * position
                        equity += pnl
                        trades.append({'type': 'sell', 'price': exit_price, 'pnl': pnl})
                        position = 0
                    # Open short
                    units = (equity * risk_pct * size_factor) / (current_price * (1 - txn_cost))
                    entry_price = current_price * (1 - txn_cost)
                    position = -units
                    trades.append({'type': 'short', 'price': entry_price, 'size': units})
                
                # Log trade decision
                trade_log.append({
                    'timestamp': data.index[i],
                    'price': current_price,
                    'strategy': strategy_name,
                    'action': action,
                    'confidence': sig_conf,
                    'signal_type': ml_signal_type
                })
            
            elif strategy_name in ("mean_reversion", "momentum_breakout", "volatility_trading"):
                # Use alpha strategies if available
                action = None
                sig_conf = 0.6
                alpha_signal_type = "real"
                if alpha_strategy is not None:
                    try:
                        alpha_sig = alpha_strategy.generate_signal(current_data)
                        action = alpha_sig.action
                        sig_conf = getattr(alpha_sig, 'confidence', 0.6)
                        alpha_signal_type = "real"
                    except Exception:
                        action = "HOLD"
                        sig_conf = 0.5
                        alpha_signal_type = "fallback"
                else:
                    # Fallback signal generation for dormant strategies
                    if strategy_name == "mean_reversion":
                        # Simple mean reversion: buy when price < 20-day MA, sell when price > 20-day MA
                        ma_period = 20
                        if i >= ma_period:
                            ma = current_data['close'].tail(ma_period).mean()
                            price_to_ma = current_price / ma
                            # Ultra-conservative mean reversion with cooldown
                            last_trade_idx = getattr(self, f'{strategy_name}_last_trade', -100)
                            cooldown_period = 100  # Minimum 100 bars between trades
                            
                            if price_to_ma < 0.88 and position == 0 and vol_adj > 0.8 and (i - last_trade_idx) > cooldown_period:
                                action = "BUY" 
                                sig_conf = 0.5
                                setattr(self, f'{strategy_name}_last_trade', i)
                            elif price_to_ma > 1.12 and position == 0 and vol_adj > 0.8 and (i - last_trade_idx) > cooldown_period:
                                action = "SELL"
                                sig_conf = 0.5
                                setattr(self, f'{strategy_name}_last_trade', i)
                            elif position > 0 and price_to_ma > 1.02:  # Exit long when 2% above MA
                                action = "SELL"
                                sig_conf = 0.9
                            elif position < 0 and price_to_ma < 0.98:  # Exit short when 2% below MA  
                                action = "BUY"
                                sig_conf = 0.9
                            else:
                                action = "HOLD"
                                sig_conf = 0.5
                        else:
                            action = "HOLD"
                            sig_conf = 0.5
                    elif strategy_name == "momentum_breakout":
                        # Simple momentum: buy on 5-day high, sell on 5-day low
                        lookback = 5
                        if i >= lookback:
                            recent_data = current_data.tail(lookback)
                            high_5d = recent_data['high'].max()
                            low_5d = recent_data['low'].min()
                            volume_ma = current_data['volume'].tail(20).mean()
                            current_vol = data.iloc[i]['volume']
                            vol_spike = current_vol > volume_ma * 1.2
                            
                            if current_price >= high_5d * 0.999 and vol_spike and position <= 0:
                                action = "BUY"
                                sig_conf = 0.75
                            elif current_price <= low_5d * 1.001 and vol_spike and position >= 0:
                                action = "SELL" 
                                sig_conf = 0.75
                            else:
                                action = "HOLD"
                                sig_conf = 0.5
                        else:
                            action = "HOLD"
                            sig_conf = 0.5
                    elif strategy_name == "volatility_trading":
                        # Simple volatility trading: sell when vol spikes, buy when vol calms
                        vol_period = 10
                        if i >= vol_period:
                            returns = current_data['close'].pct_change().tail(vol_period)
                            current_vol = returns.std()
                            avg_vol = current_data['close'].pct_change().tail(30).std()
                            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                            
                            if vol_ratio > 1.5 and position >= 0:  # High vol - sell
                                action = "SELL"
                                sig_conf = min(0.8, vol_ratio / 2.0)
                            elif vol_ratio < 0.7 and position <= 0:  # Low vol - buy
                                action = "BUY"
                                sig_conf = min(0.8, 1.0 - vol_ratio)
                            else:
                                action = "HOLD"
                                sig_conf = 0.5
                        else:
                            action = "HOLD"
                            sig_conf = 0.5
                    else:
                        action = "HOLD"
                        sig_conf = 0.5
                    alpha_signal_type = "fallback"
                # Execute action
                size_factor = max(0.5, min(1.0, sig_conf)) * vol_adj
                if action == "BUY" and position <= 0:
                    if position < 0:
                        exit_price = current_price * (1 + txn_cost)  # buy to cover
                        pnl = (entry_price - exit_price) * abs(position)
                        equity += pnl
                        trades.append({'type': 'cover', 'price': exit_price, 'pnl': pnl})
                        position = 0
                    units = (equity * risk_pct * size_factor) / (current_price * (1 + txn_cost))
                    entry_price = current_price * (1 + txn_cost)
                    position = units
                    trades.append({'type': 'buy', 'price': entry_price, 'size': units})
                elif action == "SELL" and position >= 0:
                    if position > 0:
                        exit_price = current_price * (1 - txn_cost)  # sell
                        pnl = (exit_price - entry_price) * position
                        equity += pnl
                        trades.append({'type': 'sell', 'price': exit_price, 'pnl': pnl})
                        position = 0
                    units = (equity * risk_pct * size_factor) / (current_price * (1 - txn_cost))
                    entry_price = current_price * (1 - txn_cost)
                    position = -units
                    trades.append({'type': 'short', 'price': entry_price, 'size': units})
                
                # Log trade decision
                trade_log.append({
                    'timestamp': data.index[i],
                    'price': current_price,
                    'strategy': strategy_name,
                    'action': action,
                    'confidence': sig_conf,
                    'signal_type': alpha_signal_type
                })
            
            # Update equity curve
            current_equity = equity
            if position != 0:
                current_equity += (current_price - entry_price) * position
            
            equity_curve.append(current_equity)
        
        # Close final position
        if position != 0:
            last_price = data.iloc[-1]['close']
            if position > 0:
                exit_price = last_price * (1 - txn_cost)
                final_pnl = (exit_price - entry_price) * position
                trades.append({'type': 'final_sell', 'price': exit_price, 'pnl': final_pnl})
            else:
                exit_price = last_price * (1 + txn_cost)
                final_pnl = (entry_price - exit_price) * abs(position)
                trades.append({'type': 'final_cover', 'price': exit_price, 'pnl': final_pnl})
            equity += final_pnl
        
        # Convert to pandas for analysis; left-pad to align with full data length
        pad_len = (len(data) + 1) - len(equity_curve)
        if pad_len > 0:
            equity_curve = [self.initial_capital] * pad_len + equity_curve
        equity_series = pd.Series(equity_curve)
        trades_df = pd.DataFrame(trades)
        
        # Pad trade log with empty entries for warmup period
        trade_log = [None] * warmup + trade_log
        
        # Calculate profit for each trade
        for trade in trades:
            if trade['type'] == 'long':
                trade['profit'] = (data['close'].iloc[-1] - trade['price']) * trade['size']
            elif trade['type'] == 'short':
                trade['profit'] = (trade['price'] - data['close'].iloc[-1]) * trade['size']
        
        return {
            'equity_curve': equity_series,
            'trades': trades_df,
            'trade_log': pd.DataFrame([x for x in trade_log if x is not None]),
            'metrics': self.calculate_performance_metrics(equity_series, trades_df)
        }
    
    async def run_comprehensive_backtest(self, days: int = 365) -> dict:
        """Run comprehensive backtest across multiple strategies."""
        print("Starting Comprehensive Backtesting Suite")
        print("=" * 60)
        
        # Generate market data
        print("Generating synthetic market data...")
        market_data = self.generate_synthetic_data(days)
        print(f"Generated {len(market_data):,} data points over {days} days")
        
        # Define strategies to test
        strategies_to_test = {
            "Enhanced RSI": "enhanced_rsi",
            "ML Ensemble": "ml_ensemble",
            "Mean Reversion": "mean_reversion",
            "Momentum Breakout": "momentum_breakout",
            "Volatility Trading": "volatility_trading"
        }
        
        # Run backtests for each strategy
        results = {}
        
        for strategy_name, strategy_type in strategies_to_test.items():
            print(f"\nTesting {strategy_name} Strategy...")
            
            try:
                result = self.simulate_strategy_performance(market_data, strategy_type)
                results[strategy_name] = result
                
                metrics = result['metrics']
                print(f"[SUCCESS] {strategy_name} Results:")
                print(f"   - Total Return: {metrics['total_return_pct']:+.1f}%")
                print(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"   - Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
                print(f"   - Win Rate: {metrics['win_rate_pct']:.1f}%")
                print(f"   - Total Trades: {metrics['total_trades']}")
                
            except Exception as e:
                print(f"[FAILED] {strategy_name} failed: {e}")
                continue
        
        # Combine strategies (equal weight portfolio)
        print(f"\nCreating Combined Portfolio...")
        
        if results:
            # Calculate combined equity curve with alignment/padding
            lengths = [len(r['equity_curve']) for r in results.values()]
            combined_length = max(lengths)
            combined_equity = pd.Series(0.0, index=range(combined_length))
            weight_per_strategy = 1.0 / len(results)
            
            for strategy_name, result in results.items():
                eq = result['equity_curve']
                if len(eq) < combined_length:
                    pad = pd.Series([self.initial_capital] * (combined_length - len(eq)))
                    eq = pd.concat([pad, eq], ignore_index=True)
                combined_equity += eq * weight_per_strategy
            
            # Calculate combined metrics
            combined_trades = pd.concat([r['trades'] for r in results.values()], ignore_index=True)
            combined_metrics = self.calculate_performance_metrics(combined_equity, combined_trades)
            
            results['Combined Portfolio'] = {
                'equity_curve': combined_equity,
                'trades': combined_trades,
                'metrics': combined_metrics
            }
            
            print(f"[SUCCESS] Combined Portfolio Results:")
            print(f"   - Total Return: {combined_metrics['total_return_pct']:+.1f}%")
            print(f"   - Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
            print(f"   - Max Drawdown: {combined_metrics['max_drawdown_pct']:.1f}%")
        
        return results
    
    def generate_report(self, results: dict, output_file: str = "backtest_report.json"):
        """Generate comprehensive backtest report."""
        report = {
            'backtest_summary': {
                'run_date': datetime.now(timezone.utc).isoformat(),
                'initial_capital': self.initial_capital,
                'strategies_tested': len(results),
                'best_strategy': None,
                'worst_strategy': None
            },
            'strategy_results': {},
            'rankings': {
                'by_return': [],
                'by_sharpe': [],
                'by_drawdown': []
            },
            'trade_logs': {}
        }
        
        # Process results
        for strategy_name, result in results.items():
            metrics = result['metrics']
            report['strategy_results'][strategy_name] = {
                **metrics,
                "trade_log": result.get('trade_log', pd.DataFrame()).to_dict(orient='records') if isinstance(result.get('trade_log'), pd.DataFrame) else result.get('trade_log', []),
                "trades": result['trades'].to_dict(orient='records'),
                "equity_curve": result['equity_curve'].tolist()
            }
            report['trade_logs'][strategy_name] = result.get('trade_log', pd.DataFrame()).to_dict(orient='records') if isinstance(result.get('trade_log'), pd.DataFrame) else result.get('trade_log', [])
        
        # Find best/worst strategies
        if report['strategy_results']:
            best_return = max(report['strategy_results'].items(), 
                            key=lambda x: x[1]['total_return_pct'])
            worst_return = min(report['strategy_results'].items(), 
                             key=lambda x: x[1]['total_return_pct'])
            
            report['backtest_summary']['best_strategy'] = {
                'name': best_return[0],
                'return': best_return[1]['total_return_pct']
            }
            report['backtest_summary']['worst_strategy'] = {
                'name': worst_return[0],
                'return': worst_return[1]['total_return_pct']
            }
            
            # Rankings
            report['rankings']['by_return'] = sorted(
                report['strategy_results'].items(),
                key=lambda x: x[1]['total_return_pct'],
                reverse=True
            )
            
            report['rankings']['by_sharpe'] = sorted(
                report['strategy_results'].items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )
            
            report['rankings']['by_drawdown'] = sorted(
                report['strategy_results'].items(),
                key=lambda x: x[1]['max_drawdown_pct']
            )
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

async def main():
    """Run comprehensive backtest."""
    backtester = ComprehensiveBacktester(initial_capital=100.0)
    
    # Run backtest
    results = await backtester.run_comprehensive_backtest(days=180)  # 6 months
    
    # Generate report
    print(f"\nGenerating Performance Report...")
    report = backtester.generate_report(results)
    
    print(f"\nComprehensive Backtest Complete!")
    print(f"Report saved to: backtest_report.json")
    print(f"\n[WINNER] Best Strategy: {report['backtest_summary']['best_strategy']['name']} "
          f"({report['backtest_summary']['best_strategy']['return']:+.1f}%)")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
