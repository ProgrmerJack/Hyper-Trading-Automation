#!/usr/bin/env python
"""Continuous bot runner with automatic restart and proper orchestration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import asyncio
import time
import traceback
import argparse
import json
import yaml
from typing import Optional

from hypertrader.orchestrator import TradingOrchestrator
from hypertrader.utils.logging import get_logger, log_json


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    # Default configuration
    return {
        "symbol": "BTC-USD",
        "account_balance": 100.0,
        "risk_percent": 5.0,
        "max_exposure": 3.0,
        "exchange": "binance",
        "signal_path": "data/signal.json",
        "state_path": "data/state.json",
        "live": False  # Set to True for live trading
    }


def run_with_restart(config: dict, max_restarts: int = 5, restart_delay: int = 30):
    """Run the bot with automatic restart on failure."""
    logger = get_logger()
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            log_json(logger, "bot_starting", 
                    restart_count=restart_count,
                    config=config)
            
            # Create orchestrator with continuous loop
            orchestrator = TradingOrchestrator(
                config=config,
                loop_interval=60.0,  # Run every 60 seconds if not using WebSocket
                max_cycles=None,  # Run indefinitely
                use_websocket=True  # Use WebSocket for real-time data if available
            )
            
            # This will run until interrupted
            orchestrator.start()
            
            # If we get here, the bot exited normally
            log_json(logger, "bot_stopped_normally")
            break
            
        except KeyboardInterrupt:
            log_json(logger, "bot_stopped_by_user")
            print("\nBot stopped by user. Exiting...")
            break
            
        except Exception as e:
            restart_count += 1
            error_msg = traceback.format_exc()
            log_json(logger, "bot_crashed", 
                    error=str(e),
                    traceback=error_msg,
                    restart_count=restart_count,
                    will_restart=restart_count < max_restarts)
            
            if restart_count < max_restarts:
                print(f"\nBot crashed: {e}")
                print(f"Restarting in {restart_delay} seconds... (Restart {restart_count}/{max_restarts})")
                time.sleep(restart_delay)
            else:
                print(f"\nBot crashed {max_restarts} times. Giving up.")
                print(f"Last error: {e}")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run HyperTrader bot continuously with automatic restart")
    parser.add_argument(
        "symbol", nargs="?", default="BTC-USD",
        help="Trading pair symbol (default: BTC-USD)")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--account-balance", type=float, default=100.0,
        help="Account balance (default: 100.0)")
    parser.add_argument(
        "--risk-percent", type=float, default=5.0,
        help="Risk percentage per trade (default: 5.0)")
    parser.add_argument(
        "--exchange", type=str, default="binance",
        help="Exchange to use (default: binance)")
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (default: False, paper trading)")
    parser.add_argument(
        "--max-restarts", type=int, default=5,
        help="Maximum number of automatic restarts (default: 5)")
    parser.add_argument(
        "--restart-delay", type=int, default=30,
        help="Delay in seconds between restarts (default: 30)")
    parser.add_argument(
        "--signal-path", type=str, default="data/signal.json",
        help="Path to signal JSON file (default: data/signal.json)")
    parser.add_argument(
        "--state-path", type=str, default="data/state.json",
        help="Path to state JSON file (default: data/state.json)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        # Extract nested config values if they exist
        if 'trading' in config:
            trading_config = config['trading']
            config.update({
                'symbol': trading_config.get('symbol', args.symbol),
                'account_balance': trading_config.get('account_balance', args.account_balance),
                'risk_percent': trading_config.get('risk_percent', args.risk_percent),
                'exchange': trading_config.get('exchange', args.exchange),
                'max_exposure': trading_config.get('max_exposure', 3.0),
                'live': trading_config.get('live', args.live)
            })
        # Set defaults if not present
        config.setdefault('signal_path', args.signal_path)
        config.setdefault('state_path', args.state_path)
    else:
        config = {
            "symbol": args.symbol,
            "account_balance": args.account_balance,
            "risk_percent": args.risk_percent,
            "exchange": args.exchange,
            "signal_path": args.signal_path,
            "state_path": args.state_path,
            "live": args.live
        }
    
    print("="*60)
    print("HyperTrader Continuous Bot Runner")
    print("="*60)
    print(f"Symbol: {config['symbol']}")
    print(f"Exchange: {config['exchange']}")
    print(f"Account Balance: ${config['account_balance']}")
    print(f"Risk per Trade: {config['risk_percent']}%")
    print(f"Mode: {'LIVE TRADING' if config['live'] else 'PAPER TRADING'}")
    print(f"Signal Path: {config['signal_path']}")
    print(f"State Path: {config['state_path']}")
    print(f"Max Restarts: {args.max_restarts}")
    print("="*60)
    
    if config['live']:
        print("\n⚠️  WARNING: LIVE TRADING MODE ENABLED ⚠️")
        print("Real money is at risk. Press Ctrl+C to cancel...")
        time.sleep(5)
    
    # Run with automatic restart
    run_with_restart(config, args.max_restarts, args.restart_delay)


if __name__ == "__main__":
    main()
