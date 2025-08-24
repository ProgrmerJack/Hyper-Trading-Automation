#!/usr/bin/env python3
"""Simple script to run backtesting from the organized modules."""

import sys
from pathlib import Path

def main():
    """Run backtesting based on command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python run_backtests.py [comprehensive|technical]")
        print("  comprehensive - Run comprehensive strategy backtesting")
        print("  technical     - Run technical indicator backtesting")
        return
    
    test_type = sys.argv[1].lower()
    
    try:
        if test_type == "comprehensive":
            from hypertrader.backtesting import run_comprehensive_backtest
            run_comprehensive_backtest()
        elif test_type == "technical":
            from hypertrader.backtesting import run_technical_backtest
            run_technical_backtest()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available options: comprehensive, technical")
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure hypertrader package is properly installed")
    except Exception as e:
        print(f"Error running backtest: {e}")

if __name__ == "__main__":
    main()