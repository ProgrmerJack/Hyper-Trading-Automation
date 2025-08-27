#!/usr/bin/env python3
"""
P&L Forensics - Detect synthetic P&L and unrealistic trading results
Implements the forensic checklist to identify zero-loss demo issues
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_database_losing_trades():
    """Check if any losing trades exist in the database."""
    db_path = Path("data/state.db")
    if not db_path.exists():
        print("No database found - demo hasn't run yet")
        return False, 0, 0
    
    conn = sqlite3.connect(db_path)
    try:
        # Check for fills and calculate P&L
        query = """
        SELECT 
            f.order_id,
            o.side,
            f.qty,
            f.price,
            f.fee,
            (CASE WHEN o.side = 'SELL' THEN f.qty * f.price 
                  ELSE -f.qty * f.price END) as trade_pnl
        FROM fills f
        JOIN orders o ON f.order_id = o.id
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            print("No fills found - paper trading hasn't executed any trades")
            return False, 0, 0
            
        # Count losing vs winning trades
        losing_trades = len(df[df['trade_pnl'] < 0])
        winning_trades = len(df[df['trade_pnl'] > 0])
        total_trades = len(df)
        
        print(f"Trade Analysis:")
        print(f"  Total trades: {total_trades}")
        print(f"  Winning trades: {winning_trades}")
        print(f"  Losing trades: {losing_trades}")
        print(f"  Win rate: {winning_trades/total_trades*100:.1f}%")
        
        if losing_trades == 0 and total_trades > 5:
            print("RED FLAG: Zero losing trades with significant trade count!")
            return True, losing_trades, total_trades
            
        # Show P&L distribution
        print(f"\nP&L Distribution:")
        print(f"  Min P&L: ${df['trade_pnl'].min():.4f}")
        print(f"  Max P&L: ${df['trade_pnl'].max():.4f}")
        print(f"  Mean P&L: ${df['trade_pnl'].mean():.4f}")
        print(f"  Std P&L: ${df['trade_pnl'].std():.4f}")
        
        return losing_trades == 0, losing_trades, total_trades
        
    finally:
        conn.close()

def check_state_for_synthetic_pnl():
    """Check state file for synthetic P&L mechanisms."""
    state_path = Path("data/state.json")
    if not state_path.exists():
        print("No state file found")
        return []
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    red_flags = []
    
    # Check for simulated_pnl
    if 'simulated_pnl' in state:
        red_flags.append(f"Found simulated_pnl: {state['simulated_pnl']}")
    
    # Check for unrealistic equity progression
    equity_history = state.get('equity_history', [])
    if len(equity_history) > 10:
        if isinstance(equity_history[0], dict):
            equities = [e.get('equity', 0) for e in equity_history]
        else:
            equities = [e[1] if isinstance(e, (list, tuple)) else e for e in equity_history]
        
        # Check if equity only goes up
        differences = np.diff(equities)
        negative_moves = sum(1 for d in differences if d < 0)
        
        if negative_moves == 0 and len(differences) > 10:
            red_flags.append(f"Equity curve shows zero drawdowns over {len(differences)} periods")
    
    return red_flags

def analyze_code_for_synthetic_mechanisms():
    """Analyze code for synthetic P&L generation."""
    suspicious_patterns = []
    
    # Files to check
    files_to_check = [
        "src/hypertrader/bot.py",
        "src/hypertrader/enhanced_bot.py",
        "src/hypertrader/backtest_event.py",
        "src/hypertrader/utils/advanced_risk.py"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            continue
            
        with open(path, 'r') as f:
            content = f.read()
        
        # Look for synthetic P&L patterns
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for suspicious patterns
            if any(pattern in line_lower for pattern in [
                'expected_return', 'synthetic', 'abs(', 'max(', 'np.maximum'
            ]) and ('pnl' in line_lower or 'equity' in line_lower):
                suspicious_patterns.append(f"{path.name}:{i+1}: {line.strip()}")
    
    return suspicious_patterns

def check_for_missing_costs():
    """Check if realistic trading costs are implemented."""
    db_path = Path("data/state.db")
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    try:
        # Check if fees are being charged
        fee_query = "SELECT AVG(fee) as avg_fee, COUNT(*) as count FROM fills WHERE fee > 0"
        result = pd.read_sql_query(fee_query, conn)
        
        issues = []
        if result.iloc[0]['count'] == 0:
            issues.append("No trading fees found in fills - costs not implemented")
        elif result.iloc[0]['avg_fee'] < 0.0001:
            issues.append(f"Unrealistically low fees: {result.iloc[0]['avg_fee']:.6f}")
        
        return issues
        
    finally:
        conn.close()

def main():
    """Run complete P&L forensics analysis."""
    print("P&L Forensics Analysis")
    print("=" * 50)
    
    # 1. Check database for losing trades
    print("\n1. DATABASE ANALYSIS:")
    is_red_flag, losing_count, total_count = check_database_losing_trades()
    
    # 2. Check state file for synthetic mechanisms
    print("\n2. STATE FILE ANALYSIS:")
    state_red_flags = check_state_for_synthetic_pnl()
    if state_red_flags:
        for flag in state_red_flags:
            print(f"RED FLAG: {flag}")
    else:
        print("OK: No synthetic P&L found in state file")
    
    # 3. Analyze code for synthetic mechanisms  
    print("\n3. CODE ANALYSIS:")
    code_issues = analyze_code_for_synthetic_mechanisms()
    if code_issues:
        for issue in code_issues:
            print(f"WARNING: {issue}")
    else:
        print("OK: No obvious synthetic P&L patterns in code")
    
    # 4. Check for missing costs
    print("\n4. COST ANALYSIS:")
    cost_issues = check_for_missing_costs()
    if cost_issues:
        for issue in cost_issues:
            print(f"WARNING: {issue}")
    else:
        print("OK: Trading costs appear to be implemented")
    
    # 5. Overall assessment
    print("\n5. OVERALL ASSESSMENT:")
    total_red_flags = len(state_red_flags) + len(code_issues) + len(cost_issues)
    if is_red_flag:
        total_red_flags += 1
    
    if total_red_flags == 0:
        print("OK: No major red flags detected")
    elif total_red_flags <= 2:
        print(f"WARNING: {total_red_flags} potential issues detected - investigate further")
    else:
        print(f"RED FLAG: {total_red_flags} red flags detected - likely synthetic P&L!")
    
    # Recommendations
    print("\nRECOMMENDations:")
    if is_red_flag or total_red_flags > 2:
        print("- Implement realistic fill-based P&L calculation")
        print("- Add proper fees, slippage, and spread costs")
        print("- Eliminate any synthetic return calculations") 
        print("- Add execution-aware logging for transparency")
    else:
        print("- Run longer demo to accumulate more trade data")
        print("- Monitor for consistent win rates over time")

if __name__ == "__main__":
    main()
