#!/usr/bin/env python3
"""
Component Audit Tool - Analyze which components are actually active vs claimed
This script audits the trading bot to determine which of the 118 components
are genuinely integrated and used in the trading pipeline.
"""

import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_components_yaml() -> List[str]:
    """Load component list from components.yaml."""
    components_path = Path(__file__).parent.parent / 'src' / 'hypertrader' / 'components.yaml'
    try:
        with open(components_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('components', [])
    except Exception as e:
        print(f"Error loading components.yaml: {e}")
        return []

def find_python_files() -> List[Path]:
    """Find all Python files in the hypertrader package."""
    src_dir = Path(__file__).parent.parent / 'src' / 'hypertrader'
    return list(src_dir.rglob('*.py'))

def extract_imports_and_calls(file_path: Path) -> Dict[str, Set[str]]:
    """Extract imports and function calls from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to find imports
        tree = ast.parse(content)
        imports = set()
        function_calls = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        function_calls.add(f"{node.func.value.id}.{node.func.attr}")
        
        return {
            'imports': imports,
            'function_calls': function_calls
        }
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {'imports': set(), 'function_calls': set()}

def audit_technical_indicators(files_data: Dict[Path, Dict[str, Set[str]]]) -> Dict[str, bool]:
    """Audit which technical indicators are actually used."""
    indicators = [
        'ema', 'sma', 'rsi', 'macd', 'atr', 'bollinger_bands', 'supertrend', 'vwap',
        'anchored_vwap', 'obv', 'twap', 'wavetrend', 'multi_rsi', 'roc', 'adx',
        'stochastic', 'ichimoku', 'parabolic_sar', 'keltner_channels', 'cci',
        'fibonacci_retracements', 'stochastic_oscillator', 'williams_r',
        'commodity_channel_index', 'volume_oscillator', 'chaikin_money_flow',
        'on_balance_volume', 'aroon_indicator', 'average_directional_index',
        'ultimate_oscillator', 'vortex_indicator'
    ]
    
    used_indicators = {}
    
    for indicator in indicators:
        used = False
        for file_path, data in files_data.items():
            # Check if indicator function is imported or called
            for import_name in data['imports']:
                if indicator in import_name or indicator.replace('_', '') in import_name:
                    used = True
                    break
            for call in data['function_calls']:
                if indicator in call or indicator.replace('_', '') in call:
                    used = True
                    break
            if used:
                break
        used_indicators[indicator] = used
    
    return used_indicators

def audit_strategies(files_data: Dict[Path, Dict[str, Set[str]]]) -> Dict[str, bool]:
    """Audit which trading strategies are actually used."""
    strategies = [
        'ma_cross', 'rsi_strategy', 'bollinger_strategy', 'macd_strategy',
        'ichimoku_strategy', 'psar_strategy', 'cci_strategy', 'keltner_strategy',
        'fibonacci_strategy', 'donchian_breakout', 'mean_reversion',
        'momentum_multi_tf', 'simple_ml', 'advanced_ml', 'market_maker',
        'statistical_arbitrage', 'triangular_arbitrage', 'event_trading',
        'rl_strategy', 'multi_oscillator_strategy', 'trend_strength_strategy',
        'volume_profile_strategy', 'breakout_confirmation_strategy'
    ]
    
    used_strategies = {}
    
    for strategy in strategies:
        used = False
        for file_path, data in files_data.items():
            # Check if strategy class or function is imported or called
            for import_name in data['imports']:
                if strategy in import_name:
                    used = True
                    break
            for call in data['function_calls']:
                if strategy in call:
                    used = True
                    break
            if used:
                break
        used_strategies[strategy] = used
    
    return used_strategies

def audit_ml_components(files_data: Dict[Path, Dict[str, Set[str]]]) -> Dict[str, bool]:
    """Audit ML and sentiment components."""
    ml_components = [
        'multimodal_sentiment', 'sentiment_momentum_strategy', 
        'crypto_specific_sentiment', 'social_media_filtering',
        'news_sentiment_ensemble', 'ai_momentum', 'volatility_clustering',
        'ai_var', 'drl_throttle', 'quantum_leverage', 'shap_explainability'
    ]
    
    used_components = {}
    
    for component in ml_components:
        used = False
        for file_path, data in files_data.items():
            for import_name in data['imports']:
                if component in import_name:
                    used = True
                    break
            for call in data['function_calls']:
                if component in call:
                    used = True
                    break
            if used:
                break
        used_components[component] = used
    
    return used_components

def audit_risk_management(files_data: Dict[Path, Dict[str, Set[str]]]) -> Dict[str, bool]:
    """Audit risk management components."""
    risk_components = [
        'position_sizing', 'leverage_management', 'stop_management',
        'drawdown_controls', 'kill_switch', 'fee_slippage_gating',
        'capital_compounding', 'exposure_cap', 'cost_benefit_analysis'
    ]
    
    used_components = {}
    
    for component in risk_components:
        used = False
        for file_path, data in files_data.items():
            for import_name in data['imports']:
                if component in import_name:
                    used = True
                    break
            for call in data['function_calls']:
                if component in call:
                    used = True
                    break
            if used:
                break
        used_components[component] = used
    
    return used_components

def main():
    """Main audit function."""
    print("HyperTrader Component Audit")
    print("=" * 50)
    
    # Load components from YAML
    all_components = load_components_yaml()
    print(f"Total components in components.yaml: {len(all_components)}")
    
    # Find all Python files
    python_files = find_python_files()
    print(f"Analyzing {len(python_files)} Python files...")
    
    # Extract imports and calls from all files
    files_data = {}
    for file_path in python_files:
        files_data[file_path] = extract_imports_and_calls(file_path)
    
    print("\nCOMPONENT AUDIT RESULTS")
    print("=" * 50)
    
    # Audit each category
    print("\nTECHNICAL INDICATORS:")
    indicators = audit_technical_indicators(files_data)
    active_indicators = sum(indicators.values())
    for name, used in sorted(indicators.items()):
        status = "ACTIVE" if used else "UNUSED"
        print(f"  {name}: {status}")
    print(f"Active Indicators: {active_indicators}/{len(indicators)}")
    
    print("\nTRADING STRATEGIES:")
    strategies = audit_strategies(files_data)
    active_strategies = sum(strategies.values())
    for name, used in sorted(strategies.items()):
        status = "ACTIVE" if used else "UNUSED"
        print(f"  {name}: {status}")
    print(f"Active Strategies: {active_strategies}/{len(strategies)}")
    
    print("\nML & SENTIMENT COMPONENTS:")
    ml_components = audit_ml_components(files_data)
    active_ml = sum(ml_components.values())
    for name, used in sorted(ml_components.items()):
        status = "ACTIVE" if used else "UNUSED"
        print(f"  {name}: {status}")
    print(f"Active ML Components: {active_ml}/{len(ml_components)}")
    
    print("\nRISK MANAGEMENT:")
    risk_components = audit_risk_management(files_data)
    active_risk = sum(risk_components.values())
    for name, used in sorted(risk_components.items()):
        status = "ACTIVE" if used else "UNUSED"
        print(f"  {name}: {status}")
    print(f"Active Risk Components: {active_risk}/{len(risk_components)}")
    
    # Summary
    total_audited = len(indicators) + len(strategies) + len(ml_components) + len(risk_components)
    total_active = active_indicators + active_strategies + active_ml + active_risk
    
    print("\nSUMMARY:")
    print("=" * 30)
    print(f"Components claimed in YAML: {len(all_components)}")
    print(f"Components audited: {total_audited}")
    print(f"Components actually active: {total_active}")
    print(f"Activity rate: {total_active/total_audited*100:.1f}%")
    
    if total_active < len(all_components) * 0.5:
        print("\nWARNING: Less than 50% of claimed components are actually active!")
        print("Consider updating the component claims to reflect reality.")
    
    # Find core bot.py usage
    bot_file = Path(__file__).parent.parent / 'src' / 'hypertrader' / 'bot.py'
    if bot_file in files_data:
        bot_data = files_data[bot_file]
        print(f"\nbot.py imports: {len(bot_data['imports'])} modules")
        print(f"bot.py function calls: {len(bot_data['function_calls'])} unique functions")

if __name__ == "__main__":
    main()
