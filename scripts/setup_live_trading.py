#!/usr/bin/env python3
"""
Live Trading Setup Script
Validates API credentials, sets up production environment, and performs safety checks.
"""

import os
import sys
import asyncio
import ccxt.async_support as ccxt
from pathlib import Path
from dotenv import load_dotenv
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging for live trading validation."""
    # Ensure logs directory exists before creating FileHandler
    Path('logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/setup_validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

async def validate_binance_connection(api_key: str, secret: str, testnet: bool = True) -> bool:
    """Validate Binance API connection and permissions."""
    logger = setup_logging()
    
    try:
        # Initialize exchange with testnet first
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'timeout': 10000,
        })

        # Ensure sandbox (testnet) mode is enabled when requested
        try:
            exchange.set_sandbox_mode(testnet)
        except Exception:
            pass
        
        # Test connection
        logger.info(f"Testing {'testnet' if testnet else 'live'} connection...")
        balance = await exchange.fetch_balance()
        
        # Check required permissions
        logger.info("Checking account permissions...")
        account_info = await exchange.fetch_balance()
        
        # Verify spot trading enabled
        if 'free' not in account_info:
            logger.error("Spot trading not enabled or insufficient permissions")
            return False
            
        # Test order placement (small test order on testnet)
        if testnet:
            try:
                symbol = 'BTC/USDT'
                ticker = await exchange.fetch_ticker(symbol)
                test_amount = 0.001  # Very small amount
                
                # Place and immediately cancel test order
                order = await exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='buy',
                    amount=test_amount,
                    price=ticker['bid'] * 0.95  # 5% below current price
                )
                
                # Cancel immediately
                await exchange.cancel_order(order['id'], symbol)
                logger.info("PASS - Order placement/cancellation test passed")
                
            except Exception as e:
                logger.error(f"Order test failed: {e}")
                return False
        
        # Check rate limits
        logger.info("PASS - API connection validated successfully")
        logger.info(f"Account type: {account_info.get('info', {}).get('accountType', 'Unknown')}")
        logger.info(f"Permissions: {account_info.get('info', {}).get('permissions', [])}")
        
        await exchange.close()
        return True
        
    except ccxt.AuthenticationError as e:
        logger.error(f"FAIL - Authentication failed: {e}")
        return False
    except ccxt.PermissionDenied as e:
        logger.error(f"FAIL - Permission denied: {e}")
        return False
    except Exception as e:
        logger.error(f"FAIL - Connection test failed: {e}")
        return False

def validate_environment() -> bool:
    """Validate production environment setup."""
    logger = setup_logging()
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET',
        'DISCORD_WEBHOOK_URL',
        'TELEGRAM_BOT_TOKEN',
        'EMAIL_USERNAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"FAIL - Missing required environment variables: {missing_vars}")
        logger.info("Please update your .env file with actual credentials")
        return False
    else:
        logger.info("PASS - Environment variables validated")
    return True

def setup_directories():
    """Create necessary directories for live trading."""
    directories = [
        'logs',
        'data/live',
        'data/backups',
        'data/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return True

def create_live_config():
    """Create live trading configuration file."""
    live_config = {
        "trading": {
            "symbol": "BTC/USDT",
            "account_balance": 100.0,
            "risk_percent": 3.5,
            "exchange": "binance",
            "max_exposure": 2.5,
            "live": True,
            "target_balance": 1000.0
        },
        "risk": {
            "max_daily_loss": 8.0,
            "max_position": 250.0,
            "fee_rate": 0.001,
            "slippage": 0.0002,
            "max_drawdown": 0.06,  # Stricter for live
            "stop_loss_pct": 0.025,
            "kill_switch_enabled": True
        },
        "monitoring": {
            "alerts_enabled": True,
            "dashboard_enabled": True,
            "backup_enabled": True,
            "performance_tracking": True
        }
    }
    
    with open('live_trading_config.yaml', 'w') as f:
        import yaml
        yaml.dump(live_config, f, default_flow_style=False)
    
    return True

async def run_safety_checks() -> bool:
    """Run comprehensive safety checks before live trading."""
    logger = setup_logging()
    
    logger.info("Running pre-live safety checks...")
    
    checks = []
    
    # 1. Environment validation
    checks.append(("Environment Variables", validate_environment()))
    
    # 2. Directory setup
    checks.append(("Directory Structure", setup_directories()))
    
    # 3. Configuration creation
    checks.append(("Live Config Creation", create_live_config()))
    
    # 4. API validation (testnet first)
    if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET'):
        testnet_result = await validate_binance_connection(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_SECRET'),
            testnet=True
        )
        checks.append(("Testnet Connection", testnet_result))
        
        # Only test live if testnet passes
        if testnet_result and input("Test live API connection? (y/N): ").lower() == 'y':
            live_result = await validate_binance_connection(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_SECRET'),
                testnet=False
            )
            checks.append(("Live Connection", live_result))
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("SAFETY CHECK RESULTS")
    logger.info("="*50)
    
    all_passed = True
    for check_name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        logger.info(f"{check_name:.<30} {status}")
        if not result:
            all_passed = False
    
    logger.info("="*50)
    
    if all_passed:
        logger.info("SUCCESS - All safety checks passed! Ready for live trading.")
        logger.info("To start live trading:")
        logger.info("1. Copy .env.production to .env")
        logger.info("2. Fill in your API credentials")
        logger.info("3. Run: python run_bot_continuous.py --config live_trading_config.yaml")
    else:
        logger.error("ERROR - Some safety checks failed. Fix issues before live trading.")
    
    return all_passed

if __name__ == "__main__":
    # Load environment
    load_dotenv('.env.production')
    
    print("HyperTrader Live Trading Setup")
    print("This script validates your setup for live trading")
    print("-" * 50)
    
    # Run safety checks
    asyncio.run(run_safety_checks())
