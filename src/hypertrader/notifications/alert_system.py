"""
Advanced Alert and Notification System
Multi-channel notifications for trading events, risk management, and performance tracking
"""

import asyncio
import smtplib
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from pathlib import Path
import os
import logging

@dataclass
class AlertConfig:
    enabled: bool
    channels: List[str]
    triggers: List[str]
    cooldown_minutes: int = 5

@dataclass
class Alert:
    level: str  # INFO, WARNING, CRITICAL
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    channel: str = "all"

class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config_path: str = "alerts_config.json"):
        self.config_path = config_path
        self.alert_history = []
        self.last_alert_times = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize notification channels
        self.channels = {
            'discord': DiscordNotifier(),
            'telegram': TelegramNotifier(),
            'email': EmailNotifier(),
            'slack': SlackNotifier()
        }
    
    def _load_config(self) -> AlertConfig:
        """Load alert configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return AlertConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load alert config: {e}")
        
        # Default configuration
        return AlertConfig(
            enabled=True,
            channels=["discord", "email"],
            triggers=[
                "drawdown > 8%",
                "daily_loss > 10%",
                "position_size > 350",
                "new_high_equity",
                "system_error"
            ]
        )
    
    def should_send_alert(self, alert_key: str) -> bool:
        """Check if alert should be sent based on cooldown."""
        if not self.config.enabled:
            return False
        
        now = datetime.now(timezone.utc)
        last_time = self.last_alert_times.get(alert_key)
        
        if last_time is None:
            return True
        
        minutes_since = (now - last_time).total_seconds() / 60
        return minutes_since >= self.config.cooldown_minutes
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through configured channels."""
        alert_key = f"{alert.level}_{alert.title}"
        
        if not self.should_send_alert(alert_key):
            return False
        
        success = False
        
        # Send through each enabled channel
        for channel_name in self.config.channels:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    if await channel.send(alert):
                        success = True
                        self.logger.info(f"Alert sent via {channel_name}: {alert.title}")
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel_name}: {e}")
        
        if success:
            self.last_alert_times[alert_key] = datetime.now(timezone.utc)
            self.alert_history.append(alert)
        
        return success
    
    async def check_triggers(self, trading_state: Dict[str, Any]) -> None:
        """Check trading state against alert triggers."""
        alerts_to_send = []
        
        # Drawdown alert
        drawdown = trading_state.get('drawdown_pct', 0)
        if drawdown > 8:
            alerts_to_send.append(Alert(
                level="CRITICAL",
                title="High Drawdown Alert",
                message=f"Portfolio drawdown: {drawdown:.2f}% (threshold: 8%)",
                data={"drawdown": drawdown},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Daily loss alert
        daily_pnl = trading_state.get('daily_pnl', 0)
        if daily_pnl < -10:
            alerts_to_send.append(Alert(
                level="CRITICAL",
                title="Daily Loss Limit",
                message=f"Daily P&L: ${daily_pnl:.2f} (limit: -$10)",
                data={"daily_pnl": daily_pnl},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Position size alert
        position_value = trading_state.get('position_value', 0)
        if position_value > 350:
            alerts_to_send.append(Alert(
                level="WARNING",
                title="Large Position Size",
                message=f"Position size: ${position_value:.2f} (threshold: $350)",
                data={"position_value": position_value},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # New equity high
        current_equity = trading_state.get('current_equity', 0)
        peak_equity = trading_state.get('peak_equity', 0)
        if current_equity > peak_equity and current_equity > 150:  # Significant milestone
            alerts_to_send.append(Alert(
                level="INFO",
                title="New Equity High!",
                message=f"New equity high: ${current_equity:.2f} (previous: ${peak_equity:.2f})",
                data={"current_equity": current_equity, "previous_peak": peak_equity},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Latency alert
        latency = trading_state.get('avg_latency', 0)
        if latency > 2.0:
            alerts_to_send.append(Alert(
                level="WARNING",
                title="High Latency Warning",
                message=f"Average latency: {latency:.3f}s (threshold: 2.0s)",
                data={"latency": latency},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Send all triggered alerts
        for alert in alerts_to_send:
            await self.send_alert(alert)

class DiscordNotifier:
    """Discord webhook notifications."""
    
    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        if not self.webhook_url or "your_discord_webhook" in self.webhook_url:
            return False
        
        # Color based on alert level
        colors = {
            "INFO": 0x00ff00,      # Green
            "WARNING": 0xffaa00,   # Orange
            "CRITICAL": 0xff0000   # Red
        }
        
        embed = {
            "title": f"🚨 {alert.title}",
            "description": alert.message,
            "color": colors.get(alert.level, 0x0099ff),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {"name": key.replace('_', ' ').title(), "value": str(value), "inline": True}
                for key, value in alert.data.items()
            ],
            "footer": {"text": "HyperTrader Alert System"}
        }
        
        payload = {"embeds": [embed]}
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 204
        except Exception:
            return False

class TelegramNotifier:
    """Telegram bot notifications."""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    async def send(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        if not self.bot_token or not self.chat_id or "your_telegram" in str(self.bot_token):
            return False
        
        # Format message
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}
        
        message = f"{emoji.get(alert.level, '📢')} *{alert.title}*\n\n"
        message += f"{alert.message}\n\n"
        
        if alert.data:
            message += "*Details:*\n"
            for key, value in alert.data.items():
                message += f"• {key.replace('_', ' ').title()}: `{value}`\n"
        
        message += f"\n_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

class EmailNotifier:
    """Email notifications via SMTP."""
    
    def __init__(self):
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.username = os.getenv('EMAIL_USERNAME')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.to_email = os.getenv('EMAIL_TO')
    
    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not all([self.username, self.password, self.to_email]) or "your_email" in str(self.username):
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = self.to_email
            msg['Subject'] = f"HyperTrader Alert: {alert.title}"
            
            # HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.level == 'CRITICAL' else 'orange' if alert.level == 'WARNING' else 'green'}">
                    {alert.title}
                </h2>
                <p><strong>Level:</strong> {alert.level}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                
                {f'<h3>Details:</h3><ul>' + ''.join([f'<li><strong>{k.replace("_", " ").title()}:</strong> {v}</li>' for k, v in alert.data.items()]) + '</ul>' if alert.data else ''}
                
                <hr>
                <p><em>Sent by HyperTrader Alert System</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception:
            return False

class SlackNotifier:
    """Slack webhook notifications."""
    
    def __init__(self):
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url or "your_slack_webhook" in self.webhook_url:
            return False
        
        # Color based on alert level
        colors = {
            "INFO": "good",
            "WARNING": "warning", 
            "CRITICAL": "danger"
        }
        
        attachment = {
            "color": colors.get(alert.level, "good"),
            "title": alert.title,
            "text": alert.message,
            "timestamp": int(alert.timestamp.timestamp()),
            "fields": [
                {"title": key.replace('_', ' ').title(), "value": str(value), "short": True}
                for key, value in alert.data.items()
            ],
            "footer": "HyperTrader Alert System"
        }
        
        payload = {"attachments": [attachment]}
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

# Performance monitoring alerts
class PerformanceMonitor:
    """Monitor trading performance and trigger alerts."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.performance_history = []
    
    async def monitor_performance(self, trading_state: Dict[str, Any]) -> None:
        """Monitor performance metrics and send alerts."""
        current_equity = trading_state.get('current_equity', 0)
        starting_equity = trading_state.get('original_balance', 100)
        
        # Calculate performance metrics
        total_return = (current_equity - starting_equity) / starting_equity * 100
        trade_count = trading_state.get('trade_count', 0)
        
        # Store performance snapshot
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'equity': current_equity,
            'return_pct': total_return,
            'trades': trade_count
        })
        
        # Performance milestones
        milestones = [150, 200, 300, 500, 750, 1000]  # Dollar amounts
        
        for milestone in milestones:
            if current_equity >= milestone and not any(
                h['equity'] >= milestone for h in self.performance_history[:-1]
            ):
                await self.alert_manager.send_alert(Alert(
                    level="INFO",
                    title=f"Milestone Reached: ${milestone}",
                    message=f"Portfolio reached ${current_equity:.2f} (+{total_return:.1f}% from start)",
                    data={
                        'current_equity': current_equity,
                        'total_return_pct': total_return,
                        'trades_executed': trade_count
                    },
                    timestamp=datetime.now(timezone.utc)
                ))

# Factory function
def create_alert_system() -> AlertManager:
    """Create and configure alert system."""
    return AlertManager()
