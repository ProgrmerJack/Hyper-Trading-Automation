#!/usr/bin/env python3
"""Monitor state file for synthetic PnL creation and track the source."""

import json
import time
import traceback
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class StateFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_simulated_pnl = None
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('state.json'):
            self.check_simulated_pnl(event.src_path)
    
    def check_simulated_pnl(self, file_path):
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            current_pnl = state.get('simulated_pnl')
            if current_pnl is not None:
                if self.last_simulated_pnl is None or current_pnl != self.last_simulated_pnl:
                    print(f"🚨 SYNTHETIC PnL DETECTED: ${current_pnl:.2f}")
                    print(f"Previous: {self.last_simulated_pnl}")
                    print(f"Change: {current_pnl - (self.last_simulated_pnl or 0):.2f}")
                    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Print stack trace to see what triggered this
                    print("Stack trace:")
                    traceback.print_stack()
                    print("-" * 50)
                    
                    self.last_simulated_pnl = current_pnl
                    
        except Exception as e:
            pass  # Ignore file read errors during concurrent access

def main():
    """Monitor state file for synthetic PnL creation."""
    state_path = Path("data/state.json")
    
    print("🔍 Monitoring state.json for synthetic PnL creation...")
    print("Press Ctrl+C to stop")
    
    # Initial check
    handler = StateFileHandler()
    if state_path.exists():
        handler.check_simulated_pnl(str(state_path))
    
    # Set up file watcher
    observer = Observer()
    observer.schedule(handler, str(state_path.parent), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Monitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    main()
