from __future__ import annotations
import time, csv, os, threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Metrics:
    latency_ms: float = 0.0
    traffic_in_msgs: int = 0
    traffic_out_msgs: int = 0
    errors: int = 0
    saturation_ratio: float = 0.0
    ws_reconnects: int = 0
    def inc_in(self,n:int=1): self.traffic_in_msgs += n
    def inc_out(self,n:int=1): self.traffic_out_msgs += n
    def inc_err(self,n:int=1): self.errors += n
    def inc_reconnect(self,n:int=1): self.ws_reconnects += n
    def set_latency(self,ms:float): self.latency_ms = ms
    def set_saturation(self,r:float): self.saturation_ratio = max(0.0, min(1.0, float(r)))

def csv_metrics_sink(path: str, metrics: Metrics, interval_sec: float = 2.0, stop_flag: Optional[threading.Event]=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["ts","latency_ms","in","out","errors","saturation","ws_reconnects"]
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    while stop_flag is None or not stop_flag.is_set():
        row = [int(time.time()*1000), metrics.latency_ms, metrics.traffic_in_msgs,
               metrics.traffic_out_msgs, metrics.errors, metrics.saturation_ratio,
               metrics.ws_reconnects]
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            pass
        time.sleep(interval_sec)
