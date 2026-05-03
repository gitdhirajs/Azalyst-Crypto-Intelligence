"""paper trader v2"""
import json, os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from src.risk_engine import RiskManager
STOP_LOSS_PCT=0.10
SLIPPAGE=0.0005
MIN_TRADE_USD=50

def atomic_write_json(obj: Dict, filepath: str) -> None:
    temp=filepath+".tmp"
    with open(temp,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,default=str)
    os.replace(temp,filepath)
class Portfolio:
    def __init__(self, portfolio_file="portfolio.json", risk_manager: Optional[RiskManager]=None):
        self.file=portfolio_file; self.risk_manager=risk_manager or RiskManager(); self.open_positions=[]; self.closed_trades=[]; self.cash_usdt=10000.0; self.total_deposited=10000.0; self.trade_counter=0; self.daily_pnl_realised=0.0
    def save(self): atomic_write_json({"open_positions":self.open_positions,"closed_trades":self.closed_trades,"cash_usdt":self.cash_usdt,"total_deposited":self.total_deposited,"trade_counter":self.trade_counter,"daily_pnl_realised":self.daily_pnl_realised},self.file)
    def enter_position(self,symbol,entry_price,units,confidence,current_prices):
        cost=round(entry_price*(1+SLIPPAGE)*units,2)
        if cost<MIN_TRADE_USD or cost>self.cash_usdt:return None
        self.trade_counter+=1
        pos={"id":f"C{self.trade_counter:04d}","symbol":symbol,"entry_price":entry_price*(1+SLIPPAGE),"current_price":entry_price,"units":units,"invested":cost,"entry_date":datetime.now(timezone.utc).isoformat(),"confidence":confidence,"peak_price":entry_price,"trail_stop":round(entry_price*(1-STOP_LOSS_PCT),4)}
        self.cash_usdt-=cost; self.open_positions.append(pos); self.save(); return pos
    def update_prices(self, prices):
        for p in self.open_positions:
            if p['symbol'] in prices:p['current_price']=prices[p['symbol']]
        self.save()
    def check_exits(self): return []
