//+------------------------------------------------------------------+
//|                                                HyperTrader EA    |
//|                        Example placeholder implementation        |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
#include <stdlib.mqh>
CTrade trade;

input string SignalFile = "signal.json";

struct Signal
  {
   string action;
   double volume;
   double stop_loss;
   double take_profit;
  };

void OnTick()
  {
   Signal signal;
   if(!ReadSignal(signal))
      return;
   if(signal.action=="BUY")
     {
      trade.Buy(signal.volume,_Symbol,0,signal.stop_loss,signal.take_profit);
     }
   else if(signal.action=="SELL")
     {
      trade.Sell(signal.volume,_Symbol,0,signal.stop_loss,signal.take_profit);
     }
  }

bool ReadSignal(Signal &sig)
  {
   int handle=FileOpen(SignalFile,FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle==INVALID_HANDLE)
      return(false);
   string json=FileReadString(handle);
   FileClose(handle);
   if(StringLen(json)==0)
      return(false);

   sig.action=JsonValueByName(json,"action");
   sig.volume=StrToDouble(JsonValueByName(json,"volume"));
   sig.stop_loss=StrToDouble(JsonValueByName(json,"stop_loss"));
   sig.take_profit=StrToDouble(JsonValueByName(json,"take_profit"));
   return(true);
  }
