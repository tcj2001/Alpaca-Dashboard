######################################################################################################
# Author: Thomson Mathews
######################################################################################################

import os
import alpaca_trade_api as tradeapi
import pandas as df
import numpy as np
import sys
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime, time
from pytz import timezone
import asyncio
import threading
import pytz  # $ pip install pytz
import talib
import logging

#GUI
from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtCharts import *
from PySide2.QtCore import *

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='portfolio.log',
                    filemode='w')

# get keys form environment variable
try:
    key_id = os.environ['KEY_ID']
    secret_key = os.environ['SECRET_KEY']
    base_url = os.environ['BASE_URL']
except Exception as e:
    raise Exception('Set API keys')

# alpaca api
api = tradeapi.REST(key_id, secret_key, base_url)

# region Model (Logic)

class AsyncEventLooper():
    def __init__(self):
        self._loop = asyncio.new_event_loop()

    #at class level
    def classasyncioLooper(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    # asyncio looper function for a thread
    def asyncioLooper(self,loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def add_periodic_task(self, async_func, interval):
        async def wrapper(_async_func, _interval):
            while True:
                await _async_func()
                await asyncio.sleep(_interval)
        self._loop.create_task(wrapper(async_func, interval))
        return

    def start(self):
        t = threading.Thread(target=self.asyncioLooper, args=(self._loop,))
        t.start()
        #self._loop.run_forever()
        return

#portfolio class handle alpaca data
class Portfolio(QObject):

    sendMessage=Signal(str)
    sendBuyingPower = Signal(float,float)
    openOrderLoaded = Signal(df.DataFrame)
    closedOrderLoaded = Signal(df.DataFrame)
    positionsLoaded = Signal(df.DataFrame)
    stockDataReady = Signal(str, df.DataFrame, df.DataFrame, df.DataFrame, df.DataFrame, str)

    def __init__(self,api):
        super().__init__()
        self.api=api
        self.stockHistory = {}
        self.stockOrdered = {}
        self.stockPosition = {}
        self.stockPartialPosition = {}
        self.sendPortFolio()

    def snapshot(self,symbol):
        ss = api.polygon.snapshot(symbol)
        return ss.ticker['day']['o'],ss.ticker['day']['h'],ss.ticker['day']['l'],ss.ticker['lastQuote']['p'],ss.ticker['lastTrade']['p'],ss.ticker['lastQuote']['P']

    def loadOpenPosition(self,lord_df,symbol=''):
        try:
            if symbol=='':
                existing_positions = self.api.list_positions()
                opos_df = df.DataFrame([position.__dict__['_raw'] for position in existing_positions])
                self.stockPosition=dict(zip(opos_df['symbol'], opos_df['qty'].astype('int32')))
                opos_df['profit'] = round(opos_df['unrealized_plpc'].astype(float) * 100, 2)
                self.opos_df=opos_df
                if self.opos_df is not None and lord_df is not None:
                    self.opos_df = self.opos_df.merge(lord_df, how='left', on='symbol')
                #self.positionsLoaded.emit(self.opos_df)
            if symbol!='':
                return self.opos_df[self.opos_df['symbol'] == symbol]
        except Exception as e:
            self.opos_df = None
            return None
        return self.opos_df

    def loadOpenOrders(self,symbol=''):
        try:
            if symbol=='':
                open_orders = self.api.list_orders(status='open', limit=500, direction='desc')
                oord_df = df.DataFrame([order.__dict__['_raw'] for order in open_orders])
                oord_df['submitted_at']=df.to_datetime(oord_df['submitted_at']).dt.tz_convert('US/Eastern')
                self.oord_df=oord_df
            if symbol!='':
                return self.oord_df[self.oord_df['symbol'] == symbol]
        except Exception as e:
            self.oord_df=None
            return None
        return self.oord_df

    def loadClosedOrders(self,symbol=''):
        try:
            if symbol=='':
                closed_orders= self.api.list_orders(status='closed', limit=500, direction='desc')
                cord_df=df.DataFrame([order.__dict__['_raw'] for order in closed_orders])
                cord_df['filled_on']=cord_df['filled_at'].str[:10]
                cord_df['filled_at']=df.to_datetime(cord_df['filled_at']).dt.tz_convert('US/Eastern')
                cord_df = cord_df[cord_df['status'] == 'filled']
                self.cord_df=cord_df
            if symbol!='':
                cord_df1 = self.cord_df[self.cord_df['symbol'] == symbol]
                #self.closedOrderLoaded.emit(cord_df1)
                return cord_df1
        except Exception as e:
            self.cord_df=None
            return None
        return cord_df

    def lastOrderAt(self,cord_df):
        try:
            lord_df=cord_df[['symbol', 'filled_at']]
            lord_df.set_index('symbol',inplace=True)
            lord_s = lord_df.groupby(['symbol'])['filled_at'].first()
            return lord_s
        except Exception as e:
            return None

    def allSymbols(self):
        self.symbols=[]
        try:
            if self.oord_df is not None:
                self.symbols = list(set(self.symbols) | set(self.oord_df['symbol']))
        except Exception as e:
            pass
        try:
            if self.opos_df is not None:
                self.symbols = list(set(self.symbols) | set(self.opos_df['symbol']))
        except Exception as e:
            pass
        try:
            if self.cord_df is not None:
                self.symbols = list(set(self.symbols) | set(self.cord_df['symbol']))
        except Exception as e:
            pass
        return self.symbols

    def getDailyHistory(self, symbol, multiplier, timeframe, fromdate, todate):
        #print('Fetching ...daily history for {}'.format(symbol))
        return self.api.polygon.historic_agg_v2(symbol, multiplier, timeframe, fromdate, todate).df

    def getMinutesHistory(self, symbol, minutes):
        #print('Fetching ...{} minutes history for {}'.format(minutes,symbol))
        return self.api.polygon.historic_agg(size="minute", symbol=symbol, limit=minutes).df

    def buy(self,symbol,qty,price,tag=None):
        try:
            do, dh, dl, bp, lp, ap = self.snapshot(symbol)
            if price > ap:
                o = api.submit_order(
                    symbol, qty=str(qty), side='buy',
                    type='stop', time_in_force='day',
                    stop_price=str(price),
                    extended_hours=False
                )
            else:
                o = api.submit_order(
                    symbol, qty=str(qty), side='buy',
                    type='limit', time_in_force='day',
                    limit_price=str(price),
                    extended_hours=True
                )
            self.sendMessage.emit('{} Buying {} Qty of {} @ {}'.format(tag,str(qty),symbol,str(price)))
        except Exception as e:
            del self.stockOrdered[symbol]
            self.sendMessage.emit('{} buy error: {}'.format(symbol,e.args[0]))

    def sell(self,symbol,qty,price,tag=None):
        try:
            do, dh, dl, bp, lp, ap = self.snapshot(symbol)
            if price < bp:
                o = api.submit_order(
                    symbol, qty=str(qty), side='sell',
                    type='stop', time_in_force='day',
                    stop_price=str(price),
                    extended_hours=False
                )
            else:
                o = api.submit_order(
                    symbol, qty=str(qty), side='sell',
                    type='limit', time_in_force='day',
                    limit_price=str(price),
                    extended_hours=True
                )
            self.sendMessage.emit('{} Selling {} Qty of {} @ {}'.format(tag,str(qty),symbol,str(price)))
        except Exception as e:
            del self.stockOrdered[symbol]
            self.sendMessage.emit('{} sell error: {}'.format(symbol,e.args[0]))

    def cancel(self,symbol,id):
        try:
            o = api.cancel_order(id)
            self.sendMessage.emit('Cancelling order {} of {}'.format(id,symbol))
        except Exception as e:
            self.sendMessage.emit(e.args[0])

    def cancelAll(self):
        try:
            o = api.cancel_all_orders()
            self.sendMessage.emit('Cancelling all order')
        except Exception as e:
            self.sendMessage.emit(e.args[0])

    def saveHistory(self, symbol, history):
        self.stockHistory[symbol]=history

    def saveTicks(self, symbol, data):
        ts = data.start
        ts -= timedelta(seconds=ts.second, microseconds=ts.microsecond)
        df1 = df.DataFrame([{"open": data.open, "high": data.high, "low": data.low, "close": data.close, "volume": data.volume}], [ts])

        if self.stockHistory.get(symbol) is  None:
            self.stockHistory[symbol]=df1
        else:
            try:
                current = self.stockHistory[symbol].loc[ts]
            except KeyError:
                current = None
            new_data = []
            if current is None:
                new_data = [
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume
                ]
            else:
                new_data = [
                    current.open,
                    data.high if data.high > current.high else current.high,
                    data.low if data.low < current.low else current.low,
                    data.close,
                    current.volume + data.volume
                ]
            self.stockHistory[symbol].loc[ts] = new_data

    def run(self):
        if self.threadingFunction=='sendStockData':
            self.sendStockData1(self.threadingFunctionParm1,self.threadingFunctionParm2)

    def sendStockData(self, symbol, timeFrame):
        if symbol=='':
            return
        if timeFrame == 'Minute':
            history = self.getMinutesHistory(symbol, 240)
        if timeFrame == 'Day':
            start = datetime.date(datetime.today() + relativedelta(months=-2))
            end = datetime.date(datetime.today() + timedelta(days=1))
            history = self.getDailyHistory(symbol, 1, 'day', start, end)

        oord_df1 = self.loadOpenOrders(symbol)
        cord_df1 = self.loadClosedOrders(symbol)
        opos_df1 = self.loadOpenPosition(symbol)
        self.stockDataReady.emit(symbol, history, opos_df1, oord_df1, cord_df1, timeFrame)

    def sendPortFolio(self):
        self.oord_df = self.loadOpenOrders()
        self.cord_df = self.loadClosedOrders()
        self.lord_df = self.lastOrderAt(self.cord_df)
        self.opos_df = self.loadOpenPosition(self.lord_df)

        self.positionsLoaded.emit(self.opos_df)
        self.openOrderLoaded.emit(self.oord_df)
        self.closedOrderLoaded.emit(self.cord_df)
        self.account=self.api.get_account()
        self.sendBuyingPower.emit(float(self.account.buying_power),float(self.account.last_equity)-float(self.account.equity))

#handles polygon data
class StreamingData(QObject):
    sendMessage0 = Signal(str)
    sendMessage=Signal(str)
    sendQuote=Signal(str,dict)
    sendTrade=Signal(str,dict)
    sendTick=Signal(str,dict)
    sendMTick = Signal(str, dict)
    sendAccountData = Signal(float,float)
    getMinuteHistory = Signal()
    runAlgo1 = Signal()
    runAlgo2 = Signal()

    def __init__(self,key_id,secret_key,base_url):
        super().__init__()

        self.key_id=key_id
        self.secret_key=secret_key
        self.base_url=base_url

        #streaming api
        self.conn = tradeapi.StreamConn(key_id=self.key_id, secret_key=self.secret_key, base_url=self.base_url)

        conn=self.conn
        @conn.on('status')
        async def on_status_messages(conn, channel, data):
            pass #print(data)

        @conn.on('account_updates')
        async def on_account_updates(conn, channel, data):
            self.sendAccountData.emit(float(data['buying_power']),float(data['last_equity'])-float(data['equity']))
            print(data)

        @conn.on('trade_updates')
        async def on_trade_updates(conn, channel, data):
            if data.event=='new':
                self.sendMessage.emit("new {} order placed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['qty'], data.order['submitted_at']))
            if data.event=='partially_filled':
                self.sendMessage.emit("{} order partially executed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['filled_qty'], data.order['filled_at']))
            if data.event=='fill':
                self.sendMessage.emit("{} order executed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['filled_qty'], data.order['filled_at']))
            #portfolio.sendPortFolio()

        @conn.on('A')
        async def on_tick_messages(conn, channel, data):
            self.sendMessage0.emit('{}:{}'.format(data.symbol,data.close))
            self.sendTick.emit(data.symbol,data)
            algo1.setData(data.symbol,data)
            #self.runAlgo1.emit()
            algo2.setData(data.symbol,data)
            #self.runAlgo2.emit()

            portfolio.saveTicks(data.symbol,data)
            algo1.runAlgo()
            algo2.runAlgo()



        @conn.on('AM')
        async def on_minute_messages(conn, channel, data):
            self.sendMessage0.emit('{}:{}'.format(data.symbol,data.close))
            self.sendMTick.emit(data.symbol, data)
            algo1.setData(data.symbol,data)
            #algo1.runAlgo()
            algo2.setData(data.symbol,data)
            #algo2.runAlgo()

            portfolio.saveTicks(data.symbol,data)
            algo1.runAlgo()
            algo2.runAlgo()

        @conn.on('T')
        async def on_trade_messages(conn, channel, data):
            self.sendTrade.emit(data.symbol, data)

        @conn.on('Q')
        async def on_quote_messages(conn, channel, data):
            self.sendQuote.emit(data.symbol,data)

        #conn.loop.run_until_complete(asyncio.gather(self.conn.subscribe(['account_updates','trade_updates'])))


    def getChannels(self,symbolList):
        channelList=[]
        if symbolList is not None:
            minHistory.setSymbolList(symbolList)
            self.getMinuteHistory.emit()
            for symbol in symbolList:
                symbol_channels = ['A.{}'.format(symbol), 'AM.{}'.format(symbol), 'Q.{}'.format(symbol), 'T.{}'.format(symbol)]
                channelList += symbol_channels
        return channelList

    async def subscribeChannels(self,channels):
        if channels is not None:
            await self.conn.subscribe(channels)

class MinuteHistory(QObject):
    def __init__(self):
        super().__init__()
    def setSymbolList(self,symbols):
        self.symbolList=symbols
    def getHistory(self):
        for symbol in self.symbolList:
            history = portfolio.getMinutesHistory(symbol, 100)
            portfolio.saveHistory(symbol, history)

class Study(QObject):
    def __init__(self, history):
        self.history=history

    def addEMA(self,length):
        try:
            self.history['EMA'+str(length)]=talib.EMA(self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addSMA(self,length):
        try:
            self.history['EMA'+str(length)] = talib.SMA(self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addVWAP(self):
        try:
            df = self.history
            df = df.assign(
                vwap=df.eval(
                    'wgtd = close * volume', inplace=False
                ).groupby(df.index.date).cumsum().eval('wgtd / volume')
            )
            self.history['VWAP'] = df['vwap']
        except Exception as e:
            return None

    def addMACD(self,fast,slow,signal):
        try:
            macd, signalline, macdhist = talib.MACD(
                self.history['close'],
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            self.history['MACD_F'+str(fast)+'_S'+str(slow)+'_L'+str(signal)+'_SIGNAL'] = macd-signalline
            self.history['MACD_F'+str(fast)+'_S'+str(slow)+'_L'+str(signal)+'_MACD'] = macd
            self.history['MACD_F'+str(fast)+'_S'+str(slow)+'_L'+str(signal)+'_SL'] = signalline
            self.history['MACD_F'+str(fast)+'_S'+str(slow)+'_L'+str(signal)+'_HIST'] = macdhist
        except Exception as e:
             return None

    def addRSI(self, length):
        try:
            self.history['RSI'+str(length)] =  talib.RSI(self.history['close'], timeperiod=length)
        except Exception as e:
             return None

    def addATR(self,length):
        try:
            self.history['ATR'+str(length)] =  talib.ATR(self.history['high'],self.history['low'],self.history['close'], timeperiod=length)
        except Exception as e:
             return None

    def addCCI(self, length):
        try:
            self.history['CCI'+str(length)] =  talib.CCI(self.history['high'],self.history['low'],self.history['close'], timeperiod=length)
        except Exception as e:
             return None

    def addBBANDS (self, length, devup, devdn, type):
        try:
            up,mid,low =  talib.BBANDS(self.history['close'], timeperiod=length, nbdevup=devup, nbdevdn=devdn, matype=type)
            bbp = (self.history['close']- low) / (up - low)
            self.history['BBANDS'+str(length)+'_bbp'] =  bbp
            self.history['BBANDS'+str(length)+'_up'] =  up
            self.history['BBANDS'+str(length)+'_low'] =  low
            self.history['BBANDS'+str(length)+'_mid'] =  mid
        except Exception as e:
             return None

    def bullishCandleStickPatterns(self,c1,c2,c3):
        self.bullishPattern=self.bullishCandleStickPatterns(self.history.iloc[-1],self.history.iloc[-2],self.history.iloc[-3])
        pattern=None
        #LOCH bullish
        if c1.low < c1.open < c1.close <= c1.high and \
                c1.high-c1.close < c1.open-c1.low and \
                c1.close-c1.open < c1.open-c1.low:
            pattern='hammer'
        if c1.low <= c1.open < c1.close < c1.high and \
                c1.high-c1.close > c1.open-c1.low and \
                c1.close - c1.open < c1.high - c1.close:
            pattern='inverseHammer'
        #LCOH bearish
        if c2.low < c2.close < c2.open < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c1.open < c2.close and \
                c1.close-c1.open > c2.open-c2.close:
            pattern='bullishEngulfing'
        if c2.low < c2.close < c2.open < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c1.open < c2.close and \
                c1.close > c2.close+(c2.open-c2.close)/2:
            pattern='piercingLine'
        if c3.low < c3.close < c3.open < c3.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                abs(c2.open-c2.close) < abs(c3.open-c3.close) and \
                abs(c2.open - c2.close) < abs(c1.open - c1.close) :
            pattern='morningStar'
        if  c3.low <= c3.open < c3.close < c3.high and \
                c2.low <= c2.open < c2.close < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c3.close <= c2.open and \
                c2.close <= c1.open:
            pattern='threeWhiteSoldiers'
        return pattern

    def getHistory(self):
        return self.history

#allows to select watchlist
class WatchListSelector(QObject):

    watchlistSelected = Signal(df.DataFrame)
    listOfWatchList = Signal(list)

    def __init__(self):
        super().__init__()

        #implementing three watchlist
        self.watchLists=[' ', 'Jumbo', 'Chinese', 'TopEmaCount', 'HighOfTheDay']
        self.wl1 = None
        self.wl2 = None
        self.wl3 = None
        self.wl4 = None

        #this watchlist is a static list of symbol, you change code to laod from a file
        self.wl1 = WatchList(['TSLA', 'FB', 'AAPL', 'GOOGL', 'NFLX'])
        channels = dataStream.getChannels(self.wl1.getSymbols())
        dataStream.conn.loop.run_until_complete(asyncio.gather(dataStream.conn.subscribe(channels)))

        # this watchlist is a static list of symbol, you change code to laod from a file
        self.wl2 = WatchList(['BABA', 'BIDU', 'WB'])
        channels = dataStream.getChannels(self.wl2.getSymbols())
        dataStream.conn.loop.run_until_complete(asyncio.gather(dataStream.conn.subscribe(channels)))

        async def scanner1():
            # print('scanner')
            self.wl3 = TopEmaCountScanner()
            if self.wl3 is not None:
                channels = dataStream.getChannels(self.wl3.getSymbols())
                await dataStream.conn.subscribe(channels)
                # dataStream.conn.loop.run_until_complete(asyncio.gather(dataStream.conn.subscribe(channels)))
        ael1 = AsyncEventLooper()
        ael1.add_periodic_task(scanner1, 900)
        ael1.start()

        async def scanner2():
            # print('scanner')
            self.wl4 = HighOfTheDayScanner()
            if self.wl4 is not None:
                channels = dataStream.getChannels(self.wl4.getSymbols())
                await dataStream.conn.subscribe(channels)
                # dataStream.conn.loop.run_until_complete(asyncio.gather(dataStream.conn.subscribe(channels)))
        ael2 = AsyncEventLooper()
        ael2.add_periodic_task(scanner2, 300)
        ael2.start()

    def selectWatchList(self, id):
        if id == 'Jumbo':
            if self.wl1 is not None:
                self.watchlistSelected.emit(self.wl1.wl_df)
        if id == 'Chinese':
            if self.wl2 is not None:
                self.watchlistSelected.emit(self.wl2.wl_df)
        if id == 'TopEmaCount':
            if self.wl3 is not None:
                self.watchlistSelected.emit(self.wl3.wl_df)
            else:
                self.watchlistSelected.emit(df.DataFrame())
        if id == 'HighOfTheDay':
            if self.wl4 is not None:
                self.watchlistSelected.emit(self.wl4.wl_df)
            else:
                self.watchlistSelected.emit(df.DataFrame())

    def sendWatchListNames(self):
        self.listOfWatchList.emit(self.watchLists)

# generic watchlist class
class WatchList(QObject):
    def __init__(self, symbols):
        super().__init__()
        self.symbols=None
        tickers = api.polygon.all_tickers()
        self.selectedTickers = [ticker for ticker in tickers if (ticker.ticker in symbols)]
        wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in self.selectedTickers]
        self.wl_df = df.DataFrame.from_records(wlist)
        if not self.wl_df.empty:
            self.symbols = self.wl_df[0].tolist()

    def getSymbols(self):
        return self.symbols

# scanner stocks at high of the day
class HighOfTheDayScanner(QObject):
    def __init__(self,min_share_price=10, max_share_price=100, min_last_dv=100000000,
                 tolerance = .99, volume=1000000):
        super().__init__()
        self.min_share_price = min_share_price
        self.max_share_price = max_share_price
        self.min_last_dv = min_last_dv
        self.tolerance=tolerance
        self.volume=volume

        self.account = api.get_account()
        self.assets = api.list_assets()
        self.symbols = [asset.symbol for asset in self.assets if asset.tradable]
        self.selectedSymbols=None

        #self.sendMessage.emit("Starting high of the day Scanner")
        tickers = api.polygon.all_tickers()
        self.selectedTickers = [ticker for ticker in tickers if (
                ticker.ticker in self.symbols and
                ticker.lastTrade['p'] >= self.min_share_price and
                ticker.lastTrade['p'] <= self.max_share_price and
                ticker.day['v'] * ticker.lastTrade['p'] > self.min_last_dv and
                ticker.day['c'] > ticker.prevDay['c'] and
                ticker.day['c']>=ticker.day['h']*self.tolerance and
                ticker.day['v']>=self.volume
        )]
        wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in self.selectedTickers]
        self.wl_df = df.DataFrame.from_records(wlist)
        if not self.wl_df.empty:
            self.selectedSymbols=sorted(self.wl_df[0].tolist())

    def getSymbols(self):
        return self.selectedSymbols

# stocks with top count of close above ema
class TopEmaCountScanner(QObject):
    def __init__(self,  minutes=100,   ema2=20,cnt2=10,atrlength=20,atr=1,min_share_price=10, max_share_price=100, min_last_dv=100000000,
                 tolerance = .99, volume=1000000 ):
        super().__init__()
        self.api = api
        self.buyingPower= float(self.api.get_account().buying_power)

        self.ema2 = ema2
        self.cnt2=cnt2
        self.atr=atr
        self.atrlength=atrlength

        self.min_share_price = min_share_price
        self.max_share_price = max_share_price
        self.min_last_dv = min_last_dv
        self.tolerance=tolerance
        self.volume=volume


        symbols=["AMD","WFC","SQ","ABBV","FB","AAPL","GS","C","KO","WDC","NKE","WMT","TGT","CSCO","COST","ORLY","LABD","V","MA","BABA","JD","WB","DIS","LK","TWLO","CSX","Z","BIDU","MCD","DVN","ACB","GILD","QQQ","SBUX","WBA","STX","AMZN","FAST","NFLX","CELG","AMAT","UPS","FDX","DELL","SNAP","TWTR","DWT","JBLU","AKS","CMG","DB","GLD","LABU","DGAZ","UGAZ","UAL","NUGT","WYNN","NTES","SIRI","AMGN","MAR","ROST","DLTR","LULU","AEM","UWT","H","FAS","LVS","JNPR","PAYC","CRON","DD","URI","MOS","CLDR","WYND","RH","EA","NOK","HTZ","WSM","SHOP","OKTA","IMMU","CRWD","JNUG","IBM","WORK","ADI","DHI","KB","TOL","PEP","BYND","CLX","MU","OSTK","ROKU","ULTA"]
        sdf = df.DataFrame(columns=['symbol', 'count'])
        sdf.set_index('symbol', inplace=True)

        for symbol in symbols:
            try:
                mHistory = portfolio.getMinutesHistory(symbol, minutes)
                study = Study(mHistory)
                study.addEMA(ema2)
                studyHistory = study.getHistory()
                if studyHistory['close'][-1] < self.buyingPower and \
                        studyHistory['close'][-1] > studyHistory['EMA' + str(ema2)][-1]:
                    count = studyHistory[studyHistory['close'] > studyHistory['EMA' + str(ema2)]].shape[0]
                    sdf.loc[symbol] = {'count': count}
                sdf.sort_values(by=['count'], inplace=True, ascending=False)
            except  Exception as e:
                print(e)
        self.symbols=sdf.head(cnt2).index.tolist()

        tickers = api.polygon.all_tickers()
        self.selectedTickers = [ticker for ticker in tickers if (
                ticker.ticker in self.symbols and
                ticker.lastTrade['p'] >= self.min_share_price and
                ticker.lastTrade['p'] <= self.max_share_price and
                ticker.day['v'] * ticker.lastTrade['p'] > self.min_last_dv and
                ticker.day['c'] > ticker.prevDay['c'] and
                ticker.day['c']>=ticker.day['h']*self.tolerance and
                ticker.day['v']>=self.volume
        )]
        wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in self.selectedTickers]
        self.wl_df = df.DataFrame.from_records(wlist)
        if not self.wl_df.empty:
            self.selectedSymbols=sorted(self.wl_df[0].tolist())

    def getSymbols(self):
        return self.selectedSymbols

#allows to select algo
class AlgoSelector(QObject):

    listOfAlgo = Signal(list)

    def __init__(self):
        super().__init__()
        self.algoLists={'Algo1':False,'Algo2':False}

    def sendAlgoNames(self):
        self.listOfAlgo.emit(list(self.algoLists.keys()))

    def selectAlgo(self,str,ischecked):
        self.algoLists[str]=ischecked
        pass

class Algos(QObject):
    def __init__(self):
        super().__init__()
    def setData(self,symbol,data):
        self.symbol=symbol
        self.data=data
    def runAlgo(self):
        pass

class Algo1(Algos):
    def __init__(self):
        super().__init__()
    def runAlgo(self):
        # return if algo is not checked
        if not algoSelector.algoLists['Algo1']:
            return
        # prevent reordering till last order is filled or rejected
        if portfolio.stockOrdered.get(self.symbol) is not None:
            return
        # not enough history
        if portfolio.stockHistory.get(self.symbol) is None:
            return
        # print('algo1 ',self.symbol)
        # write your algo here
        # this algo resamples the history at 5minute and generate buy or sell signal on cci(4) crosover of 100 and -100
        study = Study(portfolio.stockHistory[self.symbol].resample('5T').first().fillna(method='ffill'))
        studyHistory = study.getHistory()
        close = studyHistory['close'][-1]
        last3barmax = studyHistory['high'].tail(3).max()
        last3barmin = studyHistory['high'].tail(3).min()
        qty = 0
        if portfolio.stockPosition.get(self.symbol) is not None:
            qty = portfolio.stockPosition.get(self.symbol)
        if close > last3barmax:
            if qty <= 0:
                portfolio.stockOrdered[self.symbol] = True
                portfolio.buy(self.symbol, 1, close, 'Algo1')

        if close < last3barmin:
            if qty > 0:
                portfolio.stockOrdered[self.symbol] = True
                portfolio.sell(self.symbol, 1, close, 'Algo1')

class Algo2(Algos):
    def __init__(self):
        super().__init__()
    def runAlgo(self):
        #return if algo is not checked
        if not algoSelector.algoLists['Algo2']:
            return
        #prevent reordering till last order is filled or rejected
        if portfolio.stockOrdered.get(self.symbol) is not None:
            return
        # not enough history
        if portfolio.stockHistory.get(self.symbol) is None:
            return
        # write your algo here
        # this algo resamples the history at 5minute and generate buy or sell signal on cci(4) crosover of 100 and -100
        study = Study(portfolio.stockHistory[self.symbol].resample('5T').first().fillna(method='ffill'))
        study.addEMA(20)
        studyHistory = study.getHistory()
        close = studyHistory['close'][-1]
        ema = studyHistory['EMA20'][-1]
        qty=0
        if portfolio.stockPosition.get(self.symbol) is not None:
            qty = portfolio.stockPosition.get(self.symbol)
        if close > ema:
            if qty <= 0:
                portfolio.stockOrdered[self.symbol] = True
                portfolio.buy(self.symbol, 1, close, 'Algo2')
        if close < ema:
            if qty > 0:
                portfolio.stockOrdered[self.symbol] = True
                portfolio.sell(self.symbol, 1, close, 'Algo2')

# endregion

# region View (GUI)

#GUI main window
class MainWindow(QMainWindow):

    requestStockData = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1400,800)
        self.buyingPower=0
        self.setWindowTitle("Alpaca Portfolio!......... BuyingPower: {}".format(self.buyingPower))

        self.buySellDailog=BuySellDialog()

        mw=QWidget()
        mg=QGridLayout()
        mw.setLayout(mg)
        hs =QSplitter(Qt.Horizontal)
        hs.setStyleSheet("QSplitter::handle { background-color: lightGray; }")
        vs = QSplitter(Qt.Vertical)
        vs.setStyleSheet("QSplitter::handle { background-color: lightGray; }")

        mg.addWidget(vs, 0, 0, 1, 2)
        mg.addWidget(hs,1,0,1,2)
        self.statusline0=QLabel('')
        self.statusline=QLabel('')
        self.statusline.setWordWrap(True)
        mg.addWidget(self.statusline,2,0,1,1)
        mg.setColumnMinimumWidth(0,1200)
        mg.addWidget(self.statusline0,2,1,1,1)


        self.setCentralWidget(mw)

        lw1 = QWidget()
        lg1 = QGridLayout()
        lb1=QLabel('WatchLists')
        self.watchListcombo=WatchListCombo()
        self.watchLists = WatchLists(['Symbol', 'Bid', 'Last', 'Ask'])
        lg1.addWidget(lb1,0,0,1,1,Qt.AlignRight)
        lg1.addWidget(self.watchListcombo, 0, 1, 1, 1,Qt.AlignLeft)
        lg1.setColumnMinimumWidth(1,300)
        lb2=QLabel('Algo')
        self.algoCombo = AlgoCombo()
        lg1.addWidget(lb2,0,2,1,1,Qt.AlignRight)
        lg1.addWidget(self.algoCombo,0,3,1,1,Qt.AlignLeft)
        lg1.addWidget(self.watchLists, 1, 0, 1, 4)
        lg1.setColumnMinimumWidth(3,300)
        lw1.setLayout(lg1)
        vs.addWidget(lw1)

        lw2 = QWidget()
        lg2 = QGridLayout()
        lb2=QLabel('Positions')
        self.openPositions = Positions(['Symbol','Qty','last','AveragePrice','Profit','FilledAt'])
        lg2.addWidget(lb2,0,0,1,1)
        lg2.addWidget(self.openPositions,1,0,1,1)
        lw2.setLayout(lg2)
        vs.addWidget(lw2)

        lw4 = QWidget()
        lg4 = QGridLayout()
        lb4=QLabel('Open Orders')
        self.openOrders = OpenOrder(['Symbol','Type','Qty','LimitPrice','StopPrice','SubmittedAt','Id'])
        lg4.addWidget(lb4,0,0,1,1)
        lg4.addWidget(self.openOrders,1,0,1,1)
        lw4.setLayout(lg4)
        vs.addWidget(lw4)

        lw3 = QWidget()
        lg3 = QGridLayout()
        lb3=QLabel('Closed Orders')
        self.closedOrders = ClosedOrder(['Symbol','Type','Qty','Price','FilledAt'])
        lg3.addWidget(lb3,2,0,1,1)
        lg3.addWidget(self.closedOrders,3,0,1,1)
        lw3.setLayout(lg3)
        vs.addWidget(lw3)


        hs.addWidget(vs)

        rw=QWidget()
        rg = QGridLayout()
        lb51=QLabel('Symbol')
        rg.addWidget(lb51,0,0,1,1)
        self.ql51 = QLineEdit()
        rg.addWidget(self.ql51,0,1,1,1)
        b1=QPushButton("Show")
        rg.addWidget(b1,0,2,1,1)
        b1.clicked.connect(self.showsym)
        lb6=QLabel('TimeFrame')
        self.timeFrame=TimeFrame()
        rg.addWidget(lb6,0,3,1,1,Qt.AlignRight)
        rg.addWidget(self.timeFrame,0,4,1,1,Qt.AlignLeft)
        rg.setColumnMinimumWidth(4,100)
        self.chartview = ChartView()
        rg.addWidget(self.chartview,1,0,1,5)
        rw.setLayout(rg)
        hs.addWidget(rw)

    def showsym(self):
        self.requestStockData.emit(self.ql51.text().upper(), self.timeFrame.currentText())
        self.timeFrame.symbol = self.ql51.text().upper()
        self.ql51.setText('')

    def displayAccountData(self,bp,pf):
        self.buyingPower=bp
        self.todaysProfit=pf
        self.setWindowTitle("Alpaca Portfolio!.....  BuyingPower: {} .....  Todays Profit: {}".format(self.buyingPower,self.todaysProfit))

    def statusMessage0(self,msg):
        self.statusline0.setText(msg)
    def statusMessage(self,msg):
        print(msg)
        self.statusline.setText(msg)

#watch list combo
class WatchListCombo(QComboBox):

    selectWatchList = Signal(str)

    def __init__(self):
        super().__init__()
        self.currentIndexChanged.connect(self.selectionChanged)
        self.symbol=None

    def loadData(self,list):
        self.addItems(list)

    def selectionChanged(self):
        self.selectWatchList.emit(self.currentText())
        pass

#watchlist table
class WatchLists(QTableView):

    requestStockData = Signal(str, str)

    def __init__(self,columns):
        super().__init__()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(columns)
        self.setAlternatingRowColors(True)
        self.setAutoScroll(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setModel(self.model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.setSortingEnabled(True)
        self.qSortFilterProxyModel = QSortFilterProxyModel()
        self.qSortFilterProxyModel.setSourceModel(self.model)
        self.qSortFilterProxyModel.setFilterKeyColumn(0)
        self.qSortFilterProxyModel.setFilterCaseSensitivity(Qt.CaseSensitive)

        self.selectionModel().selectionChanged.connect(self.handleSelectionChanged)

    def updateTick(self,symbol,data):
        self.setOneValue(symbol,2,str(data.close))

    def setOneValue(self,key,setColumn1,value1):
        self.qSortFilterProxyModel.setFilterFixedString(key);
        i=0
        found=False
        while i < self.qSortFilterProxyModel.rowCount():
            if self.qSortFilterProxyModel.index(i,0).data()==key:
                found=True
                break
            i+=1
        if found:
            index = self.qSortFilterProxyModel.mapToSource(self.qSortFilterProxyModel.index(i, 0));
            if index.isValid():
               self.model.setData(index.sibling(index.row(), setColumn1),value1)

    def updateQuote(self,symbol,data):
        self.setTwoValue(symbol,1,str(data.bidprice),3,str(data.askprice))

    def setTwoValue(self,key,setColumn1,value1,setColumn2,value2):
        self.qSortFilterProxyModel.setFilterFixedString(key);
        i=0
        found=False
        while i < self.qSortFilterProxyModel.rowCount():
            if self.qSortFilterProxyModel.index(i,0).data()==key:
                found=True
                break
            i+=1
        if found:
            index = self.qSortFilterProxyModel.mapToSource(self.qSortFilterProxyModel.index(i, 0));
            if index.isValid():
               self.model.setData(index.sibling(index.row(), setColumn1),value1)
               self.model.setData(index.sibling(index.row(), setColumn2),value2)


    def showContextMenu(self, pos):
        indexes = self.selectedIndexes()
        if len(indexes) == 0:
            return
        self.col = self.indexAt(pos).column()
        if self.col !=1 and self.col !=3:
            return
        menu = QMenu(self)
        if len(indexes) > 0:
            self.selectedSymbol = indexes[0].data()
            self.qty = 1
            self.close = indexes[1].data() if self.col==1 else indexes[3].data()
            item_add_act1 = QAction("{} {}".format('Buy' if self.col==3 else 'Sell', self.selectedSymbol), self)
            item_add_act1.triggered.connect(self.add_cb1)
            menu.addAction(item_add_act1)
            menu.popup(QCursor.pos())

    def add_cb1(self):
        if self.col==1:
            self.sell(self.selectedSymbol, int(self.qty),float(self.close))
        if self.col==3:
            self.buy(self.selectedSymbol, abs(int(self.qty)),float(self.close))

    def sell(self,symbol, qty, price):
        window.buySellDailog.sell(symbol,1,price)
        window.buySellDailog.exec_()

    def buy(self, symbol, qty , price):
        window.buySellDailog.buy(symbol,1,price)
        window.buySellDailog.exec_()

    def addRow(self,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        self.model.appendRow(tuple(items))
        return items

    def loadData(self,wl_df):
        self.model.removeRows(0,self.model.rowCount())
        for row in wl_df.iterrows():
            self.addRow([str(row[1][0]),str(row[1][1]),str(row[1][2]),str(row[1][3])])

        self.resizeRowsToContents()
        #self.resizeColumnsToContents()

    def handleSelectionChanged(self, selected, deselected):
        if selected.indexes().__len__()!=0:
            symbol=self.selectionModel().selectedRows()[0].data()
            window.timeFrame.symbol=symbol
            self.requestStockData.emit(symbol, window.timeFrame.currentText())

#watch list combo
class AlgoCombo(QComboBox):

    selectAlgo = Signal(str,bool)

    def __init__(self):
        super().__init__()
        self.qsim = QStandardItemModel()
        self.setModel(self.qsim)
        self.model().itemChanged.connect(self.selectionChanged)
        self.symbol=None

    def loadData(self,list):
        for algoname in list:
            qsi = QStandardItem(algoname)
            qsi.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            qsi.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.qsim.appendRow(qsi)


    def selectionChanged(self,qsi):
        self.selectAlgo.emit(qsi.text(),qsi.checkState().name==b'Checked')
        pass

#timeframe combo
class TimeFrame(QComboBox):

    requestStockData = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.addItems(['Minute','Day'])
        self.currentIndexChanged.connect(self.selectionChanged)
        self.symbol=None


    def selectionChanged(self):
        self.requestStockData.emit(self.symbol, self.currentText())
        pass

#positions table
class Positions(QTableView):

    requestStockData = Signal(str, str)

    def __init__(self,columns):
        super().__init__()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(columns)
        self.setAlternatingRowColors(True)
        self.setAutoScroll(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setModel(self.model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.selectionModel().selectionChanged.connect(self.handleSelectionChanged)
        self.setSortingEnabled(True)
        self.qSortFilterProxyModel = QSortFilterProxyModel()
        self.qSortFilterProxyModel.setSourceModel(self.model)
        self.qSortFilterProxyModel.setFilterKeyColumn(0)
        self.qSortFilterProxyModel.setFilterCaseSensitivity(Qt.CaseSensitive)
        self.setColumnWidth(5, 200)

    def addRow(self,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        self.model.appendRow(items)
        return items

    def showContextMenu(self, pos):
        indexes = self.selectedIndexes()
        if len(indexes) == 0:
            return
        idx = self.indexAt(pos)
        menu = QMenu(self)
        if len(indexes) > 0:
            self.selectedSymbol = indexes[0].data()
            self.qty = indexes[1].data()
            self.close = indexes[2].data()
            item_add_act1 = QAction("Close {}".format(self.selectedSymbol), self)
            item_add_act1.triggered.connect(self.add_cb1)
            menu.addAction(item_add_act1)
            menu.popup(QCursor.pos())

    def add_cb1(self):
        if int(self.qty)>0:
            self.sell(self.selectedSymbol, int(self.qty),float(self.close))
        if int(self.qty)<0:
            self.buy(self.selectedSymbol, abs(int(self.qty)),float(self.close))

    def sell(self,symbol,qty,price):
        window.buySellDailog.sell(symbol,qty,price)
        window.buySellDailog.exec_()

    def buy(self, symbol, qty, price):
        window.buySellDailog.buy(symbol,qty,price)
        window.buySellDailog.exec_()

    def loadData(self,opos_df):
        self.model.removeRows(0,self.model.rowCount())
        if opos_df  is not None:
            for row in opos_df.iterrows():
                filledat=''
                try:
                    filledat=row[1]['filled_at'].strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    pass
                p = self.addRow([row[1]['symbol'], str(int(row[1]['qty'])).rjust(5), str(row[1]['current_price']), str(round(float(row[1]['avg_entry_price']),2)),'{:.2f}'.format(row[1]['profit']),filledat])
            self.resizeRowsToContents()
            #self.resizeColumnsToContents()

    def handleSelectionChanged(self, selected, deselected):
        if selected.indexes().__len__()!=0:
            symbol=self.selectionModel().selectedRows()[0].data()
            window.timeFrame.symbol=symbol
            self.requestStockData.emit(symbol, window.timeFrame.currentText())

    def updateTick(self,symbol,data):
        self.setOneValue(symbol,str(data.close))

    def setOneValue(self,key,value1):
        self.qSortFilterProxyModel.setFilterFixedString(key);
        i=0
        found=False
        while i < self.qSortFilterProxyModel.rowCount():
            if self.qSortFilterProxyModel.index(i,0).data()==key:
                found=True
                break
            i+=1
        if found:
            index = self.qSortFilterProxyModel.mapToSource(self.qSortFilterProxyModel.index(i, 0));
            ap = index.sibling(index.row(), 3).data()
            qty = index.sibling(index.row(), 1).data()
            profit = round((float(value1)*int(qty)-float(ap)*float(qty))/abs(int(qty) * float(ap))*100,2)
            if index.isValid():
               self.model.setData(index.sibling(index.row(), 2),str(value1))
               self.model.setData(index.sibling(index.row(), 4),str(profit))

#open order table
class OpenOrder(QTableView):

    requestCancel = Signal(str,str)
    requestCancelAll = Signal()
    requestStockData = Signal(str, str)

    def __init__(self,columns):
        super().__init__()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(columns)
        self.setColumnWidth(4,200)
        self.setAlternatingRowColors(True)
        self.setAutoScroll(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setModel(self.model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.selectionModel().selectionChanged.connect(self.handleSelectionChanged)
        self.selectedSymbol=None
        self.id=None
        self.setSortingEnabled(True)

    def addRow(self,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        self.model.appendRow(tuple(items))
        return items

    def loadData(self,oord_df):
        #print('load open order called.........................................................')
        self.model.removeRows(0,self.model.rowCount())
        if oord_df is not None:
            for row in oord_df.iterrows():
                p = self.addRow([row[1]['symbol'],row[1]['side'], str(row[1]['qty']).rjust(5), str(row[1]['limit_price']), str(row[1]['stop_price']), row[1]['submitted_at'].strftime('%Y-%m-%d %H:%M:%S'),row[1]['id']])
            self.resizeRowsToContents()
            self.resizeColumnsToContents()

    def handleSelectionChanged(self, selected, deselected):
        if selected.indexes().__len__()!=0:
            self.selectedSymbol=self.selectionModel().selectedRows()[0].data()
            window.timeFrame.symbol=self.selectedSymbol
            self.requestStockData.emit(self.selectedSymbol,window.timeFrame.currentText())

    def showContextMenu(self, pos):
        indexes = self.selectedIndexes()
        if len(indexes) == 0:
            return
        idx=self.indexAt(pos)
        menu = QMenu(self)
        if len(indexes) > 0:
            self.selectedSymbol = indexes[0].data()
            self.id = indexes[6].data()
            item_add_act1 = QAction("Cancel {} order".format(self.selectedSymbol), self)
            item_add_act1.triggered.connect(self.add_cb1)
            menu.addAction(item_add_act1)
            item_add_act2 = QAction("Cancel all orders", self)
            item_add_act2.triggered.connect(self.add_cb2)
            menu.addAction(item_add_act2)
            menu.popup(QCursor.pos())

    def add_cb1(self):
        self.requestCancel.emit(self.selectedSymbol,self.id)

    def add_cb2(self):
        self.requestCancelAll.emit()

#closed order table
class ClosedOrder(QTableView):
    requestStockData = Signal(str, str)
    def __init__(self,columns):
        super().__init__()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(columns)
        self.setColumnWidth(4,200)
        self.setAlternatingRowColors(True)
        self.setAutoScroll(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setModel(self.model)
        self.selectionModel().selectionChanged.connect(self.handleSelectionChanged)
        self.setSortingEnabled(True)

    def addRow(self,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        self.model.appendRow(tuple(items))
        return items

    def loadData(self,cord_df):
        self.model.removeRows(0,self.model.rowCount())
        if cord_df is not None:
            for row in cord_df.iterrows():
                p = self.addRow([row[1]['symbol'],row[1]['side'], str(row[1]['qty']).rjust(5), str(row[1]['filled_avg_price']), row[1]['filled_at'].strftime('%Y-%m-%d -%H:%M:%S')])
            self.resizeRowsToContents()
            self.resizeColumnsToContents()

    def handleSelectionChanged(self, selected, deselected):
        if selected.indexes().__len__()!=0:
            symbol=self.selectionModel().selectedRows()[0].data()
            window.timeFrame.symbol=symbol
            self.requestStockData.emit(symbol, window.timeFrame.currentText())

#simple buy sell dialog
class BuySellDialog(QDialog):

    buyRequest = Signal(str,int,float,str)
    sellRequest = Signal(str,int,float,str)

    def __init__(self):
        super().__init__()
        self.resize(300,100)
        self.grid = QGridLayout()
        self.ql0 = QLabel('Qty:')
        self.ql1 = QLineEdit()
        self.ql2 = QLabel('price')
        self.ql3 = QLineEdit()
        self.b1=QPushButton("Cancel")
        self.b2=QPushButton("Send")
        self.grid.addWidget(self.ql0)
        self.grid.addWidget(self.ql1)
        self.grid.addWidget(self.ql2)
        self.grid.addWidget(self.ql3)
        self.grid.addWidget(self.b1)
        self.grid.addWidget(self.b2)
        self.setLayout(self.grid)
        self.b1.clicked.connect(self.cancel)
        self.b2.clicked.connect(self.send)

    def sell(self,symbol,qty,price):
        self.ql1.setText(str(qty))
        self.ql3.setText(str(price))
        #self.b2.clicked.connect(self.sell)
        self.symbol=symbol
        self.price=price
        self.setWindowTitle('Sell {}'.format(symbol))
        self.type='Sell'

    def buy(self,symbol,qty,price):
        self.ql1.setText(str(qty))
        self.ql3.setText(str(price))
        #self.b2.clicked.connect(self.buy)
        self.symbol=symbol
        self.price=price
        self.setWindowTitle('Buy {}'.format(symbol))
        self.type='Buy'

    def send(self):
        self.close()
        if self.type=='Sell':
            self.sellRequest.emit(self.symbol,int(self.ql1.text()),float(self.ql3.text()),'Dashboard')
        if self.type=='Buy':
            self.buyRequest.emit(self.symbol,int(self.ql1.text()),float(self.ql3.text()),'Dashboard')

    def cancel(self):
        self.close()

#Chartview
class ChartView(QtCharts.QChartView):
    def __init__(self):
        super().__init__()
        self.setRubberBand(self.HorizontalRubberBand)
        #self.setRubberBand(self.RectangleRubberBand)
        self.setMouseTracking(True)
        self.x=0
        self.y=0
        self.xlabel=''
        self.ylabel=0
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.crosshairprice=0
        self.newChart=None

    def paintEvent(self, event):
        if self.chart().title()!="":
            super().paintEvent(event)
            qp = QPainter(self.viewport())
            #qp.begin(self)
            self.crossHair(event, qp)
            #qp.end()

    def crossHair(self,event,qp):
        qp.setPen(QColor(Qt.lightGray))
        qp.drawLine(0,self.y,self.viewport().width(),self.y)
        qp.drawLine(self.x,0,self.x,self.viewport().height())
        qp.setPen(Qt.red);
        self.crosshairprice=round(self.ylabel,2)
        qp.drawText(15,self.y,str(round(self.ylabel,2)))
        qp.drawText(self.x,self.viewport().height()-20,self.xlabel)

    def showContextMenu(self, pos):
        menu = QMenu(self)
        self.menu=menu
        item_add_act0 = QAction("ResetZoom", self)
        item_add_act0.triggered.connect(self.add_cb0)
        menu.addAction(item_add_act0)
        item_add_act1 = QAction("Buy", self)
        item_add_act1.triggered.connect(self.add_cb1)
        menu.addAction(item_add_act1)
        item_add_act2 = QAction("Sell", self)
        item_add_act2.triggered.connect(self.add_cb2)
        menu.addAction(item_add_act2)
        if self.chart().title() != "":
            menu.popup(QCursor.pos())

    def add_cb0(self, pos):
        self.chart().zoomReset()
        pass

    def add_cb1(self, pos):
        window.buySellDailog.buy(self.chart().symbol,1,self.crosshairprice)
        window.buySellDailog.exec_()


    def add_cb2(self, pos):
        window.buySellDailog.sell(self.chart().symbol,1,self.crosshairprice)
        window.buySellDailog.exec_()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        try:
            self.x=event.x()
            self.y=event.y()
            wp=event.localPos()
            sp=self.mapToScene(wp.x(),wp.y())
            ms=self.chart().mapFromScene(sp.x(),sp.y())
            mv=self.chart().mapToValue(ms)
            self.ylabel=mv.y()
            i = 0 if int(round(mv.x(),0)) < 0 else int(round(mv.x(),0))
            self.xlabel=self.chart().xseries[i]
        except Exception as e:
            pass
        self.viewport().repaint()

    def loadChart(self,symbol,history,positions,open_orders,closed_orders,type):
        self.newChart=Chart()
        self.setChart(self.newChart)
        self.newChart.loadChart(symbol,history,positions,open_orders,closed_orders,type)
        self.viewport().repaint()

    def updateTick(self,symbol,data):
        if self.newChart is not None:
            if self.newChart.symbol==symbol:
                self.newChart.updateCandleStick(symbol,data.open,data.high,data.low,data.close,data.start,window.timeFrame.currentText())
                #self.viewport().repaint()

#display chart
class Chart(QtCharts.QChart):
    def __init__(self):
        super().__init__()
        #self.setAnimationOptions(QtCharts.QChart.AllAnimations)
        self.x=0
        self.y=0
        self.high=0
        self.low=0
        self.symbol=None

    def updateCandleStick(self,symbol,open,high,low,close,t,type):
        idx=None
        newts=None
        try:
            if type == 'Day':
                newts=datetime.strftime(t, '%Y-%m-%d')
                idx = self.ts.index(newts)
            else:
                newts=datetime.strftime(t, '%Y-%m-%d %H:%M')
                idx = self.ts.index(newts)
        except Exception as e:
            pass
        if idx is None:
            css = QtCharts.QCandlestickSet()
            self.cs.remove(self.cs.sets()[0])
            del self.ts[0]
            css.setHigh(high)
            css.setLow(low)
            css.setOpen(open)
            css.setClose(close)
            self.cs.append(css)
            self.ts.append(newts)
        else:
            lcss = self.cs.sets()[-1]
            if high > lcss.high():
                lcss.setHigh(high)
            if low < lcss.low():
                lcss.setLow(low)
            lcss.setClose(close)
            bv = self.cs.take(lcss)
            self.cs.append(lcss)
            pass


    def loadChart(self,symbol,history,positions,open_orders,closed_orders,type):

        #self.removeAllSeries()
        self.legend().setVisible(False)
        self.setTitle(symbol)
        self.symbol=symbol

        #history
        self.cs = QtCharts.QCandlestickSeries()
        self.cs.setIncreasingColor(Qt.green)
        self.cs.setDecreasingColor(Qt.red)
        self.ts=[]
        for row in history.iterrows():
            css = QtCharts.QCandlestickSet()
            css.setHigh(row[1]['high'])
            css.setLow(row[1]['low'])
            css.setOpen(row[1]['open'])
            css.setClose(row[1]['close'])
            self.cs.append(css)
            if type=='Day':
                self.ts.append(datetime.strftime(row[0],'%Y-%m-%d'))
            else:
                self.ts.append(datetime.strftime(row[0],'%Y-%m-%d %H:%M'))


        #closed buy order
        sscob = QtCharts.QScatterSeries()
        sscob.setName('Closed Buy Orders')
        sscob.setMarkerShape(QtCharts.QScatterSeries.MarkerShapeCircle)
        sscob.setMarkerSize(15.0)
        sscob.setPen(QPen(Qt.green))
        if closed_orders is not None:
            for row in closed_orders.iterrows():
                if row[1]['side']=='buy':
                    idx=0
                    try:
                        if type=='Day':
                            idx=self.ts.index(datetime.strftime(row[1]['filled_at'],'%Y-%m-%d'))
                        else:
                            idx=self.ts.index(datetime.strftime(row[1]['filled_at'],'%Y-%m-%d %H:%M'))
                        sscob.append(float(idx+.5),float(row[1]['filled_avg_price']))
                        sscob.setPointLabelsFont(QFont(family='Arial',pointSize=3,weight=QFont.Thin))
                        sscob.setPointLabelsFormat("B:{}@{}".format(row[1]['qty'],row[1]['filled_avg_price']))
                        sscob.setPointLabelsClipping(False)
                        sscob.setPointLabelsVisible(True)
                    except Exception as e:
                        pass

        #closed sell order
        sscos = QtCharts.QScatterSeries()
        sscos.setName('Closed Sell Orders')
        sscos.setMarkerShape(QtCharts.QScatterSeries.MarkerShapeCircle)
        sscos.setMarkerSize(15.0)
        sscos.setPen(QPen(Qt.red))
        if closed_orders is not None:
            for row in closed_orders.iterrows():
                if row[1]['side']=='sell':
                    idx=0
                    try:
                        if type=='D':
                            idx=self.ts.index(datetime.strftime(row[1]['filled_at'],'%Y-%m-%d'))
                        else:
                            idx=self.ts.index(datetime.strftime(row[1]['filled_at'],'%Y-%m-%d %H:%M'))
                        sscos.append(float(idx+.5),float(row[1]['filled_avg_price']))
                        sscos.setPointLabelsFont(QFont(family='Arial',pointSize=3,weight=QFont.Thin))
                        sscos.setPointLabelsFormat("S:{}@{}".format(row[1]['qty'],row[1]['filled_avg_price']))
                        sscos.setPointLabelsClipping(False)
                        sscos.setPointLabelsVisible(True)
                    except Exception as e:
                        pass

        self.low=history['low'].min()
        self.high=history['high'].max()

        #add series and xaxis
        self.addSeries(self.cs)
        self.addSeries(sscob)
        self.addSeries(sscos)
        self.createDefaultAxes()
        self.axisX(self.cs).setCategories(self.ts)
        self.axisY(self.cs).setRange(self.low, self.high)
        self.axisX(sscob).setRange(0, len(self.ts))
        self.axisX(sscob).setVisible(False)
        self.axisY(sscob).setRange(self.low, self.high)
        self.axisX(sscos).setRange(0, len(self.ts))
        self.axisX(sscos).setVisible(False)
        self.axisY(sscos).setRange(self.low, self.high)


        # used to show the crosshair values
        self.yseries = self.cs
        self.xseries = self.ts

# endregion

if __name__ == "__main__":

    # region Intilialize
    #GUI
    app = QApplication(sys.argv)

    #get minute history for list of symbol
    minHistoryThread=QThread()
    minHistory=MinuteHistory()
    minHistory.moveToThread(minHistoryThread)

    window=MainWindow()
    window.show()

    #portfolio related data
    portfolioThread=QThread()
    portfolio=Portfolio(api)
    portfolio.moveToThread(portfolioThread)

    #data stream
    dataStream = StreamingData(key_id, secret_key, base_url)

    #watchlist selector
    watchListSelector=WatchListSelector()

    #algo selector
    algoSelector=AlgoSelector()

    # algo1
    algo1Thread = QThread()
    algo1 = Algo1()
    algo1.moveToThread(algo1Thread)

    # algo2
    algo2Thread = QThread()
    algo2 = Algo2()
    algo2.moveToThread(algo2Thread)

    #endregion

    # region Controller (Signals and slots)

    # connect Logic and GUI signals
    #fill GUI component when data is updated
    try:
        portfolio.positionsLoaded.connect(window.openPositions.loadData)
        portfolio.openOrderLoaded.connect(window.openOrders.loadData)
        portfolio.closedOrderLoaded.connect(window.closedOrders.loadData)
        portfolio.stockDataReady.connect(window.chartview.loadChart)
        portfolio.sendBuyingPower.connect(window.displayAccountData)
        portfolio.sendMessage.connect(window.statusMessage)

        window.watchLists.requestStockData.connect(portfolio.sendStockData)
        window.timeFrame.requestStockData.connect(portfolio.sendStockData)
        window.requestStockData.connect(portfolio.sendStockData)
        window.openPositions.requestStockData.connect(portfolio.sendStockData)
        window.openOrders.requestStockData.connect(portfolio.sendStockData)
        window.closedOrders.requestStockData.connect(portfolio.sendStockData)
        window.openOrders.requestCancel.connect(portfolio.cancel)
        window.openOrders.requestCancelAll.connect(portfolio.cancelAll)
        window.buySellDailog.sellRequest.connect(portfolio.sell)
        window.buySellDailog.buyRequest.connect(portfolio.buy)

        dataStream.sendAccountData.connect(window.displayAccountData)
        dataStream.sendQuote.connect(window.watchLists.updateQuote)
        dataStream.sendTick.connect(window.watchLists.updateTick)
        dataStream.sendTick.connect(window.openPositions.updateTick)
        dataStream.sendTick.connect(window.chartview.updateTick)
        dataStream.sendMTick.connect(window.chartview.updateTick)
        dataStream.sendMessage.connect(window.statusMessage)
        dataStream.sendMessage0.connect(window.statusMessage0)

        dataStream.getMinuteHistory.connect(minHistory.getHistory)
        dataStream.runAlgo1.connect(algo1.runAlgo)
        dataStream.runAlgo2.connect(algo2.runAlgo)
        #dataStream.sendTick.connect(portfolio.saveTicks)
        #dataStream.sendMTick.connect(portfolio.saveTicks)

        watchListSelector.listOfWatchList.connect(window.watchListcombo.loadData)
        window.watchListcombo.selectWatchList.connect(watchListSelector.selectWatchList)
        watchListSelector.watchlistSelected.connect(window.watchLists.loadData)

        algoSelector.listOfAlgo.connect(window.algoCombo.loadData)
        window.algoCombo.selectAlgo.connect(algoSelector.selectAlgo)


    except Exception as e:
        pass

    # endregion

    #region run

    #Load GUI with data

    #start min history thread
    minHistoryThread.start()

    # load watch lists in GUI
    watchListSelector.sendWatchListNames()

    # load algo lists in GUI
    algoSelector.sendAlgoNames()

    #start the threads for algos
    algo1Thread.start()
    algo2Thread.start()

    #Send positions, open orders, closed order to GUI
    portfolioThread.started.connect(portfolio.sendPortFolio)
    portfolioThread.start()

    async def sendportfolio():
        portfolio.sendPortFolio()
    ael3 = AsyncEventLooper()
    ael3.add_periodic_task(sendportfolio, 10)
    ael3.start()

    #get a lis of all symbols in the portfolio for subrcibing data from polugon
    portfolioSymbols=portfolio.allSymbols()
    channels = ['account_updates','trade_updates']
    channels+= dataStream.getChannels(portfolioSymbols)
    dataStream.conn.loop.run_until_complete(asyncio.gather(dataStream.conn.subscribe(channels)))

    #start the polygon streming loop in different thread, so that it wont block the program
    t = threading.Thread(target=AsyncEventLooper.classasyncioLooper, args=(dataStream.conn.loop,))
    t.start()


    #run the main loop for GUI
    sys.exit(app.exec_())
    #endregion