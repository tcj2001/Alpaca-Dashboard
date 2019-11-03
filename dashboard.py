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
import threading
import pytz  # $ pip install pytz
import talib
import logging


from userlogic import *

#GUI
from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtCharts import *

# region Model (Logic)

#portfolio class handle alpaca data
class Portfolio(QObject):

    sendMessageTL=Signal(str)
    sendBuyingPower = Signal(float,float)
    openOrderLoaded = Signal(df.DataFrame)
    closedOrderLoaded = Signal(df.DataFrame)
    positionsLoaded = Signal(df.DataFrame)
    stockDataReady = Signal(str, df.DataFrame, df.DataFrame, df.DataFrame, df.DataFrame, str)

    def __init__(self,env):
        super().__init__()
        self.api=env.api
        self.stockHistory = {}
        self.stockOrdered = {}
        self.stockPosition = {}
        self.stockPartialPosition = {}
        self.stockFilledAt = {}
        self.buying_power=0

    def snapshot(self,symbol):
        ss = self.api.polygon.snapshot(symbol)
        bid=ss.ticker['lastQuote']['p']
        if bid==0:
            bid=ss.ticker['lastTrade']['p']
        ask=ss.ticker['lastQuote']['P']
        if ask==0:
            ask=ss.ticker['lastTrade']['p']
        return ss.ticker['day']['o'],ss.ticker['day']['h'],ss.ticker['day']['l'],bid,ss.ticker['lastTrade']['p'],ask

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
                    self.opos_df = self.opos_df.sort_values(by='filled_at', ascending=False)
                    self.stockFilledAt=dict(zip(self.opos_df['symbol'], self.opos_df['filled_at']))
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
                self.oord_df=oord_df.sort_values(by='submitted_at', ascending=False)
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
                self.cord_df=cord_df.sort_values(by='filled_at', ascending=False)
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

    def getHistory(self, symbol, multiplier, timeframe, fromdate, todate):
        #print('Fetching ...daily history for {}'.format(symbol))
            return self.api.polygon.historic_agg_v2(symbol, multiplier, timeframe, fromdate, todate).df

    def getMinutesHistory(self, symbol, minutes):
        #print('Fetching ...{} minutes history for {}'.format(minutes,symbol))
        return self.api.polygon.historic_agg(size="minute", symbol=symbol, limit=minutes).df

    def buy(self,symbol,qty,price,tag=None):
        try:
            do, dh, dl, bp, lp, ap = self.snapshot(symbol)
            if price > ap:
                o = self.api.submit_order(
                    symbol, qty=str(qty), side='buy',
                    type='stop', time_in_force='day',
                    stop_price=str(price),
                    extended_hours=False
                )
            else:
                o = self.api.submit_order(
                    symbol, qty=str(qty), side='buy',
                    type='limit', time_in_force='day',
                    limit_price=str(price),
                    extended_hours=True
                )
            self.sendMessageTL.emit('{} Buying {} Qty of {} @ {}'.format(tag,str(qty),symbol,str(price)))
        except Exception as e:
            del self.stockOrdered[symbol]
            self.sendMessageTL.emit('{} buy error: {}'.format(symbol,e.args[0]))

    def sell(self,symbol,qty,price,tag=None):
        try:
            do, dh, dl, bp, lp, ap = self.snapshot(symbol)
            if price < bp:
                o = self.api.submit_order(
                    symbol, qty=str(qty), side='sell',
                    type='stop', time_in_force='day',
                    stop_price=str(price),
                    extended_hours=False
                )
            else:
                o = self.api.submit_order(
                    symbol, qty=str(qty), side='sell',
                    type='limit', time_in_force='day',
                    limit_price=str(price),
                    extended_hours=True
                )
            self.sendMessageTL.emit('{} Selling {} Qty of {} @ {}'.format(tag,str(qty),symbol,str(price)))
        except Exception as e:
            del self.stockOrdered[symbol]
            self.sendMessageTL.emit('{} sell error: {}'.format(symbol,e.args[0]))

    def cancel(self,symbol,id):
        try:
            o = self.api.cancel_order(id)
            self.sendMessageTL.emit('Cancelling order {} of {}'.format(id,symbol))
        except Exception as e:
            self.sendMessageTL.emit(e.args[0])

    def cancelAll(self):
        try:
            o = self.api.cancel_all_orders()
            self.sendMessageTL.emit('Cancelling all order')
        except Exception as e:
            self.sendMessageTL.emit(e.args[0])

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
        start = datetime.date(datetime.today() + relativedelta(months=-4))
        end = datetime.date(datetime.today() + timedelta(days=1))
        if symbol=='':
            return
        if timeFrame == 'Minute':
            try:
                history = self.stockHistory[symbol]
            except Exception as e:
                history = self.getMinutesHistory(symbol, 120)
        if timeFrame == '5Minute':
            start = datetime.date(datetime.today() + relativedelta(days=-1))
            end = datetime.date(datetime.today() + timedelta(days=1))
            history = self.getHistory(symbol, 5, 'minute', start, end)
        if timeFrame == '15Minute':
            start = datetime.date(datetime.today() + relativedelta(days=-2))
            end = datetime.date(datetime.today() + timedelta(days=1))
            history = self.getHistory(symbol, 15, 'minute', start, end)
        if timeFrame == 'Day':
            start = datetime.date(datetime.today() + relativedelta(months=-4))
            end = datetime.date(datetime.today() + timedelta(days=1))
            history = self.getHistory(symbol, 1, 'day', start, end)

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
        self.buying_power=float(self.account.buying_power)
        self.sendBuyingPower.emit(float(self.account.buying_power),float(self.account.last_equity)-float(self.account.equity))

#handles polygon data
class StreamingData(QObject):
    sendMessageTL = Signal(str)
    sendMessageTR=Signal(str)
    sendMessageBR=Signal(str)
    sendQuote=Signal(str,dict)
    sendTrade=Signal(str,dict)
    sendTick=Signal(str,dict)
    sendMTick = Signal(str, dict)
    sendAccountData = Signal(float,float)
    getMinuteHistory = Signal()

    def __init__(self, env):
        super().__init__()
        self.portfolio=env.portfolio
        self.conn=env.conn
        self.minHistory=env.minHistory
        self.env=env


    def setup(self,key_id,secret_key,base_url):
        self.key_id=key_id
        self.secret_key=secret_key
        self.base_url=base_url
        conn=self.conn
        @conn.on('status')
        async def on_status_messages(conn, channel, data):
            print(data)
            self.sendMessageBR.emit(data.message)

        @conn.on('account_updates')
        async def on_account_updates(conn, channel, data):
            self.sendAccountData.emit(float(data['buying_power']),float(data['last_equity'])-float(data['equity']))
            print(data)

        @conn.on('trade_updates')
        async def on_trade_updates(conn, channel, data):
            if data.event=='new':
                self.sendMessageTL.emit("new {} order placed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['qty'], data.order['submitted_at']))
            if data.event=='partially_filled':
                self.sendMessageTL.emit("{} order partially executed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['filled_qty'], data.order['filled_at']))
            if data.event=='fill':
                self.sendMessageTL.emit("{} order executed for {} with {} qty at {}".format(data.order['side'], data.order['symbol'], data.order['filled_qty'], data.order['filled_at']))

        @conn.on('A')
        async def on_tick_messages(conn, channel, data):
            #print(self.base_url)
            self.sendMessageTR.emit('{}:{}'.format(data.symbol,data.close))
            self.sendTick.emit(data.symbol,data)
            self.portfolio.saveTicks(data.symbol,data)
            for obj in self.env.algosubclasses:
                obj.setData(data.symbol,data)
                obj.runAlgo()


        @conn.on('AM')
        async def on_minute_messages(conn, channel, data):
            self.sendMessageTR.emit('{}:{}'.format(data.symbol,data.close))
            self.sendMTick.emit(data.symbol, data)
            self.portfolio.saveTicks(data.symbol,data)
            for obj in self.env.algosubclasses:
                obj.setData(data.symbol,data)
                obj.runAlgo()

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
            self.minHistory.setSymbolList(symbolList)
            self.getMinuteHistory.emit()
            for symbol in symbolList:
                symbol_channels = ['A.{}'.format(symbol), 'AM.{}'.format(symbol), 'Q.{}'.format(symbol)]
                channelList += symbol_channels
        return channelList

    async def subscribeChannels(self,channels):
        if channels is not None:
            await self.conn.subscribe(channels)

    def subscribeSymbols(self,symbol):
        channels= self.getChannels(symbol)
        if channels is not None:
            self.conn.loop.stop()
            while self.conn.loop.is_running():
                pass
            self.conn.loop.run_until_complete(asyncio.gather(self.conn.subscribe(channels)))
            self.env.t = threading.Thread(target=AsyncEventLooper.classasyncioLooper, args=(self.conn.loop,))
            self.env.t.start()

#get minutes history
class MinuteHistory(QObject):
    sendMessageBR=Signal(str)

    def __init__(self,env,minutes):
        super().__init__()
        self.api=env.api
        self.portfolio=env.portfolio
        self.minutes=minutes
    def setSymbolList(self,symbols):
        self.symbolList=symbols
    def getHistory(self):
        for symbol in self.symbolList:
            self.sendMessageBR.emit("Fetching {} minute history for {}".format(self.minutes,symbol))
            history = self.api.polygon.historic_agg(size="minute", symbol=symbol, limit=self.minutes).df
            self.portfolio.saveHistory(symbol, history)

# get last quote and trade info for watchlist symbols
class WatchListData(QObject):
    def __init__(self, api, symbols):
        super().__init__()
        self.api=api
        self.symbols=symbols
        if self.symbols is not None:
            tickers = self.api.polygon.all_tickers()
            if tickers.__len__()!=0:
                self.selectedTickers = [ticker for ticker in tickers if (ticker.ticker in symbols)]
                wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in self.selectedTickers]
            else:
                wlist = [[symbol, 0, 0, 0] for symbol in self.symbols]

            self.wl_df = df.DataFrame.from_records(wlist)
            if not self.wl_df.empty:
                self.symbols = self.wl_df[0].tolist()

    def getSymbols(self):
        return self.symbols

#watchlist class
class WatchLists(QObject):
    def __init__(self,env,name):
        super().__init__()
        self.selectedSymbols=[]
        self.env=env
        self.name=name

    def getSymbols(self):
        return self.selectedSymbols

    def save(self,symbols):
        pickle.dump(symbols, open(self.env.env + '/watchlist_'+self.name + '.p', "wb"))

    def load(self):
        try:
            symlist = pickle.load(open(self.env.env + '/watchlist_'+self.name+ '.p', "rb"))
            if symlist is not None:
                self.selectedSymbols=symlist
        except Exception as e:
            pickle.dump(self.selectedSymbols, open(self.env.env + '/watchlist_' + self.name + '.p', "wb"))
        return self.selectedSymbols

#allows to select watchlist
class WatchListSelector(QObject):

    watchlistSelected = Signal(df.DataFrame)
    listOfWatchList = Signal(list,str)
    sendMessageBR=Signal(str)

    def __init__(self,env):
        super().__init__()
        self.env=env
        self.dataStream=env.dataStream
        self.watchLists = [x.name for x in self.env.watchlistsobject]

    def selectWatchList(self, id):
        for obj in self.env.watchlistsobject:
            if obj.name == id:
                obj.load()
                symbols = obj.getSymbols()
                wl = WatchListData(self.env.api, symbols)
                if wl.getSymbols() is not None:
                    self.watchlistSelected.emit(wl.wl_df)
                else:
                    self.watchlistSelected.emit(df.DataFrame())
                #self.env.window.addButton.setEnabled(True)

    def addSymbol(self,txt):
        if '.' in txt:
            watchlistname,symbol=txt.rsplit('.',1)
            watchlistname=watchlistname.capitalize()
            symbol=symbol.upper()
            found=False
            for obj in self.env.watchlistsobject:
                if obj.name == watchlistname:
                    found=True
                    break
            if not found:
                obj = getattr(__import__(__name__), WatchLists.__name__)(self.env,watchlistname)
                self.env.watchlistsobject.append(obj)
            obj.load()
            obj.selectedSymbols.append(symbol)
            obj.save(obj.selectedSymbols)
            self.watchLists = [x.name for x in self.env.watchlistsobject]
            self.sendWatchListNames(watchlistname)
        else:
            watchlistname=self.watchListcombo.currentText()
            symbol=txt.upper()
            found=False
            for obj in self.env.watchlistsobject:
                if obj.name == watchlistname:
                    found=True
                    break
            if found:
                obj.load()
                obj.selectedSymbols.append(symbol)
                obj.save(obj.selectedSymbols)
                self.sendWatchListNames(watchlistname)

        self.subscribeSymbols.emit([symbol])
        self.requestStockData.emit(symbol, self.timeFrame.currentText())
        self.timeFrame.symbol = symbol


    def sendWatchListNames(self,watchlist):
        self.listOfWatchList.emit(self.watchLists,watchlist)

#allows to select scanners
class ScannerSelector(QObject):

    scannerSelected = Signal(df.DataFrame)
    listOfScanner = Signal(list)
    sendMessageBR=Signal(str)

    def __init__(self,env):
        super().__init__()
        self.env=env
        self.dataStream=env.dataStream
        self.scannerLists = [cls.__name__ for cls in Scanners.__subclasses__()]

    def selectScanner(self, id):
        for obj in self.env.scannersubclasses:
            if obj.__class__.__name__==id:
                symbols=obj.getSymbols()
                wl=WatchListData(self.env.api,symbols)
                if wl.getSymbols() is not None:
                    self.scannerSelected.emit(wl.wl_df)
                else:
                    self.scannerSelected.emit(df.DataFrame())
                #self.env.window.addButton.setEnabled(False)

    def sendScannerNames(self):
        self.listOfScanner.emit(self.scannerLists)

#allows to select algo
class AlgoSelector(QObject):

    listOfAlgo = Signal(list)

    def __init__(self):
        super().__init__()
        self.allogsubclasses = [cls.__name__ for cls in Algos.__subclasses__()]
        unselected = [False for cls in Algos.__subclasses__()]
        self.algoLists=dict(zip(self.allogsubclasses,unselected))


    def sendAlgoNames(self):
        self.listOfAlgo.emit(list(self.algoLists.keys()))

    def selectAlgo(self,str,ischecked):
        self.algoLists[str]=ischecked
        pass

# endregion

# region View (GUI)
#tabbed window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Alpaca Dashboard!'
        # Get the current screens' dimensions...
        screen = QDesktopWidget().frameGeometry()
        # ... and get this windows' dimensions
        mysize = self.geometry()
        # The horizontal position is calulated as screenwidth - windowwidth /2
        hpos = ( screen.width() - mysize.width() ) / 2
        # And vertical position the same, but with the height dimensions
        vpos = ( screen.height() - mysize.height() ) / 2
        # And the move call repositions the window
        self.move(100, 100)

        self.setWindowTitle(self.title)

        mw=QWidget()
        layout=QGridLayout()
        mw.setLayout(layout)
        self.setCentralWidget(mw)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs,0,0,1,1)

        self.tab1 = QWidget()
        layout1=QGridLayout()
        self.tab1.setLayout(layout1)

        self.tab2 = QWidget()
        layout2=QGridLayout()
        self.tab2.setLayout(layout2)

        self.tabs.addTab(self.tab1,'Live')
        self.tabs.addTab(self.tab2,'Paper')

#Env Window
class EnvWindow(QWidget):

    requestStockData = Signal(str, str)
    subscribeSymbols = Signal(list)
    addSymbolToWatchlist = Signal(str)


    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env=env
        self.buyingPower=0
        self.buySellDailog=BuySellDialog()

        mg=QGridLayout()
        self.setLayout(mg)

        #region top rows
        trow = QWidget()
        trowl = QGridLayout()

        lb51=QLabel('Symbol')
        trowl.addWidget(lb51,0,0,1,1,Qt.AlignRight)
        self.newSymbol = QLineEdit()
        self.newSymbol.setMaxLength(10)
        self.newSymbol.setMaximumWidth(100)
        trowl.addWidget(self.newSymbol,0,1,1,1,Qt.AlignLeft)
        self.addButton=QPushButton("Add")
        trowl.addWidget(self.addButton,0,2,1,1,Qt.AlignLeft)
        self.addButton.clicked.connect(self.addSymbol)

        lb1=QLabel('WatchLists')
        self.watchListcombo=WatchListCombo()
        trowl.addWidget(lb1,0,3,1,1,Qt.AlignRight)
        trowl.addWidget(self.watchListcombo, 0, 4, 1, 1,Qt.AlignLeft)


        lb2 = QLabel('Scanners')
        self.scannercombo = ScannerCombo()
        trowl.addWidget(lb2, 0, 5, 1, 1, Qt.AlignRight)
        trowl.addWidget(self.scannercombo, 0, 6, 1, 1, Qt.AlignLeft)

        lb3=QLabel('Algo')
        self.algoCombo = AlgoCombo()
        trowl.addWidget(lb3,0,7,1,1,Qt.AlignRight)
        trowl.addWidget(self.algoCombo,0,8,1,1,Qt.AlignLeft)

        trow.setLayout(trowl)
        mg.addWidget(trow, 0, 0, 1, 1)
        #endregion

        hs =QSplitter(Qt.Horizontal)
        hs.setStyleSheet("QSplitter::handle { background-color: lightGray; }")
        mg.addWidget(hs,2,0,1,9)

        #region middle left
        vs = QSplitter(Qt.Vertical)
        vs.setStyleSheet("QSplitter::handle { background-color: lightGray; }")

        lw1 = QWidget()
        lg1 = QGridLayout()
        self.watchListTable = WatchListTable(self, ['Symbol', 'Bid', 'Last', 'Ask'])
        lg1.addWidget(self.watchListTable, 1, 0, 1, 1)
        lw1.setLayout(lg1)
        vs.addWidget(lw1)

        lw2 = QWidget()
        lg2 = QGridLayout()
        lb2=QLabel('Positions')
        self.openPositions = Positions(self,['Symbol','Qty','last','AveragePrice','Profit%','FilledAt'])
        lg2.addWidget(lb2,0,0,1,1)
        lg2.addWidget(self.openPositions,1,0,1,1)
        lw2.setLayout(lg2)
        vs.addWidget(lw2)

        lw4 = QWidget()
        lg4 = QGridLayout()
        lb4=QLabel('Open Orders')
        self.openOrders = OpenOrder(self,['Symbol','Type','Qty','LimitPrice','StopPrice','SubmittedAt','Id'])
        lg4.addWidget(lb4,0,0,1,1)
        lg4.addWidget(self.openOrders,1,0,1,1)
        lw4.setLayout(lg4)
        vs.addWidget(lw4)

        lw3 = QWidget()
        lg3 = QGridLayout()
        lb3=QLabel('Closed Orders')
        self.closedOrders = ClosedOrder(self,['Symbol','Type','Qty','Price','FilledAt'])
        lg3.addWidget(lb3,2,0,1,1)
        lg3.addWidget(self.closedOrders,3,0,1,1)
        lw3.setLayout(lg3)
        vs.addWidget(lw3)
        #endregion

        #region middle right
        hs.addWidget(vs)

        rw=QWidget()
        rg = QGridLayout()


        lb6=QLabel('TimeFrame')
        self.timeFrame=TimeFrame()
        rg.addWidget(lb6,0,0,1,1,Qt.AlignRight)
        rg.addWidget(self.timeFrame,0,1,1,1,Qt.AlignLeft)

        self.chartview = ChartView(self)
        rg.addWidget(self.chartview,1,0,1,2)
        rw.setLayout(rg)
        hs.addWidget(rw)
        #endregion

        #region status lines
        self.statusLineTL=QLabel('')
        self.statusLineTR=QLabel('')
        self.statusLineBL=QLabel('')
        self.statusLineBR=QLabel('')
        self.statusLineTL.setWordWrap(True)
        self.statusLineTR.setWordWrap(True)
        self.statusLineBL.setWordWrap(True)
        self.statusLineBR.setWordWrap(True)
        mg.addWidget(self.statusLineTL,3,0,1,1)
        mg.addWidget(self.statusLineTR,3,1,1,8)
        mg.addWidget(self.statusLineBL,4,0,1,1)
        mg.addWidget(self.statusLineBR,4,1,1,8)
        #endregion

        self.statusLineBL.setText("BuyingPower: {}".format(self.buyingPower))

    def addSymbol(self):
        if self.newSymbol.text()!='':
            self.addSymbolToWatchlist.emit(self.newSymbol.text())
            self.newSymbol.setText('')

    def displayAccountData(self,bp,pf):
        self.buyingPower=bp
        self.todaysProfit=pf
        self.statusLineBL.setText("BuyingPower: {} .....  Todays Profit: {}".format(self.buyingPower,self.todaysProfit))

    def statusMessageTL(self,msg):
        self.statusLineTL.setText(msg)
    def statusMessageTR(self,msg):
        self.statusLineTR.setText(msg)
    def statusMessageBL(self,msg):
        self.statusLineBL.setText(msg)
    def statusMessageBR(self, msg):
        self.statusLineBR.setText(msg)


#watch list combo
class WatchListCombo(QComboBox):

    selectWatchList = Signal(str)

    def __init__(self):
        super().__init__()
        self.currentIndexChanged.connect(self.selectionChanged)
        self.view().pressed.connect(self.selectionChanged)
        self.symbol=None

    def loadData(self,list,name):
        self.clear()
        self.addItems(list)
        if name !='':
            self.setCurrentText(name)

    def selectionChanged(self):
        self.selectWatchList.emit(self.currentText())
        pass

#scanner combo
class ScannerCombo(QComboBox):

    selectScannerList = Signal(str)

    def __init__(self):
        super().__init__()
        self.currentIndexChanged.connect(self.selectionChanged)
        self.view().pressed.connect(self.selectionChanged)
        self.symbol=None

    def loadData(self,list):
        self.addItems(list)

    def selectionChanged(self):
        self.selectScannerList.emit(self.currentText())
        pass

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
        self.addItems(['Minute','5Minute','15Minute','Day'])
        self.currentIndexChanged.connect(self.selectionChanged)
        self.symbol=None


    def selectionChanged(self):
        self.requestStockData.emit(self.symbol, self.currentText())
        pass

#watchlist table
class WatchListTable(QTableView):

    requestStockData = Signal(str, str)

    def __init__(self,window,columns):
        super().__init__()
        self.window=window
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
        self.clicked.connect(self.handleClicked)

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
        self.window.buySellDailog.sell(symbol,1,price)
        self.window.buySellDailog.exec_()

    def buy(self, symbol, qty , price):
        self.window.buySellDailog.buy(symbol,1,price)
        self.window.buySellDailog.exec_()

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
        if selected.indexes().__len__() != 0:
            symbol = self.selectionModel().selectedRows()[0].data()
            self.window.timeFrame.symbol = symbol
            self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

    def handleClicked(self, qsim):
        symbol=qsim.data()
        self.window.timeFrame.symbol=symbol
        self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

#positions table
class Positions(QTableView):

    requestStockData = Signal(str, str)

    def __init__(self,window, columns):
        super().__init__()
        self.window=window
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
        self.clicked.connect(self.handleClicked)
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
        self.window.buySellDailog.sell(symbol,qty,price)
        self.window.buySellDailog.exec_()

    def buy(self, symbol, qty, price):
        self.window.buySellDailog.buy(symbol,qty,price)
        self.window.buySellDailog.exec_()

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
            self.window.timeFrame.symbol=symbol
            self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

    def handleClicked(self, qsim):
        symbol=qsim.data()
        self.window.timeFrame.symbol=symbol
        self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

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

    def __init__(self,window, columns):
        super().__init__()
        self.window=window
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
        self.clicked.connect(self.handleClicked)
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
            self.window.timeFrame.symbol=self.selectedSymbol
            self.requestStockData.emit(self.selectedSymbol,self.window.timeFrame.currentText())

    def handleClicked(self, qsim):
        symbol=qsim.data()
        self.window.timeFrame.symbol=symbol
        self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

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
    def __init__(self,window,columns):
        super().__init__()
        self.window=window
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
        self.clicked.connect(self.handleClicked)
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
            self.window.timeFrame.symbol=symbol
            self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

    def handleClicked(self, qsim):
        symbol=qsim.data()
        self.window.timeFrame.symbol=symbol
        self.requestStockData.emit(symbol, self.window.timeFrame.currentText())

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
    def __init__(self,window):
        super().__init__()
        self.window=window
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
        self.window.buySellDailog.buy(self.chart().symbol,1,self.crosshairprice)
        self.window.buySellDailog.exec_()


    def add_cb2(self, pos):
        self.window.buySellDailog.sell(self.chart().symbol,1,self.crosshairprice)
        self.window.buySellDailog.exec_()


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
                self.newChart.updateCandleStick(symbol,data.open,data.high,data.low,data.close,data.start,self.window.timeFrame.currentText())
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
            if type == '5Minute':
                newts= datetime.strftime(t - timedelta(minutes=t.minute % 5,seconds=t.second,microseconds=t.microsecond), '%Y-%m-%d %H:%M')
            if type == '15Minute':
                newts= datetime.strftime(t - timedelta(minutes=t.minute % 15,seconds=t.second,microseconds=t.microsecond), '%Y-%m-%d %H:%M')
            if type == 'Minute':
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
            if type == 'Day':
                self.ts.append(datetime.strftime(row[0],'%Y-%m-%d'))
            if type == '5Minute':
                self.ts.append(datetime.strftime(row[0] - timedelta(minutes=row[0].minute % 5,seconds=row[0].second,microseconds=row[0].microsecond),'%Y-%m-%d %H:%M'))
            if type == '15Minute':
                self.ts.append(datetime.strftime(row[0] - timedelta(minutes=row[0].minute % 15,seconds=row[0].second,microseconds=row[0].microsecond),'%Y-%m-%d %H:%M'))
            if type == 'Minute':
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

class Env():
    def __init__(self,env):
        self.env=env

        try:
            os.mkdir(env)
        except Exception:
            pass

        self.window = EnvWindow(self)

        self.logging=   logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='portfolio.log',
                        filemode='w')

        # get keys form environment variable

        # alpaca api
        if env=='Paper':
            try:
                self.key_id = os.environ['PKEY_ID']
                self.secret_key = os.environ['PSECRET_KEY']
                self.base_url = os.environ['PBASE_URL']
            except Exception as e:
                raise Exception('Set API keys')
            self.api = tradeapi.REST(self.key_id, self.secret_key, self.base_url)
            self.conn = tradeapi.StreamConn(key_id=self.key_id, secret_key=self.secret_key, base_url=self.base_url)
        if env=='Live':
            try:
                self.key_id = os.environ['KEY_ID']
                self.secret_key = os.environ['SECRET_KEY']
                self.base_url = os.environ['BASE_URL']
            except Exception as e:
                raise Exception('Set API keys')
            self.api = tradeapi.REST(self.key_id, self.secret_key, self.base_url)
            self.conn = tradeapi.StreamConn(key_id=self.key_id, secret_key=self.secret_key, base_url=self.base_url)


        #portfolio related data
        self.portfolioThread=QThread()
        self.portfolio=Portfolio(self)
        self.portfolio.moveToThread(self.portfolioThread)

        # get minute history for list of symbol
        self.minHistoryThread = QThread()
        self.minHistory = MinuteHistory(self,100)
        self.minHistory.moveToThread(self.minHistoryThread)

        # data stream
        self.dataStream = StreamingData(self)
        self.dataStream.setup(self.key_id, self.secret_key, self.base_url)

        # get a lis of all symbols in the portfolio for subrcibing data from polugon
        self.portfolioSymbols = self.portfolio.allSymbols()
        self.channels = ['account_updates', 'trade_updates']
        self.channels += self.dataStream.getChannels(self.portfolioSymbols)
        self.dataStream.conn.loop.run_until_complete(asyncio.gather(self.dataStream.conn.subscribe(self.channels)))

        self.t = threading.Thread(target=AsyncEventLooper.classasyncioLooper, args=(self.dataStream.conn.loop,))
        self.t.start()


        #algo selector
        self.algoSelector=AlgoSelector()

        # instantiate algos subclasses and move to a new thread
        self.algosubclasses=[]
        self.algosubclassesThreads=[]
        for sc in Algos.__subclasses__():
            qt=QThread()
            self.algosubclassesThreads.append(qt)
            obj = getattr(__import__('userlogic'), sc.__name__)(self)
            self.algosubclasses.append(obj)
            obj.moveToThread(qt)
            qt.start()


        # __module__instantiate watchlists class from watchlist pickles
        watchlistpickles=[x for x in os.listdir('./'+self.env) if(x.startswith('watchlist_'))]
        self.watchlistsobject = []
        for pk in watchlistpickles:
            watchlistname=pk.rsplit('_',1)[1].rsplit('.')[0]
            obj = getattr(__import__(__name__), WatchLists.__name__)(self,watchlistname)
            self.watchlistsobject.append(obj)
            obj.load()
            self.dataStream.subscribeSymbols(obj.getSymbols())

        # instantiate scanner subclasses
        self.scannersubclasses = []
        for sc in Scanners.__subclasses__():
            obj = getattr(__import__('userlogic'), sc.__name__)(self)
            self.scannersubclasses.append(obj)
            obj.start()


        # watchlist selector
        self.watchListSelectorThread = QThread()
        self.watchListSelector = WatchListSelector(self)
        self.watchListSelector.moveToThread(self.watchListSelectorThread)

        # watchlist selector
        self.scannerSelectorThread = QThread()
        self.scannerSelector = ScannerSelector(self)
        self.scannerSelector.moveToThread(self.scannerSelectorThread)

    def setupSignals(self):
        # connect Logic and GUI signals
        # fill GUI component when data is updated
        try:
            self.portfolio.positionsLoaded.connect(self.window.openPositions.loadData)
            self.portfolio.openOrderLoaded.connect(self.window.openOrders.loadData)
            self.portfolio.closedOrderLoaded.connect(self.window.closedOrders.loadData)
            self.portfolio.stockDataReady.connect(self.window.chartview.loadChart)
            self.portfolio.sendBuyingPower.connect(self.window.displayAccountData)
            self.portfolio.sendMessageTL.connect(self.window.statusMessageTL)

            self.window.watchListTable.requestStockData.connect(self.portfolio.sendStockData)
            self.window.timeFrame.requestStockData.connect(self.portfolio.sendStockData)
            self.window.requestStockData.connect(self.portfolio.sendStockData)
            self.window.openPositions.requestStockData.connect(self.portfolio.sendStockData)
            self.window.openOrders.requestStockData.connect(self.portfolio.sendStockData)
            self.window.closedOrders.requestStockData.connect(self.portfolio.sendStockData)
            self.window.openOrders.requestCancel.connect(self.portfolio.cancel)
            self.window.openOrders.requestCancelAll.connect(self.portfolio.cancelAll)
            self.window.buySellDailog.sellRequest.connect(self.portfolio.sell)
            self.window.buySellDailog.buyRequest.connect(self.portfolio.buy)
            self.window.subscribeSymbols.connect(self.dataStream.subscribeSymbols)
            self.window.addSymbolToWatchlist.connect(self.watchListSelector.addSymbol)

            self.dataStream.sendAccountData.connect(self.window.displayAccountData)
            self.dataStream.sendQuote.connect(self.window.watchListTable.updateQuote)
            self.dataStream.sendTick.connect(self.window.watchListTable.updateTick)
            self.dataStream.sendTick.connect(self.window.openPositions.updateTick)
            self.dataStream.sendTick.connect(self.window.chartview.updateTick)
            self.dataStream.sendMTick.connect(self.window.chartview.updateTick)
            self.dataStream.sendMessageTL.connect(self.window.statusMessageTL)
            self.dataStream.sendMessageTR.connect(self.window.statusMessageTR)
            self.dataStream.sendMessageBR.connect(self.window.statusMessageBR)

            self.watchListSelector.listOfWatchList.connect(self.window.watchListcombo.loadData)
            self.window.watchListcombo.selectWatchList.connect(self.watchListSelector.selectWatchList)
            self.watchListSelector.watchlistSelected.connect(self.window.watchListTable.loadData)

            self.scannerSelector.listOfScanner.connect(self.window.scannercombo.loadData)
            self.window.scannercombo.selectScannerList.connect(self.scannerSelector.selectScanner)
            self.scannerSelector.scannerSelected.connect(self.window.watchListTable.loadData)

            self.algoSelector.listOfAlgo.connect(self.window.algoCombo.loadData)
            self.window.algoCombo.selectAlgo.connect(self.algoSelector.selectAlgo)

            self.minHistory.sendMessageBR.connect(self.window.statusMessageBR)
            self.watchListSelector.sendMessageBR.connect(self.window.statusMessageBR)
            self.scannerSelector.sendMessageBR.connect(self.window.statusMessageBR)

        except Exception as e:
            pass

        self.dataStream.getMinuteHistory.connect(self.minHistory.getHistory)

    async def sendportfolio(self):
        self.portfolio.sendPortFolio()

    def run(self):
        # start min history thread
        self.minHistoryThread.start()

        # load watch lists in GUI
        self.watchListSelectorThread.start()
        self.watchListSelector.sendWatchListNames('')

        # load scanner lists in GUI
        self.scannerSelectorThread.start()
        self.scannerSelector.sendScannerNames()

        # load algo lists in GUI
        self.algoSelector.sendAlgoNames()

        # Send positions, open orders, closed order to GUI
        self.portfolioThread.started.connect(self.portfolio.sendPortFolio)
        self.portfolioThread.start()

        self.ael3 = AsyncEventLooper()
        self.ael3.add_periodic_task(self.sendportfolio, 10)
        self.ael3.start()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow=MainWindow()

    #live env
    live=Env('Live')
    live.setupSignals()
    live.run()

    #stop the conn.loop for next env
    live.dataStream.conn.loop.stop()
    while live.dataStream.conn.loop.is_running():
        pass

    # #paper env
    paper = Env('Paper')
    paper.setupSignals()
    paper.run()


    mainWindow.tab1.layout().addWidget(live.window)
    mainWindow.tab2.layout().addWidget(paper.window)
    mainWindow.show()
    sys.exit(app.exec_())
