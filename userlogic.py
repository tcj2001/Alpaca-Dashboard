######################################################################################################
# Author: Thomson Mathews
######################################################################################################
from datetime import timedelta, datetime, time
from pytz import timezone
from PySide2.QtCore import QObject
import pandas as df
import asyncio
import threading
import pickle


# region no change needed here
class AsyncEventLooper():
    def __init__(self):
        self._loop = asyncio.new_event_loop()

    # at class level
    def classasyncioLooper(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    # asyncio looper function for a thread
    def asyncioLooper(self, loop):
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
        # self._loop.run_forever()
        return


class Study():
    def __init__(self, history):
        self.history = history

    def addEMA(self, length):
        try:
            self.history['EMA' + str(length)] = talib.EMA(self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addSMA(self, length):
        try:
            self.history['EMA' + str(length)] = talib.SMA(self.history['close'], timeperiod=length)
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

    def addMACD(self, fast, slow, signal):
        try:
            macd, signalline, macdhist = talib.MACD(
                self.history['close'],
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            self.history['MACD_F' + str(fast) + '_S' + str(slow) + '_L' + str(signal) + '_SIGNAL'] = macd - signalline
            self.history['MACD_F' + str(fast) + '_S' + str(slow) + '_L' + str(signal) + '_MACD'] = macd
            self.history['MACD_F' + str(fast) + '_S' + str(slow) + '_L' + str(signal) + '_SL'] = signalline
            self.history['MACD_F' + str(fast) + '_S' + str(slow) + '_L' + str(signal) + '_HIST'] = macdhist
        except Exception as e:
            return None

    def addRSI(self, length):
        try:
            self.history['RSI' + str(length)] = talib.RSI(self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addATR(self, length):
        try:
            self.history['ATR' + str(length)] = talib.ATR(self.history['high'], self.history['low'],
                                                          self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addCCI(self, length):
        try:
            self.history['CCI' + str(length)] = talib.CCI(self.history['high'], self.history['low'],
                                                          self.history['close'], timeperiod=length)
        except Exception as e:
            return None

    def addBBANDS(self, length, devup, devdn, type):
        try:
            up, mid, low = talib.BBANDS(self.history['close'], timeperiod=length, nbdevup=devup, nbdevdn=devdn,
                                        matype=type)
            bbp = (self.history['close'] - low) / (up - low)
            self.history['BBANDS' + str(length) + '_bbp'] = bbp
            self.history['BBANDS' + str(length) + '_up'] = up
            self.history['BBANDS' + str(length) + '_low'] = low
            self.history['BBANDS' + str(length) + '_mid'] = mid
        except Exception as e:
            return None

    def bullishCandleStickPatterns(self, c1, c2, c3):
        self.bullishPattern = self.bullishCandleStickPatterns(self.history.iloc[-1], self.history.iloc[-2],
                                                              self.history.iloc[-3])
        pattern = None
        # LOCH bullish
        if c1.low < c1.open < c1.close <= c1.high and \
                c1.high - c1.close < c1.open - c1.low and \
                c1.close - c1.open < c1.open - c1.low:
            pattern = 'hammer'
        if c1.low <= c1.open < c1.close < c1.high and \
                c1.high - c1.close > c1.open - c1.low and \
                c1.close - c1.open < c1.high - c1.close:
            pattern = 'inverseHammer'
        # LCOH bearish
        if c2.low < c2.close < c2.open < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c1.open < c2.close and \
                c1.close - c1.open > c2.open - c2.close:
            pattern = 'bullishEngulfing'
        if c2.low < c2.close < c2.open < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c1.open < c2.close and \
                c1.close > c2.close + (c2.open - c2.close) / 2:
            pattern = 'piercingLine'
        if c3.low < c3.close < c3.open < c3.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                abs(c2.open - c2.close) < abs(c3.open - c3.close) and \
                abs(c2.open - c2.close) < abs(c1.open - c1.close):
            pattern = 'morningStar'
        if c3.low <= c3.open < c3.close < c3.high and \
                c2.low <= c2.open < c2.close < c2.high and \
                c1.low <= c1.open < c1.close < c1.high and \
                c3.close <= c2.open and \
                c2.close <= c1.open:
            pattern = 'threeWhiteSoldiers'
        return pattern

    def getHistory(self):
        return self.history


class Algos(QObject):
    def __init__(self):
        super().__init__()

    def setData(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def runAlgo(self):
        pass


class Scanners(QObject):
    def __init__(self):
        super().__init__()
        self.selectedSymbols = None

    def getSymbols(self):
        return self.selectedSymbols

    def start(self):
        ael1 = AsyncEventLooper()
        ael1.add_periodic_task(self.loop, self.milliseconds)
        ael1.start()

    async def loop():
        pass


# endregion

# region change or add scanners, algos, watchlist here

# names of these algo subclass will appear in the dashboard algo drop down
# buy/sell if close is above/below the last 15min high/low
class FifteenMinuteHigh(Algos):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def runAlgo(self):
        try:
            # return if algo is not checked
            if not self.env.algoSelector.algoLists[self.__class__.__name__]:
                return
            # prevent reordering till last order is filled or rejected
            if self.env.portfolio.stockOrdered.get(self.symbol) is not None:
                return
            # not enough history
            if self.env.portfolio.stockHistory.get(self.symbol) is None:
                return
            # print(self.env.env, self.__class__.__name__,self.symbol)
            # check algo starttime
            ts = self.data.start
            ts -= timedelta(seconds=ts.second, microseconds=ts.microsecond)
            if ts < ts.replace(hour=9, minute=45, second=0, microsecond=0):
                return
            # write your algo here
            study = Study(self.env.portfolio.stockHistory[self.symbol].resample('15T').first().fillna(method='ffill'))
            studyHistory = study.getHistory()
            close = studyHistory['close'][-1]
            last3barmax = studyHistory['high'].tail(2).max()
            last3barmin = studyHistory['high'].tail(2).min()
            qty = 0
            if self.env.portfolio.stockPosition.get(self.symbol) is not None:
                qty = self.env.portfolio.stockPosition.get(self.symbol)
            if close >= last3barmax:
                if qty <= 0:
                    buyqty = 1
                    if self.env.portfolio.buying_power > close * buyqty:
                        self.env.portfolio.stockOrdered[self.symbol] = True
                        self.env.portfolio.buy(self.symbol, buyqty, close, 'Algo1')

            if close <= last3barmin:
                if qty > 0:
                    # avoid daytrade
                    if not (self.env.portfolio.stockFilledAt[self.symbol] is df.NaT or
                            self.env.portfolio.stockFilledAt[self.symbol]._date_repr == datetime.today().astimezone(
                                timezone('America/New_York')).strftime('%Y-%m-%d')):
                        self.env.portfolio.stockOrdered[self.symbol] = True
                        self.env.portfolio.sell(self.symbol, 1, close, 'Algo1')
        except Exception as e:
            pass


# buy/sell if close is above/below 20ema for symmbols in top20ema count watchlist
class EMA20(Algos):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def runAlgo(self):
        try:
            # return if algo is not checked
            if not self.env.algoSelector.algoLists[self.__class__.__name__]:
                return
            # prevent reordering till last order is filled or rejected
            if self.env.portfolio.stockOrdered.get(self.symbol) is not None:
                return
            # not enough history
            if self.env.portfolio.stockHistory.get(self.symbol) is None:
                return
            # check algo starttime
            ts = self.data.start
            ts -= timedelta(seconds=ts.second, microseconds=ts.microsecond)
            if ts < ts.replace(hour=9, minute=45, second=0, microsecond=0):
                return
            # print(self.env.env, self.__class__.__name__,self.symbol)
            # write your algo here
            study = Study(self.env.portfolio.stockHistory[self.symbol].resample('5T').first().fillna(method='ffill'))
            study.addEMA(20)
            studyHistory = study.getHistory()
            close = studyHistory['close'][-1]
            ema = studyHistory['EMA20'][-1]
            qty = 0
            if self.env.portfolio.stockPosition.get(self.symbol) is not None:
                qty = self.env.portfolio.stockPosition.get(self.symbol)
            if close >= ema:
                if qty <= 0:
                    buyqty = 1
                    if self.env.portfolio.buying_power > close * buyqty:
                        self.env.portfolio.stockOrdered[self.symbol] = True
                        self.env.portfolio.buy(self.symbol, buyqty, close, 'Algo2')
            if close <= ema:
                if qty > 0:
                    # avoid daytrade
                    if not (self.env.portfolio.stockFilledAt[self.symbol] is df.NaT or
                            self.env.portfolio.stockFilledAt[self.symbol]._date_repr == datetime.today().astimezone(
                                timezone('America/New_York')).strftime('%Y-%m-%d')):
                        self.env.portfolio.stockOrdered[self.symbol] = True
                        self.env.portfolio.sell(self.symbol, qty, close, 'Algo2')
        except Exception as e:
            pass


# names of these scanner subclass will appear in the dashboard scanner dropdown
# scanner stocks at high of the day
class HighOfTheDayScanner(Scanners):
    def __init__(self, env, min_share_price=10, max_share_price=100, min_last_dv=100000000,
                 tolerance=.99, volume=1000000):
        super().__init__()
        self.env = env
        self.milliseconds = 300  # required
        self.min_share_price = min_share_price
        self.max_share_price = max_share_price
        self.min_last_dv = min_last_dv
        self.tolerance = tolerance
        self.volume = volume

    async def loop(self):
        # print('high of day  starting')
        self.account = self.env.api.get_account()
        self.assets = self.env.api.list_assets()
        self.symbols = [asset.symbol for asset in self.assets if asset.tradable]
        self.selectedSymbols = []

        # self.sendMessage.emit("Starting high of the day Scanner")
        tickers = self.env.api.polygon.all_tickers()
        self.selectedTickers = [ticker for ticker in tickers if (
                ticker.ticker in self.symbols and
                ticker.lastTrade['p'] >= self.min_share_price and
                ticker.lastTrade['p'] <= self.max_share_price and
                ticker.day['v'] * ticker.lastTrade['p'] > self.min_last_dv and
                ticker.day['c'] > ticker.prevDay['c'] and
                ticker.day['c'] >= ticker.day['h'] * self.tolerance and
                ticker.day['v'] >= self.volume
        )]
        wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in
                 self.selectedTickers]
        self.wl_df = df.DataFrame.from_records(wlist)
        if not self.wl_df.empty:
            self.selectedSymbols = sorted(self.wl_df[0].tolist())
        if self.selectedSymbols is not None:
            channels = self.env.dataStream.getChannels(self.selectedSymbols)
            await self.env.dataStream.conn.subscribe(channels)
        # print('high of day  done')


# stocks with top count of close above ema
class TopEmaCountScanner(Scanners):
    def __init__(self, env, minutes=100, ema2=20, cnt2=10, atrlength=20, atr=1, min_share_price=10, max_share_price=100,
                 min_last_dv=100000000,
                 tolerance=.99, volume=1000000):
        super().__init__()
        self.milliseconds = 400  # required
        self.env = env
        self.ema2 = ema2
        self.cnt2 = cnt2
        self.atr = atr
        self.atrlength = atrlength

        self.min_share_price = min_share_price
        self.max_share_price = max_share_price
        self.min_last_dv = min_last_dv
        self.tolerance = tolerance
        self.volume = volume

    async def loop(self):
        # print('top20ema  starting')
        symbols = ["AMD", "WFC", "SQ", "ABBV", "FB", "AAPL", "GS", "C", "KO", "WDC", "NKE", "WMT", "TGT", "CSCO",
                   "COST", "ORLY", "LABD", "V", "MA", "BABA", "JD", "WB", "DIS", "LK", "TWLO", "CSX", "Z", "BIDU",
                   "MCD", "DVN", "ACB", "GILD", "QQQ", "SBUX", "WBA", "STX", "AMZN", "FAST", "NFLX", "CELG", "AMAT",
                   "UPS", "FDX", "DELL", "SNAP", "TWTR", "DWT", "JBLU", "AKS", "CMG", "DB", "GLD", "LABU", "DGAZ",
                   "UGAZ", "UAL", "NUGT", "WYNN", "NTES", "SIRI", "AMGN", "MAR", "ROST", "DLTR", "LULU", "AEM", "UWT",
                   "H", "FAS", "LVS", "JNPR", "PAYC", "CRON", "DD", "URI", "MOS", "CLDR", "WYND", "RH", "EA", "NOK",
                   "HTZ", "WSM", "SHOP", "OKTA", "IMMU", "CRWD", "JNUG", "IBM", "WORK", "ADI", "DHI", "KB", "TOL",
                   "PEP", "BYND", "CLX", "MU", "OSTK", "ROKU", "ULTA"]
        sdf = df.DataFrame(columns=['symbol', 'count'])
        sdf.set_index('symbol', inplace=True)

        for symbol in symbols:
            try:
                mHistory = self.api.polygon.historic_agg(size="minute", symbol=symbol, limit=minutes).df
                study = Study(mHistory)
                study.addEMA(ema2)
                studyHistory = study.getHistory()
                if studyHistory['close'][-1] > studyHistory['EMA' + str(ema2)][-1]:
                    count = studyHistory[studyHistory['close'] > studyHistory['EMA' + str(ema2)]].shape[0]
                    sdf.loc[symbol] = {'count': count}
                sdf.sort_values(by=['count'], inplace=True, ascending=False)
            except  Exception as e:
                pass  # print(e)
        self.symbols = sdf.head(cnt2).index.tolist()

        tickers = self.env.api.polygon.all_tickers()
        self.selectedTickers = [ticker for ticker in tickers if (
                ticker.ticker in self.symbols and
                ticker.lastTrade['p'] >= self.min_share_price and
                ticker.lastTrade['p'] <= self.max_share_price and
                ticker.day['v'] * ticker.lastTrade['p'] > self.min_last_dv and
                ticker.day['c'] > ticker.prevDay['c'] and
                ticker.day['v'] >= self.volume
        )]
        wlist = [[ticker.ticker, ticker.lastQuote['p'], ticker.lastTrade['p'], ticker.lastQuote['P']] for ticker in
                 self.selectedTickers]
        self.wl_df = df.DataFrame.from_records(wlist)
        if not self.wl_df.empty:
            self.selectedSymbols = sorted(self.wl_df[0].tolist())
        if self.selectedSymbols is not None:
            channels = self.env.dataStream.getChannels(self.selectedSymbols)
            await self.env.dataStream.conn.subscribe(channels)

        # print('top20ema  done')

# endregion
