#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:04:16 2021

@author: jinyang
"""

import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date

%matplotlib inline


start = '2020-01-01'
end = date.today()

# using DataReader
cba = web.DataReader("CBA.AX", 'yahoo', start, end)

#using yahoo finance
data = yf.download("SPY AAPL", start=start, end=end)
pfe = yf.Ticker('PFE')
old = pfe.history(start=start, end=end)

## comparing the big four banks
cba = web.DataReader("CBA.AX", 'yahoo', start, end)
wbc = web.DataReader("WBC.AX", 'yahoo', start, end)
anz = web.DataReader("ANZ.AX", 'yahoo', start, end)
nab = web.DataReader("NAB.AX", 'yahoo', start, end)

cba['Open'].plot(label='CBA Open')
cba['Close'].plot(label='CBA Close')
plt.legend()
plt.show()

# comparing banks






