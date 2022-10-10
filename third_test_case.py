# Imports
import numpy as np
import pandas as pd
import pandas_ta as ta
from cmath import asin
from math import ceil
import yfinance as yf

#                                                Special Functions
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ----------------Functions which are unique to this file/logic------------------//

# @function complex number ceil
# @param complex_num -- > example is 1.0000j
# @returns the ceil of complex

def complex_number_ceil(complex_num):
    complex_tostring = str(complex_num)

    #find the decimal point

    index = complex_tostring.index('.')

    #get the number with 1 decimal place
    complex_tostring = complex_tostring[:index+2]

    #convert to usable float
    complex_tofloat = float(complex_tostring)

    #get the ceil of the float
    complex_ceil = ceil(complex_tofloat)


    #return the ceil value
    return complex_ceil

# @function series valuewhen
# @param condition -- > most times it is a boolean value e.g np.where(pd['close]>pd['open],True,False)
# @param series -- > take in pandas sereis
# @param occurence -- > the index from the series to be returned must be type int
# @returns the series[occurence] when the condition is true or sereis[occurence+1] when the condition is false
# Might have to make this function a procedure in discriminator to avoid keyh errors

def series_valuewhen(condition,series,occurence):
    #set value
    df = series.copy()

    # test condition
    if condition == True:
        df['value'] = series.shift(occurence+1)
    else:
        df['value'] = series.shift(occurence+1)

    #return output value
    return df['value']


#                                                   discriminator function
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ----------------Homodyne Discriminator w/ Hilbert DC------------------//
# CycPart=0.5
# @function Homodyne Discriminator w/ Hilbert Dominant Cycle
# @function With Hilbert transformer, the third algorithm for computing the dominant cycle is the homodyne approach. Homodyne means the signal is multiplied by itself. More precisely, we want to multiply the signal of the current bar with the complex value of the signal one bar ago. The complex conjugate is, by definition, a complex number whose sign of the imaginary component has been reversed.
# @param  Price --> price series. Default --> value of range of close and open
# @param  CycPart --> Cycle Part . Default --> 0.5
# @returns DomCycle series --> DomCycle output series

def discriminator(dataframe,cycpart=0.5):
    #duplicate dataframe for manipulation
    df = dataframe.copy()
    #Declare initial state of variables
    pi = 2 * asin(1)
    df['smooth'] = 0
    df['detrender'] = 0
    df['i1'] = 0
    df['q1'] = 0
    df['ji'] = 0
    df['jq'] = 0
    df['i2'] = 0
    df['q2'] = 0
    df['re'] = 0
    df['im'] = 0
    df['period'] = 0
    df['smoothperiod'] = 0
    df['domcycle'] = 0

    #Start Hilbert transform and filtering
    df['smooth'] = (4 * df['Close'] + 3 * df['Close'].shift(1)  + 2 * df['Close'].shift(2)  + df['Close'].shift(3) ) / 10 
    df['detrender'] = (.0962 * df['smooth'] + .5769 * df['smooth'].shift(2) - .5769 * df['smooth'].shift(4) - .0962 * df['smooth'].shift(6)) * (.075 * df['period'].shift(1) + .54) 
    df['q1'] = (.0962 * df['detrender'] + .5769 * df['detrender'].shift(2) - .5769 * df['detrender'].shift(4) - .0962 * df['detrender'].shift(6)) * (.075 * df['period'].shift(1) + .54)
    df['i1'] = df['detrender'].shift(3)

    #Advance the phase of i1 and q1 by 90 degrees
    df['ji'] = (.0962 * df['i1'] + .5769 * df['i1'].shift(2) - .5769 * df['i1'].shift(4)  - .0962 * df['i1'].shift(6)) * (.075 * df['period'].shift(1) + .54) 
    df['jq'] = (.0962 * df['q1'] + .5769 * df['q1'].shift(2) - .5769 * df['q1'].shift(4)  - .0962 * df['q1'].shift(6)) * (.075 * df['period'].shift(1) + .54) 

    #Phasor addition for 3 bar averaging
    df['i2']  = df['i1'] - df['jq']
    df['q2'] = df['q1'] + df['ji']

    #Smooth the I and Q components before applying the discriminator
    df['i2'] =  .2 * df['i2']  + .8 * df['i2'].shift(2) 
    df['q2'] = .2 * df['q2']+ .8 * df['q2'].shift(2) 

    #Homodyne Discriminator
    df['re'] = df['i2'] * df['i2'].shift(1) + df['q2'] * df['q2'].shift(1)
    df['im'] = df['i2'] * df['q2'].shift(1) - df['q2'] * df['i2'].shift(1)
    df['re'] = .2 * df['re'] + .8 * df['re'].shift(1)
    df['im'] = .2 * df['im']+ .8 * df['im'].shift(1) 

    #Filter with time period
    #df['period'] =  2 * pi / atan(df['im'] / df['re'])  if  (df['im'] != 0) & (df['re'] != 0)  else  df['period']
    # df['period'] =  1.5 * df['period'].shift(1)  if  df['period'] > 1.5 * df['period'].shift(1)  else  df['period']
    # df['period'] =  .67 * df['period'].shift(1)  if  df['period'] < .67 * df['period'].shift(1)  else  df['period']
    df['period'] = np.where(df['im'].gt(0), 2 * pi / np.arctan(df['im'] / df['re'])  , np.where(df['im'].lt(0), -2 * pi / np.arctan(df['im'] / df['re'])  , df['period']))
    df['period'] = np.where(df['re'].gt(0), 2 * pi / np.arctan(df['im'] / df['re'])  , np.where(df['re'].lt(0), -2 * pi / np.arctan(df['im'] / df['re'])  , df['period']))    
    df['period']  = np.where(df['period'].gt(1.5 * df['period'].shift(1)), 1.5 * df['period'].shift(1), df['period'])
    df['period']  = np.where(df['period'].lt(.67 * df['period'].shift(1)), .67 * df['period'].shift(1), df['period'])

    #Limit Period to be within the bounds of 6 bar and 50 bar cycles
    # df['period'] =  6 if df['period'] < 6  else df['period']
    # df['period'] =  50 if df['period'] > 50  else df['period']
    df['period'] = np.where(df['period'].lt(6),6,df['period'])
    df['period'] = np.where(df['period'].gt(50),50,df['period'])
    df['period'] = .2 * df['period'] + .8 * df['period'].shift(1)
    df['smoothperiod'] = .33 * df['period'] + .67 * df['smoothperiod'].shift(1)

    #Add final filter to Period here to keep values within processing range
    #Find equivalent of gt() and lt() in numpy
    df['smoothperiod'] = df['smoothperiod'] * cycpart
    #The value gotten is complex128 making it impossible to apply the np.ceil function
    #I am forcing the ceil method on its head awon werey
    df['smoothperiod'] = df['smoothperiod'].astype(float)
    df['smoothperiod'] = df['smoothperiod'].apply(np.ceil)
    #df['domcycle'] = 34 if np.ceil(cycpart * df['smoothperiod']) > 34 else 1 if np.ceil(cycpart * df['smoothperiod']) < 1 else np.ceil(cycpart * df['smoothperiod'])
    df['domcycle'] = np.where(df['smoothperiod'].gt(34),34, np.where(df['smoothperiod'].lt(1),1,df['smoothperiod']))
  
    #Return discrminated signal
    return df['domcycle']



#                                                       Trade Actions
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ----------------Logic for Buys/Sells/Martingales------------------//
#Set a position size
#Set a martingale size

#Download GBPUSD data 'GBPUSD=X'
price_data = yf.download(tickers ='GBPUSD=X'  ,period ='5d', interval = '15m')

# Apply ceil the discriminator values so I have stuff like 1.00 and then 2.00 not 1.4 and then 1.5... That is bad
# If there is a ceil 1.4 == 1.0 and 1.5 = 2.0 and there is peace
#Apply ceil to the discriminated data *** I applied it in the discriminator function

discriminated_signal = discriminator(price_data,0.1)
print(f"{price_data['Close']}Price below  {discriminated_signal} and Signal below")

'''
#---------------------------------------------------------Keep Track of Trades
call_traded = False # Keep track if we've made a call trade
put_traded = False # Keep track if we've made a put trade
reports = [] # Keep track of wins and losses

#---------------------------------------------------------Trade Logic
candle_bull_one_step_back = np.where(price_data['Close'].shift(1).gt(price_data['Open'].shift(1)),1,0)
candle_bear_one_step_back  = np.where(price_data['Close'].shift(1).lt(price_data['Open'].shift(1)),1,0)
indicator_signal = np.where(discriminated_signal.gt(discriminated_signal.shift(1)),1,np.where(discriminated_signal.lt(discriminated_signal.shift(1))),1,0)
indicator_signal_one_step_back  = np.where(discriminated_signal.shift(1).gt(discriminated_signal.shift(2)),1,np.where(discriminated_signal.shift(1).lt(discriminated_signal.shift(2))),1,0)

# ----------------------------------------------------------Call Logic
if candle_bull_one_step_back == 1 and indicator_signal == 1:
    print('CALL') #Call trade with position size
    call_traded = True

# Check if call trade was win or loss
if call_traded == True and indicator_signal_one_step_back == 1:
    if candle_bull_one_step_back == 1:
        reports.insert(0,'win')
        call_traded = False
    if candle_bull_one_step_back == 0:
        reports.insert(0,'loss')
        call_traded = False
        print('Martingale PUT')#Martingale code here
        #Put trade with the position size * martingale size if there is no trade signal that is


# ----------------------------------------------------------Put Logic
if candle_bull_one_step_back == 1 and indicator_signal == 1:
    print('PUt') #Put trade with position size
    put_traded = True

# Check if put trade was win or loss
if put_traded == True and indicator_signal_one_step_back == 1:
    if candle_bear_one_step_back == 1:
        reports.insert(0,'win')
        put_traded = False
    if candle_bear_one_step_back == 0:
        reports.insert(0,'loss')
        put_traded = False
        print('Martingale CALL')#Martingale code here
        #Call trade with the position size * martingale size if there is no trade signal that is
'''
