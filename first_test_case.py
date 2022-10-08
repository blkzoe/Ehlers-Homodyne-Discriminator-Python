from cmath import asin,atan
from math import ceil
from os import read

#from math import pi
import numpy as np
import pandas as pd
import pandas_ta as ta



# @function valuewhen
# @function replicate tradingview's valuewhen function
# @param  condition --> bool
# @param  source --> float
# @param  occurence --> int
# @returns source[occurence] series --> valwhen output series

def valuewhen(condition,source,occurence):#Might replace np.where()
    source_list = [0,0,0,0,0,0]
    source_list.append(source)
    source_list = source_list[::-1]
    value = source_list[occurence] if condition is True else source_list[occurence+1]
    return value


# Homodyne Discriminator w/ Hilbert Dominant Cycle
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
    df['smooth'] = (4 * df['close'] + 3 * df['close'].shift(1)  + 2 * df['close'].shift(2)  + df['close'].shift(3) ) / 10 
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
    df['period'] =  2 * pi / atan(df['im'] / df['re'])  if  df['im'] != 0 and df['re'] != 0  else  df['period']
    df['period'] =  1.5 * df['period'].shift(1)  if  df['period'] > 1.5 * df['period'].shift(1)  else  df['period']
    df['period'] =  .67 * df['period'].shift(1)  if  df['period'] < .67 * df['period'].shift(1)  else  df['period']

    #Limit Period to be within the bounds of 6 bar and 50 bar cycles
    df['period'] =  6 if df['period'] < 6  else df['period']
    df['period'] =  50 if df['period'] > 50  else df['period']
    df['period'] = .2 * df['period'] + .8 * df['period'].shift(1)
    df['smoothperiod'] = .33 * df['period'] + .67 * df['smoothperiod'].shift(1)

    #Add final filter to Period here to keep values within processing range
    df['domcycle'] = 34 if ceil(cycpart * df['smoothperiod']) > 34 else 1 if ceil(cycpart * df['smoothperiod']) < 1 else ceil(cycpart * df['smoothperiod'])
    return df['domcycle']



#Check for key errors when getting price data
#Not sure if I should use shift() or iloc[]
