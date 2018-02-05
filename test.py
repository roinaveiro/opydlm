#!/usr/bin/env python
import pandas as pd
from opydlm import odlm, trend  

ts = [
 0.5429682543922109,
 0.5296058346035057,
 0.5403294585554494,
 0.542441925561093,
 0.5435209708555084,
 0.5430676782288945,
 0.5429877208796179,
 0.5429721282202071,
 0.5429690254184671,
 0.5449758960859548,
 0.5457612294317765,
 0.5434065016617284,
 0.5430519745276086,
 0.5436459000038072,
 0.5437794184525637]


## Version 1
model = odlm([]) + trend(degree=2, discount=0.95, name='trend1') + seasonality(7)
model.stableMode(False)

d = {}
for idx, el in enumerate(ts):
    print(el)
    model.append([el], component='main')
    model.fitForwardFilter()
    print()

mean, var = model.predictN(N=1, date=model.n-1)
d[idx] = mean

df1 = pd.DataFrame.from_dict(d, orient="index")

## Version 2
model = dlm([]) + trend(degree=2, discount=0.95, name='trend1') + seasonality(7)
model.stableMode(False)

d = {}
for idx, el in enumerate(ts):
    model.append([el], component='main')
    model.fitForwardFilter()

mean, var = model.predictN(N=1, date=model.n-1)
d[idx] = mean

df2 = pd.DataFrame.from_dict(d, orient="index")

## Vemos los resultados
print(df1)
print(df2)
