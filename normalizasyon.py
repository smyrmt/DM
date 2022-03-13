import numpy as np
import pandas as pd
import sklearn.metrics as mt
from sklearn import preprocessing
#import dataset
df=pd.read_csv("student-mat.csv")
print(df.shape)
df.head()
#normalizasyon için verileri kopyalayalım 1. ve 2. dönem notları için normalizasyon
df_norm=df.copy()
df_norm[['G1','G2']]=preprocessing.minmax_scale(df[['G1','G2']])
#sonuçları görelim
print(df_norm[['G1','G2']].values)
