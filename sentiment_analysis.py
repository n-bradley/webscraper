#%%
import pandas as pd 
import numpy as np 
from textblob import TextBlob
import matplotlib.pyplot as plt



data = pd.read_pickle('markets_master.pkl')

#%%
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['text'].apply(pol)
data['subjectivity'] = data['text'].apply(sub)
data

#%%

x = data['polarity']
y = data['subjectivity']
plt.scatter(x, y, color='b')
# plt.text(x+.001, y+.001, 'a', fontsize=10)
# plt.xlim(-.01, .12) 

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)


# %%
data.iloc[2]['text']