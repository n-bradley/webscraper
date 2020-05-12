#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle



#%%
def homepage_soup():
    r1 =  requests.get("https://www.marketwatch.com/")
    homepage = r1.content
    return BeautifulSoup(homepage, 'html5lib')

#%%

def get_article_links():
    coverpage_articles =homepage_soup().find_all('div', class_='article__content')
    articles = {}

    #get headline
    headline = coverpage_articles[1].find('a')
    headline_text = headline.get_text().strip()
    headline['href']
    articles[headline_text] = headline['href']

    secondary = coverpage_articles[1].find_all('li') 

    for item in secondary:
        art = item.find('a')
        title = art.get_text().strip()
        title = title.replace('\n','').replace('  ','')
        hl = art['href']
        articles[title] = hl
    
    return articles

articles = get_article_links()


#%%
links = list(articles.values())

def get_article_soup(link):
    req = requests.get(link)
    article_page = req.content
    return BeautifulSoup(article_page, 'html5lib')

def clean_article_text(text):
    cleantext = ''
    for t in text:
        t = t.replace('  ','').replace('\n',' ')
        cleantext += (t+'\n')
    cleantext = cleantext
    return cleantext

def get_article_text(link):
    content = get_article_soup(link).find('div', class_='column column--full article__content' )
    
    article_text = []

    for para in content.find_all('p'):
        article_text.append(para.get_text())
    
    article_text = clean_article_text(article_text)

    return article_text

def get_article_stocks(link):
    content = get_article_soup(link).find_all('span', class_='symbol')

    article_stock_links = []
    symbols = [x.get_text() for x in content]

    return symbols
    
def create_article_table(art_dict):
    art_dict = {'titles':list(art_dict.keys()),
            'links':list(art_dict.values())}
    art_dict['text'] = [get_article_text(link) for link in art_dict['links']]
    art_dict['stocks'] = [get_article_stocks(link) for link in art_dict['links']]
    table = pd.DataFrame.from_dict(art_dict)
    return table

df = create_article_table(articles)

df.to_pickle('marketwatch_headlines.pkl')
#%%
df = pd.read_pickle('marketwatch_headlines.pkl')

#%%

import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    chars = ["”","“", "—",  "‘"]
    for char in chars:
        text = text.replace(char," ")
    text = text.replace("’","") 
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\s+',' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df['clean_text'] = df.text.apply(clean_text_round1)

#%%

data_clean = df[['clean_text']].copy()


#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.clean_text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index


#%%

data=data_dtm.transpose()


#%%

top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict

#%%

for article_no, top_words in top_dict.items():
    print(article_no)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')

#%%

# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 30 words for each comedian
words = []
for article in data.columns:
    top = [word for (word, count) in top_dict[article]]
    for t in top:
        words.append(t)
        
words

#%%
Counter(words).most_common()
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
add_stop_words

#%%
# Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer


# # Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.clean_text)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

#%%


# Let's make some word clouds!
# Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud
from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [168, 6]


# Create subplots for each comedian
for index, article in enumerate(data.columns):
    wc.generate(data_clean.clean_text[article])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(index)
    
plt.show()