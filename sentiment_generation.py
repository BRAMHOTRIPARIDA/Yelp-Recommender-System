import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
import os
import sys
import math
import json
import time
import xgboost as xgb
import pickle
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sc = SparkContext.getOrCreate();
spark = SparkSession(sc)
sc.setLogLevel('WARN')


text_sentiment = SentimentIntensityAnalyzer()
def text_sentiment_func(txt):
    global text_sentiment
    if not isinstance(txt, str):
        return 0
    txt_sentiment = text_sentiment.polarity_scores(txt)
    return txt_sentiment["compound"]


tr_folder = '$ASNLIB/publicdata/'
tips_data = sc.textFile(os.path.join(tr_folder, 'tip.json')).map(lambda x: json.loads(x)).map(lambda x: ((x['business_id'], x['user_id']), text_sentiment_func(x['text']))).collectAsMap()
tips_data

with open('tips_dict.pickle', 'wb') as fl:
    pickle.dump(tips_data, fl)