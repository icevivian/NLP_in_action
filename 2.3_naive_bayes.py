from nlpia.data.loader import get_data
movies = get_data('hutto_movies')
print(movies.head())
# 分词
import pandas as pd
pd.set_option('display.width', 75)
from nltk.tokenize import casual_tokenizer
bags_of_words = []
from collections import Counter
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenizer(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
# 情感分析模型训练
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiments>0)   # 分类模型
movies['predicted_sentiment'] = nb.predict_proba(df_bows)*8-4 #用于回归预测
movies['error'] = (movies.predicted_sentiment-movies.sentiment).abs()
print(movies.error.mean().round(1)) # 预测平均误差
movies['sentiment_ispositive'] = (movies.sentiment >0).astype(int)
movies['predicted_ispositive'] = (movies.predicted_sentiment>0).astype(int)
print (movies['sentiment predicted_sentiment sentiment_ispositive predicted_ispositive'.split()].head()) #预测具体对比
print (movies.predicted_ispositive == movies.sentiment_ispositive).sum()/len(movies)  #总体准确率