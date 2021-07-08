# charpter 4 -- 4.1 from word counts to topic scores
import numpy as np
import pandas as pd

def noname():
    topic = {}
    tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))
    print(tfidf)
    topic['petness'] = (0.3*tfidf['cat']+0.3*tfidf['dog']+0*tfidf['apple']+0*tfidf['lion']-0.2*tfidf['NYC']+0.2*tfidf['love'])
    topic['animalness'] = (0.1*tfidf['cat']+0.1*tfidf['dog']+0.1*tfidf['apple']+.5*tfidf['lion']+.1*tfidf['NYC']+.1*tfidf['love'])
    topic['cityness'] = (0*tfidf['cat']+0.1*tfidf['dog']+0.2*tfidf['apple']+0.1*tfidf['lion']+0.5*tfidf['NYC']+0.1*tfidf['love'])
    print(topic)
    # 思考如何让机器理解给不同的词赋予不同的权重：例如petness主题中，给cat,dog较高的权重0.3，而给apple,lion权重0，给NYC负的权重。
    word_vector = {}
    word_vector['cat'] = .3*topic['petness']+.1*topic['animalness']+0*topic['cityness']
    word_vector['dog'] = .3*topic['petness']+.1*topic['animalness']+.1*topic['cityness']


def sample1():
    '''
    LDA
    垃圾邮件分类示例：获取非垃圾类质心到垃圾类质心的距离，距离夹角判别分类。并不依赖于独立的词
    '''
    from nlpia.data.loaders import get_data
    pd.options.display.width = 120
    sms = get_data('sms-spam')
    index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
    sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
    #        spam text
    # sms0     0  Go until jurong point, crazy.. Available only ...
    # sms1     0                      Ok lar... Joking wif u oni...
    # sms2!    1  Free entry in 2 a wkly comp to win FA Cup fina...
    # sms3     0  U dun say so early hor... U c already then say...
    # sms4     0  Nah I don't think he goes to usf, he lives aro...
    # sms5!    1  FreeMsg Hey there darling it's been 3 week's n...
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize.casual import casual_tokenize
    tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
    tfidf_docs = tfidf_model.fit_transform(raw_documents = sms.text).toarray()
    mask = sms.spam.astype(bool).values  # trick：使用mask进行数据提取
    spam_centroid = tfidf_docs[mask].mean(axis=0)
    ham_centroid = tfidf_docs[~mask].mean(axis=0)
    spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid) #dot是矩阵相乘
    from sklearn.preprocessing import MinMaxScaler
    sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
    sms['lda_predict'] = (sms.lda_score>0.5).astype(int)
    print(sms['spam lda_predict lda_score'.split()].round(2).head(6)) # trick:split获取columns
    from pugnlp.stats import confusion
    print(confusion(sms['spam lda_predict'.split()]))


# charpter 4 -- 4.2 Latent semantic analysis
def sample2():
    '''
    LSA 隐语义分析，它的本质是SVD分解，能够抓住数据的本质并且忽略噪声，当SVD分解用于文本处理中时，这种方式叫做LSA，也叫截断式SVD
    当机器看到几个词经常同时出现时，例如“cat","dog"，就会把它们归为一个主题
    通过上下文，机器可以推断出这个词的含义：“Mad Libs games”【因为机器仅通过已知的词含义就能推断出这个句子的隐含语义？】
    '''
    from nlpia.book.examples.ch04_catdog_lsa_3x6x16 import word_topic_vectors
    

# charpter 4 -- 4.3 Singular value decomposition
def sample3():
    '''
    SVD分解
    '''
    # from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm
    # bow_svd, tfidf_svd = lsa_models()
    # prettify_tdm(**bow_svd)
    # tdm = bow_svd['tdm']  # 生成词-文档矩阵，维度6*10
    import numpy as np
    import pandas as pd
    tdm = pd.DataFrame([[0,0,0,0,0,0,1,1,1,0,1],[0,0,0,0,0,0,0,0,0,0,1],[1,1,0,1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,0,0,0],[1,1,1,1,1,0,0,0,0,1,0],[0,0,1,0,0,0,0,0,1,1,0]], index='cat,dog,apple,lion,nyc,love'.split(','))
    print(tdm) 
    U, s, Vt = np.linalg.svd(tdm)  
    # print(pd.DataFrame(U, index = tdm.index).round(2))  # 得到U矩阵，维度6*6，含义就是将词向量转换为主题向量（没有进行截断时为6个主题）
    S = np.zeros((len(U), len(Vt)))
    pd.np.fill_diagonal(S, s)
    # print(pd.DataFrame(S).round(1)) # 得到S矩阵，维度6*10， 主题-主题向量，表示了每个主题的重要度
    # print(pd.DataFrame(Vt).round(2)) # Vt矩阵，维度10*10， 主题-文档向量

    # 当截取n维主题时，查看对应的error
    err = []
    for numdim in range(len(s), 0, -1):
        S[numdim-1, numdim-1] = 0
        reconstructed_tdm = U.dot(S).dot(Vt)
        print(numdim,'维')
        print(reconstructed_tdm)
        err.append(np.sqrt( ( (reconstructed_tdm-tdm).values.flatten()**2 ).sum()/ np.product(tdm.shape) ))
    print(np.array(err).round(2))   #删除的主题数越多，得到的误差就越大

sample3()

# charpter 4 -- 4.4 Principal component analysis
def sample4():
    '''
    PCA是使用SVD做降维处理的方法名称
    '''
    pd.set_option('display.max_columns', 6)
    from sklearn.decomposition import PCA
    import seaborn
    from matplotlib import pyplot as plt
    from nlpia.data.loaders import get_data
    df = get_data('pointcloud').sample(1000)
    pca = PCA(n_components=2)      # 设置降维后的维度
    df2d = pd.DataFrame(pca.fit_transform(df), columns = list('xy'))
    df2d.plot(kind = 'scatter', x='x', y='y')
    plt.show()

    '''
    以下实验数据和4.1相同，都是垃圾邮件分类数据
    '''
    pd.options.display.width= 120
    sms = get_data('sms-spam')
    index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
    sms.index = index
    sms.head(6)
    #        spam text
    # sms0     0  Go until jurong point, crazy.. Available only ...
    # sms1     0                      Ok lar... Joking wif u oni...
    # sms2!    1  Free entry in 2 a wkly comp to win FA Cup fina...
    # sms3     0  U dun say so early hor... U c already then say...
    # sms4     0  Nah I don't think he goes to usf, he lives aro...
    # sms5!    1  FreeMsg Hey there darling it's been 3 week's n...
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize.casual import casual_tokenize
    tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
    tfidf_docs = tfidf_model.fit_transform(raw_documents = sms.text).toarray()
    len(tfidf_model.vocabulary_) #9232
    tfidf_docs = pd.DataFrame(tfidf_docs)
    tfidf_docs = tfidf_docs - tfidf_docs.mean()  # 减去均值
    tfidf_docs.shape # 4837, 9232
    sms.spam.sum()   # 638 [4837个文件中638个垃圾邮件，文件词汇量为9232。由此可以看出数据的两个特点：1.正负样本不均，2.词汇量大于文本量。这样直接去训练的话会导致严重的过拟合]
    # 做PCA降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=16)
    pca = pca.fit(tfidf_docs)
    pca_topic_vectors = pca.transform(tfidf_docs)
    columns = ['topic{}'.format(i) for i in range(pca.n_components)]
    pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
    pca_topic_vectors.round(3).head(6)
    # 做truncated-SVD降维
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=16, n_iter=100)
    svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
    svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns, index=index)
    svd_topic_vectors.round(3).head(6)
    # LSA 
    import numpy as np
    svd_topic_vectors = (svd_topic_vectors.T/np.linalg.norm(svd_topic_vectors,axis=1)).T
    svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1)
# sample4()

# charpter 4 -- 4.5 Latent Dirichlet allocation
def sample5():
    '''
    LDiA 假设词频满足Dirichlet分布，假设每篇文档都由任意数量的主题混合而成，该数量是在开始训练模型时选择的。
    1.计算平均文档词个数 2.确定主题个数K
    '''
    # BOW矢量
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.tokenize import casual_tokenize
    np.random.seed(42)
    counter = CountVectorizer(tokenizer=casual_tokenize)
    bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)
    columns_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(), counter.vocabulary_.keys())))
    bow_docs.columns = terms

    # LDiA分解
    from sklearn.decomposition import LatentDirichletAllocation as LDiA
    ldia  = LDiA(n_components=16, learning_method='batch')
    ldia = ldia.fit(bow_docs)
    ldia.components_.shape    # (16, 9232)   模型将9232个词转换为16个主题

    pd.set_option('display.width', 75)
    components = pd.DataFrame(ldia.components_.T, index = terms, columns = columns)
    components.round(2).head(3)
#         topic0  topic1  topic2   ...     topic13  topic14  topic15
# !      184.03   15.00   72.22   ...      297.29    41.16    11.70
# "        0.68    4.22    2.41   ...       62.72    12.27     0.06
# #        0.06    0.06    0.06   ...        4.05     0.06     0.06   
    ldia16_topic_vectors = ldia.transform(bow_docs)
    ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index = index, columns = columns)
    ldia16_top_vectors.round(2).head()
#            topic0  topic1  topic2   ...     topic13  topic14  topic15
# sms0     0.00    0.62    0.00   ...        0.00     0.00     0.00
# sms1     0.01    0.01    0.01   ...        0.01     0.01     0.01
# sms2!    0.00    0.00    0.00   ...        0.00     0.00     0.00
# sms3     0.00    0.00    0.00   ...        0.00     0.00     0.00
# sms4     0.39    0.00    0.33   ...        0.00     0.00     0.00
# 可以看出矩阵是稀疏的，零值很多

    # LDiA+LDA=spam classfier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size = 0.5, random_state=271828)
    lda = LDA(n_components=1)
    lda = lda.fit(X_train, y_train)
    sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
    round(float(lda.score(X_test, y_test)), 2)
