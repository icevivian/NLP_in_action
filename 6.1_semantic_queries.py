# charpter 6 Reasoning with word vectors (Word2vec)
# charpter 6 -- 6.1 semantic queries and analogies 语义查询和类比
# charpter 6 -- 6.2 word vectors

def sample1():
    '''
    使用google已经训练好的word2vec：先下载文件到本地，再导入
    '''
    from gensim.models.keyedvectors import KeyedVectors 
    word_vectors = KeyedVectors.load_word2vec_format('/Users/liwei/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
    print("相似")
    print(word_vectors.most_similar(positive=['cooking','potatoes'], topn=5))
    # word_vectors.most_similar(positive=['germany', 'france'], topn=1)  # [('europe', 0.7222039699554443)]
    print("找出列表中找出最不相关的")
    print(word_vectors.doesnt_match("potatoes milk cake computer".split()))  # 'computer'
    print("类比")
    print(word_vectors.most_similar(positive=['king','woman'], negative=['man'], topn=2)) # [('queen', 0.7118192315101624), ('monarch', 0.6189674139022827)]
    print("计算相似度")
    print(word_vectors.similarity('princess', 'queen'))  # 0.707
    print("直接查看词矢量")
    print(word_vectors['phone']) # google输出为1*300的矢量
sample1()

def sample2():
    '''
    自己训练一个word2vec模型
    '''
    from gensim.models.word2vec import Word2Vec
    num_features = 300 
    min_word_count = 3
    num_workers = 2
    window_size = 6
    subsampling = 1e-3
    model = Word2Vec(token_list, workers=num_workers, size=num_features, min_count=min_word_count, window=window_size, sample=subsampling)
    model.init_sims(replace=True) # freeze the model, discarding the output weights. 这种方式丢弃了模型结构，只保存词向量
    model_name = "my_domain_specific_word2vec_model"
    model.save(model_name)

def sample3():
    '''
    使用fastText训练好的词向量
    '''
    from gensim.models.fasttext import FastText
    ft_model = FastText.load_fasttext_format(model_file = model_path)
    ft_model.most_similar('soccer')