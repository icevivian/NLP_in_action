# charpter 3 -- 3.1 bag of words
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

def sample1():
    sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
    # 分词
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(sentence.lower())
    print(tokens)

    # 计算count  
    bag_of_words = Counter(tokens)
    print(bag_of_words)
    bag_of_words.most_common(4)   # 找到最常出现的词topN
    times_harry_appears = bag_of_words['harry']
    num_unique_words = len(bag_of_words)
    tf = times_harry_appears/num_unique_words  # 计算normalized term frequency
    print(round(tf, 4))
# sample1()

#===================================================================================
# charpter 3 -- 3.2 Vectorizing
# from nlpia.data.loader import kite_text
import nltk

def sample2():
    tokenizer = TreebankWordTokenizer()
    # tokens = tokenizer.tokenize(kite_text.lower())
    # token_counts =  Counter(tokens)
    # print(token_counts)
    # nltk.download('stopwords', quiet=True)
    # stopwords = nltk.corpus.stopwords.words('english')
    # tokens = [x for x in tokens if x not in stopwords]
    # kite_counts = Counter(tokens)

    
    # document_vector = []
    # doc_length = len(tokens)
    # for key, value in kite_counts.most_common():
    #     document_vector.append(value/doc_length)
    # print(document_vector)


    docs = ["The faster Harry got to the store, the faster Harry, the faster, would get home."]
    docs.append("Harry is hairy and faster than Jill.")
    docs.append("Jill is not as hairy as Harry.")
    # 根据句子构造词典
    doc_tokens = []
    for doc in docs:
        doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
    all_doc_tokens = sum(doc_tokens, [])
    lexion = sorted(set(all_doc_tokens))
    print(lexion)

    from collections import OrderedDict
    zero_vector = OrderedDict((token, 0) for token in lexion)
    print(zero_vector)
    # 每个句子构造矢量
    import copy
    doc_vectors = []
    for doc in docs:
        vec = copy.copy(zero_vector)
        tokens = tokenizer.tokenize(doc.lower())
        token_counts = Counter(tokens)
        for key, value in token_counts.items():
            vec[key] = value/len(lexion)
        doc_vectors.append(vec)
    print(doc_vectors[0])

# sample2()

'''
句子之间的距离可以通过计算两个矢量之间的欧式距离/cosine相似度来衡量。
'''
import math
def cosine_sim(vec1, vec2):
    '''
    计算cosine相似度。若为1说明两个矢量方向重合，若为0说明两个矢量完全正交
    '''
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i,v in enumerate(vec1):
        dot_prod += v.vec2[i]
    
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod/(mag_1*mag_2)

#===================================================================================
# charpter 3 -- 3.3 Zipf's Law
def sample3():
    nltk.download('brown')
    from nltk.corpus import brown
    print(brown.words()[:10])
    print(brown.tagged_words()[:5])
    puncs = set((',','.','--','-','!','?',':',';','``',"''",'(',')','[',']'))
    word_list = (x.lower() for x in brown.words() if x not in puncs)
    token_counts = Counter(word_list)
    print(token_counts.most_common(20))
    '''
    可以看出，最常出现的词都是无意义词，例如the, and, of。
    '''

# sample3()

#===================================================================================
# charpter 3 -- 3.4 Topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
def sample4():
    docs = ["The faster Harry got to the store, the faster Harry, the faster, would get home."]
    docs.append("Harry is hairy and faster than Jill.")
    docs.append("Jill is not as hairy as Harry.")
    corpus = docs
    vectorizer = TfidfVectorizer(min_df=1)
    model = vectorizer.fit_transform(corpus)
    print(model.todense().round(2))

sample4()
    