# charpter 7 Getting words in order with convolutional neural networks (CNNs)
# charpter 7.1 Learning meaning  1.词序，2.词语之间的距离

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPool1D

import glob
import os
from random import shuffle


def pre_process_data(filepath):
    """
    导入数据
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []

    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))
    
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))
    
    shuffle(dataset)
    return dataset



from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data
word_vectors = get_data('w2v', limit = 200000)
# from gensim.models.keyedvectors import KeyedVectors 
# word_vectors = KeyedVectors.load_word2vec_format('/Users/liwei/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
    

def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])  # 分词
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass # 没有找到对应的词矢量
        vectorized_data.append(sample_vecs)
    return vectorized_data

def collect_expected(dataset):
    """
    Peel off the target values from the dataset
    """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

def pad_trunc(data, maxlen):
    """
    for a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


# preparing the data
dataset=pre_process_data('/Users/liwei/aclImdb/train')
# print(dataset[0])
# (1, 'I, as a teenager really enjoyed this movie! Mary Kate and Ashley worked 
# ➥ great together and everyone seemed so at ease. I thought the movie plot was 
# ➥ very good and hope everyone else enjoys it to! Be sure and rent it!! Also 
# they had some great soccer scenes for all those soccer players! :)')

vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data)*.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = vectorized_data[split_point:]

maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

# network
model = Sequential()
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(maxlen, embedding_dims)))

# pooling
model.add(GlobalMaxPool1D())

# Dropout : prevent overfitting
model.add(Dense(hidden_dims))  # Dense 表示全连接层
model.add(Dropout(0.2))
model.add(Activation('relu'))

# funnel
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv1d (Conv1D)              (None, 398, 250)          225250   300*3*250+250(bias)=225250
# _________________________________________________________________
# global_max_pooling1d (Global (None, 250)               0
# _________________________________________________________________
# dense (Dense)                (None, 250)               62750    250*250+250(bias)=62750
# _________________________________________________________________
# dropout (Dropout)            (None, 250)               0
# _________________________________________________________________
# activation (Activation)      (None, 250)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 251
# _________________________________________________________________
# activation_1 (Activation)    (None, 1)                 0
# =================================================================
# Total params: 288,251
# Trainable params: 288,251
# Non-trainable params: 0

# compile the CNN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # output layer for categorical variable (word)
# model.add(Dense(num_classes))
# model.add(Activation('sigmoid'))

# training
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
# Train on 20000 samples, validate on 5000 samples
# Epoch 1/2 [================================] - 417s - loss: 0.3756 -
# acc: 0.8248 - val_loss: 0.3531 - val_acc: 0.8390
# Epoch 2/2 [================================] - 330s - loss: 0.2409 -
# acc: 0.9018 - val_loss: 0.2767 - val_acc: 0.8840

# save your model
model_structure = model.to_json()
with open('cnn_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('cnn_weights.h5')

# test examples
sample_1 = "I hate that the dismal weather had me down for so long, hen will it break! Ugh, when does happiness return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend."
vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict(test_vec)
model.predict_classes(test_vec)