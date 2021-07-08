# charpter 8 Loopy(recurrent) neural networks (RNNs)
# charpter 8.1 Remembering with recurrent networks
import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data
word_vectors = get_data('wv')

# Load and prepare your data
dataset = pre_process_data('/Users/liwei/aclImdb/train')
vectorized_data = tokenize_and_vectorizer(dataset)
expected = collected_expected(dataset)
split_point = int(len(vectorized_data)*.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = vectorized_data[split_point:]

maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

import numpy as np
x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN
num_neurons = 50
model = Sequential()
model.add(SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims))) # return_sequences = True表示每次隐藏神经元的输出都保存下来，因为有400次输入，所以输出维度为（400，50）

model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
model.summary()
# Layer (type)            Output Shape          Param # 
# ================================================================= 
# simple_rnn_1(SimpleRNN) (None, 400, 50)        17550     (300+50)*50+50(bias)=17550
# _________________________________________________________________ 
# dropout_1 (Dropout)     (None, 400, 50)         0 
# _________________________________________________________________ 
# flatten_1 (Flatten)     (None, 20000)           0 
# _________________________________________________________________ 
# dense_1 (Dense)          (None, 1)             20001 
# ================================================================= 
# Total params: 37,551.0
# Trainable params: 37,551.0
# Non-trainable params: 0.0
# __________________________

model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, validation_data=(x_test, y_test))
# Train on 20000 samples, validate on 5000 samples
# Epoch 1/2
# 20000/20000 [==============================] - 287s - loss: 0.9063 -
# acc: 0.6529 - val_loss: 0.5445 - val_acc: 0.7486
# Epoch 2/2
# 20000/20000 [==============================] - 240s - loss: 0.4760 -
# acc: 0.7951 - val_loss: 0.5165 - val_acc: 0.7824
model_structure = model.to_json()
with open("simplernn_model1.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights('simplernn_weights.h5')

# predict
sample_1 = "I hate that the dismal weather had me down for so long, hen will it break! Ugh, when does happiness return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend."
from keras.model import model_from_json
with open('samplernn_model1.json', 'r') as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights('simplernn_weights.h5')
vec_list = tokenize_and_vectorizer([(1,sample_1)])
test_vec_list = pad_trunc(vec_list)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict_classes(test_vec)