# charpter 9 LSTM
import numpy as np

from tools import pre_process_data, tokenize_and_vectorizer, collected_expected, pad_trunc
dataset = pre_process_data('./aclimdb/train')
vectorized_data = tokenize_and_vectorizer(dataset)
expected = collected_expected(dataset)
split_point = int(len(vectorized_data)*.8)

x_train = vectorized_data[:split_point]
y_train = vectorized_data[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_train = np.array(y_train)
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense, flatten, LSTM, Dropout
model = Sequential()
model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())
# Layer (type)            Output Shape                  Param 
# ================================================================= 
# lstm_1 (LSTM)           (None, 400, 50)                70200  （300+50+1)*50=17550  17550*4=70200 # 三个门sigmoid+一个数据tanh
# _________________________________________________________________ 
# dropout_1 (Dropout)     (None, 400, 50)                  0 
# _________________________________________________________________ 
# flatten_1 (Flatten)     (None, 20000)                    0 
# _________________________________________________________________ 
# dense_1 (Dense)          (None, 1)                     20001 
# ================================================================= 
# Total params: 90,201.0
# Trainable params: 90,201.0
# Non-trainable params: 0.0

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test))
# Train on 20000 samples, validate on 5000 samples
# Epoch 1/2
# 20000/20000 [==============================] - 548s - loss: 0.4772 -
# acc: 0.7736 - val_loss: 0.3694 - val_acc: 0.8412
# Epoch 2/2
# 20000/20000 [==============================] - 583s - loss: 0.3477 -
# acc: 0.8532 - val_loss: 0.3451 - val_acc: 0.8516
# <keras.callbacks.History at 0x145595fd0>

model_structure = model.to_json()
with open('lstm_model1.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('lstm_weights1.h5')

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