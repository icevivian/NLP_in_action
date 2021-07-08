# charpter 5 -- 5.1 Neural networks, the ingredient list

def sample1():
    '''
    OR problem setup
    '''
    sample_data = [[0,0],[0,1],[1,0],[1,1]]
    expected_results = [0, 1, 1, 1]
    activation_threshold = 0.5

    from random import random
    import numpy as np
    weights = np.random.random(2)/1000    #[8.24692822e-05 9.23576854e-04]
    bias_weight = np.random.random()/1000

    for iteration_num in range(5):
        correct_answers = 0
        for idx, sample in enumerate(sample_data):
            input_vector = np.array(sample)
            activation_level  = np.dot(input_vector, weights)+(bias_weight*1)
            # print(activation_level)
            if activation_level > activation_threshold:
                perceptron_output = 1
            else:
                perceptron_output = 0
            if perceptron_output == expected_results[idx]:
                correct_answers += 1
            new_weights = []
            for i, x in enumerate(sample):
                new_weights.append(weights[i]+(expected_results[idx]-perceptron_output)*x)  # 输入x越大，对weight的影响越大，反之
            bias_weight = bias_weight + (expected_results[idx]-perceptron_output)*1         # 实际上是损失函数为交叉熵损失函数时得到的梯度下降公式
            weights = np.array(new_weights)
            # print(weights)
            # print(bias_weight)
        print('{} correct answers out of 4, for iteration {}'.format(correct_answers, iteration_num))
# sample1()

def sample2():
    '''
    XOR Keras network
    '''
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import SGD
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([[0],[1],[1],[0]])
    model = Sequential()
    num_neurons = 10
    model.add(Dense(num_neurons, input_dim=2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    sgd = SGD(lr=0.1) # 随机梯度下降 lr:learning rate
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.predict(X_train) # 初始化参数给出的结果，必定是不准确的
    model.fit(X_train, y_train, epochs=100)
    model.predict_classes(X_train)

    # 存储模型
    import h5py
    model_structure = model.to_json()
    with open("basic_model.json", "w") as json_file:
        json_file.write(model_structure)  # 这部分存储模型结构
    model.save_weights("basic_weights.h5")  # 这部分存储参数

sample2()
