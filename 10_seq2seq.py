encoder_inputs  = Input(shape = (None, input_vocab_size))
encoder = LSTM(num_neurons, return_state = True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = (state_h, state_c)  # state_h是最后一次隐藏单元序列，state_c是最后时刻的记忆单元

decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequence = True, return_state = True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(output_vocab_size, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs = decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs)

