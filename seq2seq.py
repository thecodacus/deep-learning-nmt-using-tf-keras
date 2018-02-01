import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
import numpy as np


class Seq2Seq:
    def __init__(self,decode_max_len,start_seq_index,input_bits,output_bits):

        #creating the model
        self.input_bits = input_bits
        self.output_bits=output_bits
        self.encoded_bits = 256
        self.maxlen = decode_max_len
        self.start_seq_index=start_seq_index

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.input_bits))
        encoder = LSTM(self.encoded_bits, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.output_bits))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.encoded_bits, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.output_bits, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.trainingModel = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.trainingModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # Inference setup:

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.encoded_bits,))
        decoder_state_input_c = Input(shape=(self.encoded_bits,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


    def trainModel(self, encoder_input_data,decoder_input_data, decoder_target_data,batch_size=10, epochs=1):
        self.trainingModel.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size,
                               epochs=epochs)
        self.trainingModel.save('s2s.h5')

    def loadWeights(self):
        self.trainingModel.load_weights('s2s.h5')

    def setInput(self,input_data):
        # Encode the input as state vectors.
        self.states_value = self.encoder_model.predict ( input_data )
        self.target_seq = np.zeros ( (1 , 1 , self.output_bits) )
        self.target_seq[ 0 , 0 , self.start_seq_index ] = 1

    def predictNext(self):

        output, h, c = self.decoder_model.predict([self.target_seq] + self.states_value)

        # Sample a token
        output_index = np.argmax(output[0, -1, :])

        self.target_seq = np.zeros((1, 1, self.output_bits))
        self.target_seq[0, 0, output_index] = 1

        # Update states
        self.states_value = [h, c]
        return output[0, -1, :]
