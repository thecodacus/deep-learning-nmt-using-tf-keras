import json
import numpy as np
from seq2seq import Seq2Seq
import pickle
data_path = 'Dataset/Bible/Bengali/combined.json'
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
input_character_index={}
target_character_index={}

lines = open(data_path, 'r', encoding='utf-8').read().split('\n')

epochs=500
num_samples=65021020

for line in lines[: min(num_samples, len(lines) - 1)]:
    line=line.strip('\n')
    line = json.loads(line)
    input_text = line['source']
    target_text = line['target']
    target_text = '\t' + target_text + '\n'

    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)

    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

input_character_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_character_index = dict([(char, i) for i, char in enumerate(target_characters)])

input_bits = len(input_characters)
output_bits = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, input_bits), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, output_bits), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, output_bits), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_character_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_character_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_character_index[char]] = 1.



parameters=[input_characters,target_characters,input_character_index,target_character_index,input_bits,output_bits,
            max_encoder_seq_length,max_decoder_seq_length]
pickle.dump(parameters,open('parameters.pkl', 'wb'))
s2s=Seq2Seq(max_decoder_seq_length, target_character_index['\t'], input_bits=input_bits, output_bits=output_bits)
s2s.trainModel(encoder_input_data,decoder_input_data,decoder_target_data,batch_size=100,epochs=epochs )
