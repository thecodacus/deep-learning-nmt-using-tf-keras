from seq2seq import Seq2Seq
import pickle
import numpy as np


[input_characters,target_characters,input_character_index,target_character_index,input_bits,output_bits,
            max_encoder_seq_length,max_decoder_seq_length]=pickle.load(open('parameters.pkl','rb'))
s2s=Seq2Seq(max_decoder_seq_length, target_character_index['\t'], input_bits=input_bits, output_bits=output_bits)

s2s.loadWeights()



while True:
    text=input('>>')
    encoder_input_data = np.zeros ( (1 , max_encoder_seq_length , input_bits) , dtype='float32' )
    for t, char in enumerate(text):
        encoder_input_data[0, t, input_character_index[char]] = 1.
    s2s.setInput(encoder_input_data)
    outsentense = ''

    out=s2s.predictNext();
    out_index=np.argmax(out)
    outChar=target_characters[out_index]
    outsentense+=outChar
    while str(outChar)!=str('\n') and len(outsentense)<max_decoder_seq_length:
        out = s2s.predictNext ( ) ;
        out_index = np.argmax ( out )
        outChar = target_characters[ out_index ]
        outsentense += outChar
    print(str(outsentense))

