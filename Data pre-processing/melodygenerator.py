import tensorflow.keras as keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH,MAPPING_PATH
import music21 as m21

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        # if self.model :
            # print(" \n\n 1> model found at ",self.model_path)
            # print("Expected insput shape ",self.model.input_shape)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # print(" \n\n 2> generate_melody called with  ", seed, num_steps,max_sequence_length,temperature)
        # creating a seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        # print(" \n\n 3> seed initialised with : ",seed) 
        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        # print(" \n\n 4> seed mapped to : ",seed) 
        output_symbols_test = []
        # print("\n len(self._mappings) : ",len(self._mappings), "which includes  ", self._mappings,"\n\n")
        for i in range(num_steps):

            # limitng seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            # if i % 50 == 0:
            #     print(f" \n\n seed for {i} th iteration is  : ",seed)
            # one-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            # if i % 50 == 0:
            #     print(f" onhot seed original dimensions for {i}th iteration is  : ",onehot_seed)
            onehot_seed = onehot_seed[np.newaxis, ...]    ## two dimensions to 3 dimensions( 1st one batch size 1)
            # if i % 50 == 0:
            #     print(f" onhot seed for {i}th iteration is  : ",onehot_seed)
            # making the prediction
            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)
            # if i % 50 == 0:
            #     print(f" output_int for {i}th iteration is  : ",output_int)
            # updateing seed
            seed.append(output_int)
            
            # mapping int back to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            output_symbols_test.append(output_symbol)

            # if i % 50 == 0:
            #     print(f" output_symbol for {i}th iteration is  : ",output_symbol)
            # checking whether we are at end of melody
            if output_symbol == "/":
                break

            # updating the melody
            melody.append(output_symbol)
        #     if i % 50 == 0:
        #         print(f"melody after {i}th iteration is  : ",melody)            
        # print("\n\n output_symbols_test : \n", output_symbols_test)
        return melody


    def _sample_with_temperature(self, probabilites, temperature):

        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index



    def save_melody(self, melody, step_duration =0.25 , format="midi", file_name = "mel.mid"):

        # creating a music21 stream (object oriented)
        stream = m21.stream.Stream()

        # parsing all symbols from melody to create note/rest objects
        # eg  : 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handiling case for note/rest
            if symbol != "_" or i+1 == len(melody):
                
                # ensuring we are dealing with note/rest beyond the first one
                if start_symbol is not None :

                    quarter_length_duration = step_duration * step_counter   # 0.25 * 4 = 1

                    # handling rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handling note
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reseting the step counter
                    step_counter = 1

                start_symbol = symbol

            # handling case for prolongation sign "_"
            else :
                step_counter += 1

        # writting m21 stream to a midi file
        stream.write(format,file_name)



if __name__ == "__main__":
    
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    mg.save_melody(melody)






