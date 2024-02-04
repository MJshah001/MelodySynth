import tensorflow.keras as keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH,MAPPING_PATH

class MelodyGenerator():

    def __init__(self,model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        if self.model :
            print(" \n\n 1> model found at ",self.model_path)
            print("Expected insput shape ",self.model.input_shape)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH    

    def generate_melody(self, seed, num_steps,max_sequence_length,temperature):
        print(" \n\n 2> generate_melody called with  ", seed, num_steps,max_sequence_length,temperature)
        # creating a seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        print(" \n\n 3> seed initialised with : ",seed) 
        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
        print(" \n\n 4> seed mapped to : ",seed) 
        output_symbols_test = []
        print("\n len(self._mappings) : ",len(self._mappings), "which includes  ", self._mappings,"\n\n")
        for i in range(num_steps):
            print("\n\n --------------------------------------------------------\n\n")
            # limitng seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            if i % 50 == 0:
                print(f" \n\n seed for {i} th iteration is  : ",seed)
            # one-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings) )
            if i % 50 == 0:
                print(f" onhot seed original dimensions for {i}th iteration is  : ",onehot_seed)
            onehot_seed = onehot_seed[np.newaxis,...]    ## two dimensions to 3 dimensions( 1st one batch size 1)
            if i % 50 == 0:
                print(f" onhot seed for {i}th iteration is  : ",onehot_seed)
            # making the prediction
            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities,temperature)
            if i % 50 == 0:
                print(f" output_int for {i}th iteration is  : ",output_int)
            # updateing seed
            seed.append(output_int)
            
            # mapping int back to our encoding
            output_symbol = [k for k,v in self._mappings.items() if v == output_int] [0]

            output_symbols_test.append(output_symbol)

            if i % 50 == 0:
                print(f" output_symbol for {i}th iteration is  : ",output_symbol)
            # checking whether we are at end of melody
            if output_symbol == "/":
                break

            # updating the melody
            melody.append(output_symbol)
            if i % 50 == 0:
                print(f"melody after {i}th iteration is  : ",melody)            
        print("\n\n output_symbols_test : \n", output_symbols_test)
        return melody


    def _sample_with_temperature(self,probabilities,temperature):

        predictions = np.log(probabilities)/temperature
        probabilities = np.exp(predictions)/np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices,p=probabilities)

        return index



if __name__ == "__main__":
    
    mg = MelodyGenerator()
    # seed = "67 _ _ _ 67 _ _ _ 64 _ _ _ 60 _"
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print("\n\n\n\n\n--------------------->>>>>>>>>>\n",melody)







