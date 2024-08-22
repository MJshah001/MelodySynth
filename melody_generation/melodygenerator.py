"""
This module contains a class that generates melodies using the pretrained models.

The MelodyGenerator class uses pretrained deep learning models to generate melodies.
The model is trained on a dataset of melodies and is able to predict the next symbol in the melody given a sequence of previous symbols.
The model is trained using the preprocess.py module which preprocesses the dataset and creates a mapping of symbols to integers.
The MelodyGenerator class uses this mapping to encode and decode the symbols.
The class has a method generate_melody that generates a melody given a seed, number of steps, maximum sequence length, and temperature. 
The seed is a string of symbols that the melody generation process starts with.
The number of steps is the number of symbols to be generated.
The maximum sequence length is the number of previous symbols to consider for the next symbol prediction.
The temperature is a parameter that controls the randomness in the prediction.
The higher the temperature the more random the prediction.
The class also has a method save_melody that converts a melody into a MIDI file.


Example:
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    number_of_steps = 500
    max_sequence_length = 32
    temperature = 0.7
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    mg.save_melody(melody)

"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras as keras
import json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.preprocess import SEQUENCE_LENGTH
from model_training.train import LEARNING_RATE,SAVE_MODEL_DIR_PATH
import music21 as m21
import datetime
MAPPING_PATH = "../data_preprocessing/mapping.json"

class MelodyGenerator:
    """
    This class generates melodies using a pretrained model.

    """

    def __init__(self,selected_model = "LSTM", learning_rate=LEARNING_RATE):
        """
        :param selected_model (str): name of the model to be loaded default is LSTM
        :param learning_rate (float): learning rate for the optimizer ( if model was trained with different tensor flow version for gpu )

        """
        model_path = os.path.join(SAVE_MODEL_DIR_PATH,f"{selected_model}_model.h5")
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.load_and_modify_model()
        self.model = keras.models.load_model(model_path)
        print(f"Model {selected_model} loaded successfully")

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def load_and_modify_model(self):
        # Load the model without compiling
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        # Change the optimizer settings
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

        # Save the model again
        self.model.save(self.model_path)
        print(f"Model {self.model_path} loaded and modified successfully")

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """
        Generate a melody using the pretrained deep learning model.

        :param seed (str): seed to start the melody generation process e.g. "55 _ _ _"
        :param num_steps (int): number of steps to be generated
        :param max_sequence_length (int): number of previous steps to consider for the next step
        :param temperature (float): randomness in the prediction. The higher the temperature the more random the prediction

        :return melody (list of str): a list of symbols representing the melody

        """

        # creating a seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # mapping seed to integers
        seed = [self._mappings[symbol] for symbol in seed]
        for i in range(num_steps):

            # limitng seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]    ## two dimensions to 3 dimensions( 1st one batch size 1)

            # making the prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # updating seed
            seed.append(output_int)
            
            # mapping integer back to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # checking whether we are at end of melody
            if output_symbol == "/":
                break

            # updating the melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """
        Samples an index from a probability array reinterpreting it using a temperature
        :param probabilites (ndarray): an array of probabilities
        :param temperature (float): randomness in the prediction. The higher the temperature the more random the prediction
        :return index (int): the selected index
            
        """ 

        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) 
        index = np.random.choice(choices, p=probabilites)

        return index



    def save_melody(self, melody, step_duration =0.25 , format="midi", file_name = "generated_melody"):
        """
        Converts a melody into a MIDI file
        :param melody (list of str): a list of symbols representing the melody
        :param step_duration (float): duration of each time step in quarter length
        :param format (str): format of the file to save
        :param file_name (str): name of the file to save
            
        """
        # Creating a timestamp and appending it to the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name}_{timestamp}.mid"

        # Define the path to the 'generated melodies' folder
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_melodies")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, file_name)

        # creating a music21 stream object
        stream = m21.stream.Stream()

        # parsing all symbols from melody to create note/rest objects
        # eg  : 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handling case for note/rest
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
        stream.write(format,full_path)
        print(f"Melody saved as {file_name}")



if __name__ == "__main__":

    # create a list of model names by iterating over SAVE_MODEL_DIR_PATH directory
    model_names = [model_name.split("_")[0] for model_name in os.listdir(SAVE_MODEL_DIR_PATH)]

    #get index of a model from the list from user and swith case to select the model

    print("Select a model from the following list:")
    for index,model_name in enumerate(model_names):
        print(f"{index+1}. {model_name}")
    
    try:
        selected_model_index = int(input("Enter the index of the model you want to use: "))
        if selected_model_index > 0 and selected_model_index < (len(model_names)+1):
            selected_model = model_names[selected_model_index-1]
            mg = MelodyGenerator(selected_model)
            seed = "55 _ 60 _ 62 _ 64 _ 65 _ 67 _ _ _ 69 _ 65 _ 64 _ _ _ 62 _ _ _ 60 _ _ _ r _"
            number_of_steps = 500
            temperature = 0.7
            print(f"Generating melody using {selected_model} model with seed: {seed} and temperature: {temperature}...")
            melody = mg.generate_melody(seed=seed, num_steps=number_of_steps,max_sequence_length=SEQUENCE_LENGTH, temperature=temperature)
            mg.save_melody(melody)
        else:
            print("Invalid model index. Please enter a valid index.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    

    






