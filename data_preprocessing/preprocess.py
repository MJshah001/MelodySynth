"""
This script preprocesses the dataset and prepares it for training.

It does the following:
1. Load the data from the data source.
2. Filter out songs that have non-acceptable durations
3. Transpose songs to a common key
4. Encode songs with music time series representation
5. Save songs to text file
6. Create a single file dataset
7. Create a mapping for the dataset
8. Generate training sequences

- The single file dataset is a string that contains all the songs in the dataset separated by a delimiter.
- The mapping is a json file that maps symbols of the songs to integers.
- The training sequences are one-hot encoded sequences that will be used to train the model.
- The single file dataset and mapping are saved to the disk to be used later for training the model.
- The training sequences are returned by the script and can be used to train the model.
- The script assumes that the dataset is in the kern format. The kern format is a symbolic music notation format that represents music as text.

The script also assumes that the dataset is stored in a directory with the following structure:
datasource/
    datasetname/
        composer1/
            song1.krn
            song2.krn
            ...
        composer2/
            song1.krn
            song2.krn
            ...
    ...         # and so on

usage:
    python preprocess.py

"""

import os
import json
import music21 as m21
from music21 import environment
import numpy as np
import tensorflow.keras as keras


KERN_DATASET_PATH = "../datasource/deutschl/erk" # Path to dataset
SAVE_DIR = "dataset"  # Path to save the encoded songs
print("current path",os.getcwd())
SINGLE_FILE_DATASET = "file_dataset" # Path to save the single file dataset
MAPPING_PATH = "mapping.json" # Path to save the mapping
SEQUENCE_LENGTH = 64    # Number of time steps to be considered for prediction

#Durations are in Quarter Length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5,  # 8th note
    0.75,
    1.0,  # quarter note
    1.5,
    2,    # Half note
    3,
    4     # full note 
]

def load_songs_in_kern(dataset_path):
    """ 
    Load all kern pieces in dataset using music21.
    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    
    """
    songs = []

    # reading all files in dataset and loading them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    """
    Boolean that returns True if piece has all acceptable duration, False Otherwise
    :param song (m21 stream): Piece to check
    :param acceptable_durations (list of floats): Acceptable durations
    :return (bool): True if all durations are acceptable, False Otherwise
    """
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """
    Transposes song to C major / A minor.
    :param song (m21 stream): Piece to transpose
    :return transposed_song (m21 stream): Transposed piece
    """
    # getting key from the song         # if it is stored
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]         # if not we will estimate as follows

    # estimating key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # print("Original Key : ", key)
    # getting interval got transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transposing song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song


def encode_song(song, time_step=0.25):
    """
    Convert a score into a time-series format
    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter Length
    :return encoded_song (str): String representing the encoded song
    """
    # p= 60 , d=1.0 -> ["60", "_","_","_" ]
    encoded_song = []

    for event in song.flatten().notesAndRests:

        # handling notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # converting note/rest into timeseries notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # casting encoded song to a string
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


def preprocess(dataset_path):
    """
    Encodes dataset and saves it to a json file
    :param dataset_path (str): Path to dataset
    """
    print("\n\n dataset_path",dataset_path,"\n\n")
    # loading Folk Songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    print("Preprocessing songs...")
    # delete SAVEDIR and its content if already exists and then create new
    if os.path.exists(SAVE_DIR):
        os.system("rmdir /s /q " + SAVE_DIR)
    os.mkdir(SAVE_DIR)

    for i, song in enumerate(songs):

        # filtering out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue # if not acceptable then skip

        # transposing Songs to Cmaj/Amin
        song = transpose(song)

        # encodeing songs with music time series representation
        encoded_song = encode_song(song)


        # save encoded songs to dataset folder
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    """
    Load encoded song from path
    :param file_path (str): Path to file
    :return song (str): String representing the song
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of preprocess.py
    full_file_path = os.path.join(base_dir, file_path)
    with open(full_file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """
    Create a single file dataset from all the encoded songs and add a delimiter between each song
    :param dataset_path (str): Path to dataset
    :param file_dataset_path (str): Path to file dataset
    :param sequence_length (int): Number of time steps to consider for prediction
    :return songs (str): String containing all dataset
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    #loading encoded songs and adding delimeter
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # removing empty space from last character of string
    songs = songs[:-1]

    #saving the whole string to the file
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs



def create_mapping(songs,mapping_path):
    """
    creates a json file that maps symbols of song into integers.
    :param songs (str): String containing all songs
    :param mapping_path (str): Path to save mapping

    """
    mappings = {}

    # identifying the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # creating mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # Delete mapping_path if already exists
    if os.path.exists(mapping_path):
        os.system("del /f " + mapping_path)

    # saving voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)



def convert_songs_to_int(songs):
    """
    Convert songs to a list of integers
    :param songs (str): String containing all songs
    :return int_songs (list of int): List of int representing the songs
    """

    int_songs = []

    # loading mappings
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of preprocess.py
    full_mapping_path = os.path.join(base_dir, MAPPING_PATH)
    with open(full_mapping_path, "r") as fp:
        mappings = json.load(fp)

    # transforming songs string to list
    songs = songs.split()

    # maping songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    """
    Generate training sequences from the dataset
    :param sequence_length (int): Number of time steps to consider for prediction
    :return inputs (np.array): Training sequences
    :return targets (np.array): Target sequence
    """
    # [1,2,3,4,...] -> input:[1,2],target:3 ;  input:[2,3], target:14
    # loading songs and mapping them to int
    songs = load(SINGLE_FILE_DATASET)

    int_songs = convert_songs_to_int(songs)

    # generating the training sequences
    # eg : 100 symbols, 64 is sequence length , 100 - 64 = 36

    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

   
    # one-hot encoding the sequences
    vocabulary_size = len(set(int_songs))

    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets





def main():
    preprocess(KERN_DATASET_PATH)
    print("Creating single file dataset...")
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    print("Creating mapping...")
    create_mapping(songs, MAPPING_PATH)
    # print("Creating training sequences...")
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    # print("inputs.shape",inputs.shape)
    # print("targets.shape",targets.shape)
    print("\n\nPreprocessing completed.")


if __name__ == "__main__":
    
    # setting up enviornment for MuseScore 4
    env = environment.Environment()
    env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    main()
      


## Note: Other possible errors and their solutions

# >>>>>> for tensorflow error run following in cmd
# set TF_ENABLE_ONEDNN_OPTS=0
# python preprocess.py

# >>>>> for WARNING:tensorflow: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.       
# pip install --upgrade keras

# >>>>>> for music21 error run following in cmd
# pip install --upgrade music21


# >>>>> for music21.environment.EnvironmentException: Cannot find a valid application path for MuseScore. Please add it to your PATH or set it manually with environment.set('musescoreDirectPNGPath', '/path/to/your/musescore').
# TO set the path to MuseScore in the above code use below given steps

# Step1 : Option 1 : ( to print the path of MuseScore 4)
# env = m21.environment.Environment()
# print('Environment settings:')
# print('musicXML:  ', env['musicxmlPath'])
# # C:\Users\Monil Shah\Documents\GitHub\Melody-Generation-with-RNN---LSTM\Data pre-processing\C:\Program Files\MuseScore 3\bin\MuseScore3.exe
# print('musescore: ', env['musescoreDirectPNGPath'])
# # C:\Users\Monil Shah\Documents\GitHub\Melody-Generation-with-RNN---LSTM\Data pre-processing\C:\Program Files\ 3\bin\MuseScore3.exe

# Step1: Option 2: (to print the path of MuseScore 4)
# env = m21.environment.Environment(forcePlatform='darwin')
# us = m21.environment.UserSettings()
# print(us.keys())
# us['musicxmlPath'] = '/Applications/Finale Reader.app'
# us['musicxmlPath']

# Step2 : (to set the path of MuseScore 4)
# env['musescoreDirectPNGPath'] = 'C:/Users/Monil Shah/Documents/GitHub/Melody-Generation-with-RNN---LSTM/Data pre-processing/C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
# env['musicxmlPath'] = 'C:/Users/Monil Shah/Documents/GitHub/Melody-Generation-with-RNN---LSTM/Data pre-processing/C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

# env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
# env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

# print('musicXML:  ', env['musicxmlPath'])
# print('musescore: ', env['musescoreDirectPNGPath'])



