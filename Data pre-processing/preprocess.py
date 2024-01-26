import os
import json
import music21 as m21
from music21 import environment



KERN_DATASET_PATH = "deutschl/test"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

#Durations are in Quarter Lenght
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
    Loads all kern pices in datasset using music21

    """

    songs = []

    # reading all files in dataset and loading them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    """
    Boolean that returns True if piece has all acceptable duration, False Otherwise

    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """
    Transposes song to C major / A minor.
    """
    # getting key from the song         # if it is stored
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]          # if not we will estimate as follows

    # estimating key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    print("Original Key : ", key)
    # getting interval got transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic , m21.pitch.Pitch("C"))

    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic , m21.pitch.Pitch("A"))

    # transposing song by calculated interval
    transposed_song = song.transpose(interval)
    

    return transposed_song


def encode_song(song, time_step = 0.25):
    # p= 60 , d=1.0 -> ["60", "_","_","_" ]
    encoded_song = []

    for event in song.flat.notesAndRests:

        #handling notes
        if isinstance(event , m21.note.Note):
            symbol = event.pitch.midi #60
        #handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # converting note/rest into timeseries notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0 :
                encoded_song.append(symbol)
            else :
                encoded_song.append("_")

    # casting encoded song to a string

    encoded_song = " ".join(map(str,encoded_song))

    return encoded_song






def preprocess(dataset_path):

    # loading Folk Songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):

        # filtering out songs that have non-acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue

        # transposing Songs to Cmaj/Amin
        song = transpose(song)

        # encodeing songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR,str(i))
        with open (save_path,"w") as fp:
            fp.write(encoded_song)



def load(file_path):
    with open(file_path,"r") as fp:
            song = fp.read() 
    return song

def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length):
    new_song_delimeter = "/ " * sequence_length
    songs = ""

    #loading encoded songs and adding delimeter
    for path , _ , files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimeter
    
    songs = songs [:-1]

    #saving the whole strinng to the file
    with open (file_dataset_path, "w") as fp :
        fp.write(songs)

    return songs


def create_mapping(songs,mapping_path):
    """
    creates a json file that maps symbols of song into integers.
    """
    mappings = {}

    # identifying the Vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    
    # creating mapping
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # saving vocabulary to json
    with open(mapping_path, "w") as fp:
        json.dump(mappings,fp, indent=4)


def convert_song_to_int(songs):
    int_songs = []

    #loading mappings
    with open(SINGLE_FILE_DATASET,"r") as fp:
        mappings = json.load(fp)
    
    #casting songs string to a list
    songs = songs.split()

    #map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

# def generate_training_sequences(sequence_length):
#     # [11,12,13,14,...] -> i:[11,12], t:13; i:[12,13], t:14




def main():

    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)


if __name__ == "__main__":
    
    # env = m21.environment.Environment()
    # print('Environment settings:')
    # print('musicXML:  ', env['musicxmlPath'])
    # # C:\Users\Monil Shah\Documents\GitHub\Melody-Generation-with-RNN---LSTM\Data pre-processing\C:\Program Files\MuseScore 3\bin\MuseScore3.exe
    # print('musescore: ', env['musescoreDirectPNGPath'])
    # # C:\Users\Monil Shah\Documents\GitHub\Melody-Generation-with-RNN---LSTM\Data pre-processing\C:\Program Files\ 3\bin\MuseScore3.exe
    
    # env = m21.environment.Environment(forcePlatform='darwin')
    # us = m21.environment.UserSettings()
    # print(us.keys())
    # us['musicxmlPath'] = '/Applications/Finale Reader.app'
    # us['musicxmlPath']

    # env['musescoreDirectPNGPath'] = 'C:/Users/Monil Shah/Documents/GitHub/Melody-Generation-with-RNN---LSTM/Data pre-processing/C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    # env['musicxmlPath'] = 'C:/Users/Monil Shah/Documents/GitHub/Melody-Generation-with-RNN---LSTM/Data pre-processing/C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

    # env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    # env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'


    # print('musicXML:  ', env['musicxmlPath'])
    # print('musescore: ', env['musescoreDirectPNGPath'])

    # setting up enviornment for MuseScore 4
    env = environment.Environment()
    env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    main()
######
    # songs = load_songs_in_kern(KERN_DATASET_PATH)
    # print(f"Loaded {len(songs)} songs.")
    # song = songs[0]

    # print(f"Has acceptable duration ? { has_acceptable_durations(song,ACCEPTABLE_DURATIONS)}")
    # print("\n\n ------------------------- \n")
    
    # preprocess(KERN_DATASET_PATH)

  
    # transposed_song = transpose(song)

    # song.show()           # original song
    # transposed_song.show()


    # songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)

    # create_mapping(songs,MAPPING_PATH)

    #######
