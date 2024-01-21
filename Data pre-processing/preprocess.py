import os
import music21 as m21
from music21 import environment


KERN_DATASET_PATH = "deutschl/test"

def load_songs_in_kern(dataset_path):

    songs = []

    # reading all files in dataset and loading them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

                






def preprocess(dataset_path):

    # loading Folk Songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    # filtering out songs that have non-acceptable durations

    # transposing Songs to Cmaj/Amin

    # encodeing songs with music time series representation

    # save songs to text file






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

    env = environment.Environment()
    env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
    env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    song.show()
