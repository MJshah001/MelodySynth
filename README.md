
# Monophonic Folk Melody Generation with Memory based Deep Learning Models

## Overview

Monophonic Folk Melody Generation with Memory based Deep Learning Models is a project that explores the use of various deep learning models to generate melodies. The project focuses on leveraging recurrent neural networks (RNNs) such as LSTM and GRU to learn patterns from a dataset of melodies and generate new musical sequences autonomously.

## Features

- Train various deep learning models on a dataset of Kern/MIDI format melodies.
- Generate new melodies based on learned patterns from the trained models.
- Choose from a selection of RNN architectures including LSTM, Bi-LSTM, Stacked-LSTM, GRU, Bi-GRU, and Simple RNN.
- Customize the melody generation process with parameters such as seed melody, number of steps, and temperature.
- Save generated melodies as MIDI files for further exploration and use.

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/MJshah001/MelodySynth.git
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Prepare data for model training by running the following command:

    ```
    python preprocess.py
    ```
Users can customize the following paths based on their project directory and structure if needed:

- `KERN_DATASET_PATH`: Path to the dataset.
- `SAVE_DIR`: Path to save the encoded songs.
- `SINGLE_FILE_DATASET`: Path to save the single file dataset.
- `MAPPING_PATH`: Path to save the mapping.
- `SEQUENCE_LENGTH`: Number of time steps to be considered for prediction.

These parameters can be modified directly in the script `preprocess.py` to suit specific requirements.


2. Train the model by running the following command:

    ```
    python train.py
    ```
Follow the prompts to select the desired RNN architecture and enter training parameters.
```
1. LSTM
2. Bi-LSTM
3. Stacked-LSTM
4. GRU
5. Bi-GRU
6. RNN
```

Users can customize the following parameters based on their needs and computational resources:

- `NUM_UNITS`: Number of units in the hidden layers.
- `LOSS`: Loss function for the model.
- `LEARNING_RATE`: Learning rate for the model.
- `EPOCHS`: Number of epochs to train the model.
- `BATCH_SIZE`: Batch size for the model.

These parameters can be modified directly in the script `train.py` to suit specific requirements.


3. Generate new melodies using these trained models by running the following command:

    ```
    python melodygenerator.py
    ```

### Customizing Melody Generation

Users can customize the melody generation process by adjusting the following parameters:

- `Seed`: The initial sequence of symbols to start the melody generation process. Users can define the seed according to their desired musical motif or starting theme.

- `Number of Steps`: Determines the length of the generated melody in terms of the number of symbols. Adjusting this parameter allows users to control the overall duration of the melody.

- `Maximum Sequence Length`: Defines the number of previous symbols considered for predicting the next symbol in the melody. Users can set this parameter to capture longer-term dependencies in the music generation process.

- `Temperature`: Controls the randomness in the prediction. A higher temperature leads to more random and diverse outputs, while a lower temperature results in more deterministic predictions.

These parameters can be modified directly in the script `melodygenerator.py` to suit specific requirements.

Example:

```python
seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
number_of_steps = 500  
max_sequence_length = 32  
temperature = 0.7  

melody = mg.generate_melody(seed, number_of_steps, max_sequence_length, temperature)
mg.save_melody(melody)
```




Acknowledgments
----------------
- This project was made possible by the Essen Folk Song Database (Essac), which provided the dataset used for training the melody generation models.
- Special thanks to the contributors of the Music21 library for providing tools for computer-aided musicology.
- This project was inspired by the work of researchers and developers in the field of music generation using deep learning.

