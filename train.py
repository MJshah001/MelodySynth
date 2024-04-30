"""
This script trains the model using the training sequences generated from the preprocessed data.
The model is trained using the training sequences and the corresponding targets.
The model is saved after training.
The training history is plotted and saved as an image.
The training time is printed.
The user can select the model to train from the available models by entering the index of the model.
The available models are:
1. LSTM
2. Bi-LSTM
3. Stacked-LSTM
4. GRU
5. Bi-GRU
6. RNN


To run the script, execute the following command:
python train.py

"""
import sys
sys.path.insert(0, '../data_preprocessing')
# from data_preprocessing.preprocess import generate_training_sequences, SEQUENCE_LENGTH
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import tensorflow.keras as keras
import tensorflow as tf
import timeit
import json
import sys

# Checking if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
    # Set TensorFlow to use GPU memory dynamically
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
else:
    print("GPU is not available")



NUM_UNITS = [256] # number of units in the hidden layers
LOSS = "sparse_categorical_crossentropy" # loss function for the model
LEARNING_RATE = 0.001 # learning rate for the model
EPOCHS = 50 # number of epochs to train the model
BATCH_SIZE = 64 # batch size for the model


MAPPING_PATH = "mapping.json" # path to the json file
with open(MAPPING_PATH, 'r') as f:
    mapping_data = json.load(f)
OUTPUT_UNITS = int(len(mapping_data))    # number of symbols in our json file will be the output units
print("Output Units:", OUTPUT_UNITS)
print("sequence length:", SEQUENCE_LENGTH)

def build_model_lstm(output_units, num_units, loss, learning_rate):
    """
    To Build the LSTM model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: LSTM model

    """

    ## using functional API approach  # can also be done with sequencial mode
    input = keras.layers.Input(shape=(None, output_units))    
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                  optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics = ["accuracy"]
                  )

    model.summary()     #to print summary/information of each layer
    return model

def build_model_stacked_lstm(output_units, num_units, loss, learning_rate):
    """
    To build the Stacked LSTM model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: Stacked LSTM model

    """

    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0], return_sequences=True)(input)  # return_sequences=True to pass the sequence to the next LSTM layer
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(num_units[1])(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()
    return model


def build_model_bi_lstm(output_units, num_units, loss, learning_rate):
    """
    To build the Bidirectional LSTM model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: Bidirectional LSTM model
    """

    input = keras.layers.Input(shape=(None, output_units))      
    x = keras.layers.Bidirectional(keras.layers.LSTM(num_units[0]))(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                metrics = ["accuracy"]
                )

    model.summary()    
    return model

def build_model_gru(output_units, num_units, loss, learning_rate):
    """
    To build the GRU model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: GRU model
    """

    input = keras.layers.Input(shape=(None, output_units))     
    x = keras.layers.GRU(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                metrics = ["accuracy"]
                )

    model.summary()    
    return model

def build_model_bi_gru(output_units, num_units, loss, learning_rate):
    """
    To build the Bidirectional GRU model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: Bidirectional GRU model
    """

    input = keras.layers.Input(shape=(None, output_units))     
    x = keras.layers.Bidirectional(keras.layers.GRU(num_units[0]))(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                metrics = ["accuracy"]
                )

    model.summary()    
    return model

def build_model_rnn(output_units, num_units, loss, learning_rate):
    """
    To build the RNN model architecture.
    :param output_units: int: Number of output units
    :param num_units: list: Number of units in the hidden layers
    :param loss: string: Loss function for the model
    :param learning_rate: float: Learning rate for the model
    :return: keras.Model: RNN model
    """

    input = keras.layers.Input(shape=(None, output_units))  
    x = keras.layers.SimpleRNN(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                metrics = ["accuracy"]
                )

    model.summary()    
    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE,model_name="LSTM",validation_split=0.1):
    """
    To train the model.
    :param output_units: int: Number of output units.
    :param num_units: list: Number of units in the hidden layers.
    :param loss: string: Loss function for the model.
    :param learning_rate: float: Learning rate for the model.
    :param model_name: string: Name of the model.
    :param validation_split: float: Fraction of the data to be used as validation data.
    :return: history: keras.callbacks.History: Training history of the model.
    :return: time_taken_minutes: float: Time taken to train the model in minutes.
    """

    # generating training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # splitting the data into training and validation sets
    from sklearn.model_selection import train_test_split
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=validation_split, random_state=42)
    
    # Delete inputs and targets arrays to free up memory
    del inputs
    del targets
   
    # building the network
    if model_name == 'LSTM':
      SAVE_MODEL_PATH = "lstm_model.h5"
      model = build_model_lstm(output_units, num_units, loss, learning_rate)

    ## to retrain the same model for where we left of
    # model = keras.models.load_model(SAVE_MODEL_PATH)

    if model_name == 'Bi-LSTM':
      SAVE_MODEL_PATH = "bi_lstm_model.h5"
      model = build_model_bi_lstm(output_units, num_units, loss, learning_rate)

    if model_name == 'Stacked-LSTM':
      print("training", model_name,"\n")
      SAVE_MODEL_PATH = "stacked_lstm_model.h5"
      model = build_model_stacked_lstm(output_units, num_units, loss, learning_rate)

    if model_name == 'GRU':
      SAVE_MODEL_PATH = "gru_model.h5"
      model = build_model_gru(output_units, num_units, loss, learning_rate)

    if model_name == 'Bi-GRU':
      SAVE_MODEL_PATH = "bi_gru_model.h5"
      model = build_model_bi_gru(output_units, num_units, loss, learning_rate)

    if model_name == 'RNN':
      SAVE_MODEL_PATH = "simple_rnn_model.h5"
      model = build_model_rnn(output_units, num_units, loss, learning_rate)

    # training the model
    start_time = timeit.default_timer()
    history = model.fit(inputs_train,targets_train,epochs = EPOCHS, batch_size = BATCH_SIZE,validation_data=(inputs_val, targets_val))
    end_time = timeit.default_timer()
    time_taken_minutes = (end_time - start_time) / 60.0
    
    # Deleting inputs and targets arrays to free up memory
    del inputs_train
    del targets_train

    # saving the model
    model.save(SAVE_MODEL_PATH)
    
    return history,time_taken_minutes

def plot_training_history(history, model_name):
    """
    To plot the training history of the model.
    :param history: keras.callbacks.History: Training history of the model.
    :param model_name: string: Name of the model.
    """

    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(model_name + ' Learning Curve')
    plt.legend()
    
    # Save the plot as an image
    image_path = f"{model_name}_learning_curve.png"
    plt.savefig(image_path)
    # print("Learning curve picture saved at:", image_path)
    
    plt.show()

if __name__ == "__main__":
   
   # Available Models
   model_names = ['LSTM','Bi-LSTM','Stacked-LSTM','GRU', 'Bi-GRU','RNN']

   print("Available models:")
   for i, name in enumerate(model_names):
      print(f"{i + 1}. {name}")

   model_index = input("Enter the index of the model you want to use: ")

   try:
        model_index = int(model_index)        
        if 1 <= model_index <= len(model_names):
            # Get the selected model name
            model_name = model_names[model_index - 1]
            print(f"Training {model_name} model... \n")
            
            # Proceed with training using the selected model
            history, time_taken_minutes = train(model_name=model_name)

            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]

            # print("Final Loss:", final_loss)
            # print("Final Accuracy:", final_accuracy)
            # print("Time taken for training:", time_taken_minutes, "minutes")

            plot_training_history(history, model_name)

        else:
            print("Invalid model index. Please enter a valid index.")
            sys.exit(1)
   
   except ValueError:
        print("Invalid input. Please enter a valid integer index.")
        sys.exit(1)
