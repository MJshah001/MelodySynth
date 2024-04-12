from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import tensorflow.keras as keras

OUTPUT_UNITS = 38 # number of symbols in our json file
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 1 #50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

# build lstm model
def build_model_lstm(output_units, num_units, loss, learning_rate):

    # creating model architecture
    ## using functional API approach  # can also be done with sequencial mode
    input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compiling model
    model.compile(loss=loss,
                #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                  optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics = ["accuracy"]
                  )

    model.summary()     #to print summary/information of each layer
    return model

# build bi lstm model
def build_model_bi_lstm(output_units, num_units, loss, learning_rate):
    
        # creating model architecture
        ## using functional API approach  # can also be done with sequencial mode
        input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
        x = keras.layers.Bidirectional(keras.layers.LSTM(num_units[0]))(input)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(output_units, activation="softmax")(x)
    
        model = keras.Model(input, output)
    
        # compiling model
        model.compile(loss=loss,
                    #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics = ["accuracy"]
                    )
    
        model.summary()     #to print summary/information of each layer
        return model

# build gru model
def build_model_gru(output_units, num_units, loss, learning_rate):
    
        # creating model architecture
        ## using functional API approach  # can also be done with sequencial mode
        input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
        x = keras.layers.GRU(num_units[0])(input)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(output_units, activation="softmax")(x)
    
        model = keras.Model(input, output)
    
        # compiling model
        model.compile(loss=loss,
                    #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics = ["accuracy"]
                    )
    
        model.summary()     #to print summary/information of each layer
        return model

# build bi gru model
def build_model_bi_gru(output_units, num_units, loss, learning_rate):
    
        # creating model architecture
        ## using functional API approach  # can also be done with sequencial mode
        input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
        x = keras.layers.Bidirectional(keras.layers.GRU(num_units[0]))(input)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(output_units, activation="softmax")(x)
    
        model = keras.Model(input, output)
    
        # compiling model
        model.compile(loss=loss,
                    #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics = ["accuracy"]
                    )
    
        model.summary()     #to print summary/information of each layer
        return model

# build simple rnn model
def build_model_rnn(output_units, num_units, loss, learning_rate):
    
        # creating model architecture
        ## using functional API approach  # can also be done with sequencial mode
        input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
        x = keras.layers.SimpleRNN(num_units[0])(input)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(output_units, activation="softmax")(x)
    
        model = keras.Model(input, output)
    
        # compiling model
        model.compile(loss=loss,
                    #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics = ["accuracy"]
                    )
    
        model.summary()     #to print summary/information of each layer
        return model

# build lstm and gru combined model
def build_model_lstm_gru(output_units, num_units, loss, learning_rate):
    
        # creating model architecture
        ## using functional API approach  # can also be done with sequencial mode
        input = keras.layers.Input(shape=(None, output_units))      # None for no limitation on length of sequence generation /limitless
        x = keras.layers.LSTM(num_units[0], return_sequences=True)(input)
        x = keras.layers.GRU(num_units[0])(x)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(output_units, activation="softmax")(x)
    
        model = keras.Model(input, output)
    
        # compiling model
        model.compile(loss=loss,
                    #   optimizer = keras.optimizers.Adam(lr=learning_rate),
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics = ["accuracy"]
                    )
    
        model.summary()     #to print summary/information of each layer
        return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generating training sequences

    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    print("\n\n\n--------     special ---------- \n\n ",len(inputs),"\n\n",len(targets),"\n\n --------------------------------")
    # building the network
    model = build_model_lstm(output_units, num_units, loss, learning_rate)

    # training the model
    model.fit(inputs,targets,epochs = EPOCHS, batch_size = BATCH_SIZE)

    # saving the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
   # train()
   # due to GPU constraints use google collab gpu to train the model 
   pass