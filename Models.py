from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import keras

def baseline_model(in_dim, n_out):
    model = Sequential()
    model.add(Dense(in_dim, input_dim= in_dim, activation="sigmoid"))
    model.add(Dense(0.5*in_dim, activation="sigmoid"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(n_out, activation="sigmoid"))

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
    # losses = keras.losses.mean_absolute_error()
    model.compile(loss="MAE", optimizer=opt )

    return model