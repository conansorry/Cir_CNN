from keras.models import Sequential
from keras.layers import Dense, LeakyReLU

from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from Replication_Padding import ReplicationPadding2D
import keras

def baseline_model(in_dim, n_out):
    model = Sequential()
    model.add(Dense(in_dim, input_dim= in_dim, activation="sigmoid"))
    model.add(Dense(0.5*in_dim, activation="sigmoid"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(n_out, activation="sigmoid"))

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
    opt_ad = keras.optimizers.Adam()
    # losses = keras.losses.mean_absolute_error()
    model.compile(loss="MAE", optimizer=opt )

    return model




def CNN_model_1( n_pix, n_out):
    # create model
    model = Sequential()

    model.add(Conv2D(8, (1, 1), input_shape=(n_pix, n_pix, 1), activation='relu'))
    model.add(ReplicationPadding2D(padding=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='sigmoid'))

    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='sigmoid'))
    model.add(ReplicationPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(n_out, activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
    model.compile(loss='MAE', optimizer=opt)

    return model


def CNN_model_2( n_pix, n_out):
    # create model
    model = Sequential()

    model.add(UpSampling2D((2,2), input_shape=(n_pix, n_pix, 1)))
    model.add(ReplicationPadding2D(padding=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(ReplicationPadding2D(padding=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='sigmoid'))
    model.add(ReplicationPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    #
    # model.add(Conv2D(64, (3, 3), activation='sigmoid'))
    # model.add(ReplicationPadding2D(padding=(1, 1)))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(n_out, activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.Adam()
    model.compile(loss='MAE', optimizer=opt)

    return model


def CNN_model_ori( n_pix, n_out):
    # create model
    model = Sequential()
    model.add(Conv2D(16, (4, 4), input_shape=(n_pix, n_pix, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(n_out, activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
    model.compile(loss='MAE', optimizer=opt)

    return model