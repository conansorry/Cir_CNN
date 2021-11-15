from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import math
from keras.utils import plot_model
import matplotlib.pyplot as plt
import Gloable_Var
import Models

def plt_his(history, name):

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+".png")
    plt.show()

n_max =20000
name = "test3"


def CNN_Model():
    index_Array = np.loadtxt(Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    # index_Array = np.arange(180)
    index_Array = index_Array[0:n_max].astype(int)

    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)
    num_pixels = X_Buff.shape[1]

    n_out = 2
    # print(X_Buff.shape)
    X_Buff = X_Buff[:, :, :, np.newaxis]

    print(X_Buff.shape)
    L_Max = 0.8 * max(Y_Buff[:, 0])
    C_Max = 0.8 * max(Y_Buff[:, 1])

    Y_Buff = np.zeros(np.shape(Y_Buff))
    Y_Buff[:, 0] = Y_Buff[:, 0] / L_Max
    Y_Buff[:, 1] = Y_Buff[:, 1] / C_Max

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.2, random_state=422)

    model = Models.CNN_model_2(num_pixels, n_out)
    model.summary()
    plot_model(model, "CNN_2.png", show_shapes=True)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=50, verbose=2)
    np.save("Res", history.history)
    scores = model.evaluate(X_test, Y_test, verbose=0)



def baseline_model():


    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    index_Array = index_Array[0:n_max].astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)
    # X_Buff = Load_data.up_Sampling_Data(X_Buff)

    num_pixels = X_Buff.shape[1]



    L_Max = 0.8 * max(Y_Buff[:,0])
    C_Max = 0.8 * max(Y_Buff[:,1])

    Y_Buff[:, 0] = Y_Buff[:, 0] / L_Max
    Y_Buff[:, 1] = Y_Buff[:, 1] / C_Max


    # all in line input
    n_input = num_pixels * num_pixels
    X_Buff = X_Buff.reshape((X_Buff.shape[0], n_input)).astype('float32')
    #
    n_out = 2

    # 0.125 inpput
    # X_Buff, n_input = Load_data.mini_Input(X_Buff)
    # Y_Buff, n_out = Load_data.modi_Y(Y_Buff)
    #
    # print( statistics.mean(Y_Nor[:,0]), statistics.mean(Y_Nor[:,1]) )

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.2, random_state=62)


    model = Models.baseline_model(n_input, n_out)
    model.summary()
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=50, verbose=2)
    np.save(name, history.history)
    plot_model(model, name + ".png", show_shapes=True)
    plt_his(history,name )




