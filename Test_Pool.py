from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import math
from keras.utils import plot_model

import Gloable_Var
import Models

def mini_Input(X_Buff):
    half_num_pixels = int(X_Buff.shape[1] * 0.5 )
    n_input = int(half_num_pixels * (half_num_pixels + 1) *0.5)
    X_test = np.zeros((X_Buff.shape[0], n_input))


    for i in range(X_Buff.shape[0]):
        s_ind = 0
        for j in range(half_num_pixels):
            X_test[i, s_ind:s_ind + half_num_pixels - j] = X_Buff[i,j,j:half_num_pixels]
            s_ind += half_num_pixels - j
    return X_test, n_input

def modi_Y(Y_Buff):
    Y_test = np.zeros((Y_Buff.shape[0], 1))
    for i in range(Y_test.shape[0]):
        Y_test[i] = math.sqrt(Y_Buff[i,0]*Y_Buff[i,1])

    return Y_test, 1



def CNN_Model():
    index_Array = np.loadtxt(Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    # index_Array = np.arange(180)
    index_Array = index_Array.astype(int)

    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)
    num_pixels = X_Buff.shape[1]

    n_out = 2
    # print(X_Buff.shape)
    X_Buff = X_Buff[:, :, :, np.newaxis]

    print(X_Buff.shape)
    L_Max = 0.8 * max(Y_Buff[:, 0])
    C_Max = 0.8 * max(Y_Buff[:, 1])

    Y_Nor = np.zeros(np.shape(Y_Buff))
    Y_Nor[:, 0] = Y_Buff[:, 0] / L_Max
    Y_Nor[:, 1] = Y_Buff[:, 1] / C_Max

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.2, random_state=422)

    model = Models.CNN_model_2(num_pixels, n_out)
    model.summary()
    plot_model(model, "CNN_2.png", show_shapes=True)

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=50, verbose=2)

    scores = model.evaluate(X_test, Y_test, verbose=0)



def baseline_model():
    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    index_Array = index_Array.astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)
    X_Buff = Load_data.up_Sampling_Data(X_Buff)

    num_pixels = X_Buff.shape[1]



    L_Max = 0.8 * max(Y_Buff[:,0])
    C_Max = 0.8 * max(Y_Buff[:,1])
    Y_Nor = np.zeros(np.shape(Y_Buff))
    Y_Nor[:, 0] = Y_Buff[:, 0] / L_Max
    Y_Nor[:, 1] = Y_Buff[:, 1] / C_Max


    # all in line input
    # n_input = num_pixels * num_pixels
    # X_Buff = X_Buff.reshape((X_Buff.shape[0], n_input)).astype('float32')
    #
    n_out = 2

    # 0.125 inpput
    X_Buff, n_input = mini_Input(X_Buff)
    # Y_Nor, n_out = modi_Y(Y_Buff)
    #
    # print( statistics.mean(Y_Nor[:,0]), statistics.mean(Y_Nor[:,1]) )

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.33, random_state=42)


    model = Models.baseline_model(n_input, n_out)
    model.summary()
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=5, verbose=2)
    plot_model(model, "linear_R.png", show_shapes=True)
    scores = model.evaluate(X_test, Y_test, verbose=0)

