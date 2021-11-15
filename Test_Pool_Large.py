from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import math
from keras.utils import plot_model
import matplotlib.pyplot as plt
import Gloable_Var
import Models

def plt_his(history, name, log = 0):
    fig = plt.figure()
    ax = fig.add_subplot()
    if (log == 1):
        ax.set_yscale('log')
    # summarize history for loss

    ax.plot(history.history['loss'],)
    ax.plot(history.history['val_loss'])

    ax.set_title('model loss')

    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')

    plt.savefig(name+"_res.png")
    plt.show()

n_max =6000
name = "test1"


def CNN_Model():
    index_Array = np.loadtxt(Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    # index_Array = np.arange(180)
    index_Array = index_Array[0:n_max].astype(int)

    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)

    X_Buff = X_Buff[:, :, :, np.newaxis]

    L_Max = 1.0 * max(Y_Buff[:, 0])
    C_Max = 1.0 * max(Y_Buff[:, 1])
    Y_Buff[:, 0] = Y_Buff[:, 0] / L_Max
    Y_Buff[:, 1] = Y_Buff[:, 1] / C_Max

    X_Buff, Xf_test, Y_Buff, Yf_test = train_test_split(X_Buff, Y_Buff, test_size=1.0/6, random_state=3)

    num_pixels = X_Buff.shape[1]

    n_out = 2
    # print(X_Buff.shape)


    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.2, random_state=422)

    model = Models.CNN_model_1(num_pixels, n_out)
    model.summary()
    plot_model(model, name+".png", show_shapes=True)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=100, verbose=2)
    np.save(name, history.history)
    scores = model.evaluate(Xf_test, Yf_test, verbose=2)
    print(scores)
    Y_res = model.predict(Xf_test)
    plt.plot(Y_res[:,0], Yf_test[:,0], 'o')
    plt.plot(Y_res[:, 1], Yf_test[:, 1], 'o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(name+"_LC.png")
    model.save(name + ".h5")

    plt_his(history, name )


def baseline_model():


    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    index_Array = index_Array[0:n_max].astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)
    # X_Buff = Load_data.up_Sampling_Data(X_Buff)

    num_pixels = X_Buff.shape[1]



    L_Max = 1.0 * max(Y_Buff[:,0])
    C_Max = 1.0 * max(Y_Buff[:,1])

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
    X_Buff, Xf_test, Y_Buff, Yf_test = train_test_split(X_Buff, Y_Buff, test_size=1.0 / 6, random_state=43443)

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.2, random_state=62)


    model = Models.baseline_model(n_input, n_out)
    model.summary()
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=50, verbose=2)
    np.save(name, history.history)

    scores = model.evaluate(Xf_test, Yf_test, verbose=2)
    print(scores)
    Y_res = model.predict(Xf_test)
    plt.plot(Y_res[:, 0], Yf_test[:, 0], 'o')
    plt.plot(Y_res[:, 1], Yf_test[:, 1], 'o')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(name + "_LC.png")
    model.save(name + ".h5")

    plt_his(history,name )




