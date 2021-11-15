from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import statistics
from keras.utils import plot_model

import Gloable_Var
import Models

def CNN_model():
    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    # index_Array = np.arange(180)
    index_Array = index_Array[0:10000].astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)


    num_pixels = X_Buff.shape[1]
    n_out = 2
    print(X_Buff.shape)
    X_Buff = X_Buff[:, :,:,np.newaxis]
    print(X_Buff.shape)
    L_Max = 0.8 * max(Y_Buff[:,0])
    C_Max = 0.8 * max(Y_Buff[:,1])

    Y_Nor = np.zeros(np.shape(Y_Buff))
    Y_Nor[:,0] = Y_Buff[:,0]/L_Max
    Y_Nor[:,1] = Y_Buff[:,1]/C_Max


    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.33, random_state=42)

    Model_name = "CNN_2"
    Training_ite = Model_name+"_Try_3_larger"
    model = Models.CNN_model_2(num_pixels, n_out)
    plot_model( model, Model_name+".png", show_shapes=True )

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=500, verbose=2)
    np.save(Training_ite+".npy", history.history)

    scores = model.evaluate(X_test, Y_test, verbose=0)

def Baseline_model():
    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    # index_Array = np.arange(180)
    index_Array = index_Array[0:10000].astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)


    num_pixels = X_Buff.shape[1]
    n_out = 2
    num_pixels = X_Buff.shape[1] * X_Buff.shape[2]
    X_Buff = X_Buff.reshape((X_Buff.shape[0], num_pixels)).astype('float32')

    L_Max = 0.8 * max(Y_Buff[:,0])
    C_Max = 0.8 * max(Y_Buff[:,1])

    Y_Nor = np.zeros(np.shape(Y_Buff))
    Y_Nor[:,0] = Y_Buff[:,0]/L_Max
    Y_Nor[:,1] = Y_Buff[:,1]/C_Max


    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.1, random_state=4642)

    Model_name = "Baseline_2"
    Training_ite = Model_name+"_Try_1"
    model = Models.baseline_model(num_pixels, n_out)
    plot_model( model, Model_name+".png", show_shapes=True )

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=500, verbose=2)
    np.save(Training_ite+".npy", history.history)

    scores = model.evaluate(X_test, Y_test, verbose=0)


Baseline_model()