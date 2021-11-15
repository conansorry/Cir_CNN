from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import statistics

import Gloable_Var
import Models



def baseline_model():
    index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
    index_Array = index_Array.astype(int)


    X_Buff, Y_Buff = Load_data.load_data_with_array(index_Array)


    num_pixels = X_Buff.shape[1] * X_Buff.shape[2]
    n_out = 2


    L_Max = 0.8 * max(Y_Buff[:,0])
    C_Max = 0.8 * max(Y_Buff[:,1])

    Y_Nor = np.zeros(np.shape(Y_Buff))
    Y_Nor[:,0] = Y_Buff[:,0]/L_Max
    Y_Nor[:,1] = Y_Buff[:,1]/C_Max
    num_pixels = X_Buff.shape[1] * X_Buff.shape[2]
    X_Buff = X_Buff.reshape((X_Buff.shape[0], num_pixels)).astype('float32')

    # print( statistics.mean(Y_Nor[:,0]), statistics.mean(Y_Nor[:,1]) )

    X_train, X_test, Y_train, Y_test = train_test_split(X_Buff, Y_Buff, test_size=0.33, random_state=42)


    model = Models.baseline_model(num_pixels, n_out)

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=5, verbose=2)

    scores = model.evaluate(X_test, Y_test, verbose=0)

