import numpy as np
import Gloable_Var as Var
import math

N_pack = Var.N_pack
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
def load_data_with_array(index_Array):
    Max = int((index_Array[-1]) / N_pack)
    print(index_Array[-1], N_pack)
    load_y_p = Var.data_p + "LC_fit\\"
    Buff_LC = np.array([])
    for i in range(Max + 1):
        print(i)
        lc_nm = "LC_P" + str(i) + ".txt"
        if (Buff_LC.size == 0):
            Buff_LC = np.loadtxt(load_y_p + lc_nm)
        else:
            Buff_LC = np.append(Buff_LC, np.loadtxt(load_y_p + lc_nm), axis=0)
        print(np.shape(Buff_LC))

    load_x_p = Var.data_p + "Mat\\"

    x = np.array([])
    print(x.size)
    y = np.array([])
    for i in range(len(index_Array)):
        x_nm = "Mat_" + str(index_Array[i]) + ".txt"
        if (np.size(x) == 0):
            x = [np.loadtxt(load_x_p + x_nm, delimiter=',')]
        else:
            x = np.append(x, [np.loadtxt(load_x_p + x_nm, delimiter=',')], axis=0)

        if (np.size(y) == 0):
            y = [Buff_LC[index_Array[i], 1:3]]
        else:
            y = np.append(y, [Buff_LC[index_Array[i], 1:3]], axis=0)

    print(np.shape(y))
    return x, y

def up_Sampling_Data(X_in):
    X_return = np.zeros((X_in.shape[0], X_in.shape[1]*2, X_in.shape[2]*2))

    for i in range(X_in.shape[0]):
        X_return[i] = Up_Sampling_Mat(X_in[i,:,:])

    return X_return

def Up_Sampling_Mat(X_mat):
    n_pix = X_mat.shape[1]
    _n = int(n_pix/2)
    X_return = np.zeros((2*n_pix, 2*n_pix))

    for i in range(_n):
        for j in range(_n):

            if (X_mat[i, j] == 0):
                X_return[2*i:2*(i+1), 2*j:2*(j+1)] = 1.0

            elif(X_mat[i, j] == 0.5):
                if (i != 0 and j != _n):
                    if (X_mat[i - 1, j] == 0 and X_mat[i, j + 1] == 0):
                        X_return[2*i, 2*j] = 1.0
                if (i != _n and j != _n):
                    if (X_mat[i, j + 1] == 0 and X_mat[i + 1, j] == 0):
                        X_return[2*i, 2*j+1] = 1.0
                if (i != _n and j != 0):
                    if (X_mat[i + 1, j] == 0 and X_mat[i, j - 1] == 0):
                        X_return[2*i+1, 2*j] = 1.0
                if (i != 0 and j != 0):
                    if (X_mat[i, j - 1] == 0 and X_mat[i - 1, j] == 0):
                        X_return[2*i+1, 2*j+1] = 1.0

    for i in range(2 * _n):
        for j in range( 2 * _n):
            X_return[2 * n_pix - i - 1, j] = X_return[i, j]
            X_return[i, 2 * n_pix -j- 1] = X_return[i, j]
            X_return[2 * n_pix - i- 1, 2 * n_pix -j- 1] = X_return[i, j]
    return X_return




