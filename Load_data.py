import numpy as np
import Gloable_Var as Var

N_pack = Var.N_pack

def load_data_with_array(index_Array):
    Max = int((index_Array[-1]) / N_pack)
    print(index_Array[-1], N_pack)
    load_y_p = Var.data_p + "LC_fit\\"
    Buff_LC = np.array([])
    for i in range(Max + 1):
        lc_nm = "LC_P" + str(i) + ".txt"
        if (Buff_LC.size == 0):
            Buff_LC = np.loadtxt(load_y_p + lc_nm)
        else:
            Buff_LC = np.append(Buff_LC, np.loadtxt(load_y_p + lc_nm), axis=0)
        print(i, "in", Max )
    print("finish loading LC cir")
    load_x_p = Var.data_p + "Mat\\"

    x = np.array([])
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
        if(i%N_pack == 0):
            print(i, "in", range(len(index_Array)))
    print("print finish loading X")

    return x, y


