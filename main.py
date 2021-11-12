from sklearn.model_selection import train_test_split

import Load_data
import numpy as np
import statistics
from keras.utils import plot_model

import Gloable_Var
import Models

index_Array = np.loadtxt( Gloable_Var.data_p + "\LC_fit\Good_Index.txt")
# index_Array = np.arange(180)
index_Array = index_Array.astype(int)


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


model = Models.CNN_model_2(num_pixels, n_out)
plot_model( model, "CNN_2.png", show_shapes=True )

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=10, verbose=2)

scores = model.evaluate(X_test, Y_test, verbose=0)

