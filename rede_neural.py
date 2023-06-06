from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def rede_neural():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(9,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model