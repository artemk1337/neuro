# BEST 120 e 15
seq = Sequential()
seq.add(Conv1D(5, 3, input_shape=input_shape, activation='relu'))
seq.add(MaxPool1D(2))
seq.add(Conv1D(7, 3, input_shape=input_shape, activation='relu'))
seq.add(MaxPool1D(2))
seq.add(Conv1D(9, 3, input_shape=input_shape, activation='relu'))
seq.add(MaxPool1D(2))
seq.add(Conv1D(12, 3, input_shape=input_shape, activation='relu'))
seq.add(Flatten())
seq.add(Dense(16, activation='sigmoid'))

