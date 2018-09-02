import numpy as np
from view_hex_data import show_board, show_move, show_values

data = np.load("hex_data.npz")
values = data["values"]
states = data["states"].reshape(values.shape[0], values.shape[1], values.shape[2], 1)
visits = data["visits"]
turns = data["turns"]
moves = data["moves"]

print("states:", states.shape)
print("values:", values.shape)
print("visits:", visits.shape)
print("turns:", turns.shape)
print("moves:", moves.shape)

index = np.random.randint(states.shape[0])
print(show_move(states[index], moves[index], turns[index]))
print(show_values(states[index], values[index]))

states[turns==-1, :, :, :] = np.transpose((states[turns==-1, :, :, :]*-1), axes=[0,2,1,3])

for i in range(len(states)):
  if turns[i] == -1:
    temp = moves[i][0]
    moves[i][0] = moves[i][1]
    moves[i][1] = temp

#Split into training and test sets
train_X = states[:4*states.shape[0] // 5]
test_X = states[4*states.shape[0] // 5:]

print("train_X:", train_X.shape)
print("test_X:", test_X.shape)

train_Y = moves[:4*moves.shape[0] // 5]
test_Y = moves[4*moves.shape[0] // 5:]

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
neural_net = Sequential()

neural_net.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(8, 8, 1)))
neural_net.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1),
                 activation='relu',
                 input_shape=(8, 8, 1)))
neural_net.add(Flatten())
neural_net.add(Dense(2, activation='relu'))

neural_net.summary()

neural_net.compile(loss="mean_squared_error",
              optimizer="Adam",
              metrics=['accuracy'])
history = neural_net.fit(train_X, train_Y,
          epochs=12,
          verbose=1,
          validation_data=(test_X, test_Y))

loss, accuracy = neural_net.evaluate(test_X, test_Y, verbose=0)
print("accuracy: {}%".format(accuracy*100))
