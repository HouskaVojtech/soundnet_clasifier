from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop,Adam

num_classes = 10
# y = softmax (Wx+b)

x = np.load("activations.npy")
y = np.load("labels.npy")

classifier = Sequential()
# TODO: input shape does not have to be this way
classifier.add(Dense(num_classes, activation='softmax',input_shape=(3328,)))

classifier.summary()

#Define optimization function and compile the model
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])

batch_size = 64
epochs = 20

history = classifier.fit(x_train,z_train, validation_data=(x_test,z_test), epochs=epochs,batch_size=batch_size)
