import tensforflow as tf 
import numpy as np 
import matplotlib as plt 

print('this is a test print')


#creating a simple nn
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
]) 

#choosing what loss function to optimize and what algorithm to use to do that
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

#setting the number of training loops
model.fit(x_train_flattened, y_train, epochs=5)
