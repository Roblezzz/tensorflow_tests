import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer])

model.compile(
	optimizer=tf.keras.optimizers.Adam(0.1),
 loss='mean_squared_error'
)

print("training the model")
history= model.fit(celsius, fahrenheit, epochs=1000)
print("model trained")

 
plt.xlabel("Loop")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])

result = model.predict([25])

print(str(result))

 