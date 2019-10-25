
import tensorflow as tf
import matplotlib.pyplot as plt
#plt.imshow(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(x_test[0])

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits from 0-9
(x_train,y_train),(x_test,y_test) = mnist.load_data()  # distibution labels of input data

x_train = tf.keras.utils.normalize(x_train,axis=1) # normalization of training data
x_test = tf.keras.utils.normalize(x_test,axis=1)  # normalization of testing data

model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten())  
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) 


model.compile (optimizer='adam',
               loss="sparse_categorical_crossentropy",
               metrics =['accuracy']
               )

model.fit(x_train,y_train,epochs=3)

val_loss, val_accuracy = model.evaluate(x_test,y_test)
print(val_loss,val_accuracy)
