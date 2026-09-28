import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as np 

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# 5. Train model
history = model.fit(x_train, y_train, epochs=2, batch_size=64,validation_data=(x_test, y_test))

# 6. Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


seven_index = np.where(y_test == 8)[0][0] 
single_image = x_test[seven_index]
actual_label = y_test[seven_index]

# Reshape the single image for prediction (add batch dimension)
single_image_processed = single_image.reshape(1, 28, 28, 1)

# Make a prediction
prediction = model.predict(single_image_processed)
predicted_label = np.argmax(prediction)

plt.imshow(single_image.reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()
