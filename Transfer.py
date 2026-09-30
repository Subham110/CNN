import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Create the VGG16 model as a feature extractor.
conv_base = VGG16(
    weights='imagenet',  # Load pre-trained weights from the ImageNet dataset.
    include_top = False,
    input_shape=(150,150,3)
)

# Create a Sequential neural network model to stack layers.
model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Create a Sequential neural network model to stack layers.
conv_base.trainable = False

# Set the number of images processed in each training batch.
batch_size = 32

# Create an image data generator for training data with augmentation.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# Create a data generator for validation/test data with only rescaling.
test_datagen = ImageDataGenerator(rescale=1./255) 

# Load training images from a directory and generate batches.
train_generator = train_datagen.flow_from_directory(
        '/content/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
) 

 # Load Test images from a directory and generate batches.
validation_generator = test_datagen.flow_from_directory(
        '/content/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
)


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_generator,epochs=10,validation_data=validation_generator)


plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()


plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()